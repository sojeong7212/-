import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Dataset class to handle image sequences
class ImageSequenceDataset(Dataset):
    def __init__(self, data_paths, H_prev=20, H_post=10, transform=None):
        self.data = []
        self.H_prev = H_prev
        self.H_post = H_post
        self.transform = transform

        # Load data and create sequences
        for path in data_paths:
            sequence = np.load(path)  # shape: (T, H, W)
            T = sequence.shape[0]
            # Generate input-output pairs
            for i in range(T - H_prev - H_post + 1):
                x = sequence[i:i+H_prev]
                y = sequence[i+H_prev:i+H_prev+H_post]
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # Convert to tensors
        x = torch.from_numpy(x).float().unsqueeze(1)  # (H_prev, 1, H, W)
        y = torch.from_numpy(y).float().unsqueeze(1)  # (H_post, 1, H, W)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


# ConvLSTM Cell Definition
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,  # Include h_cur
            out_channels=4 * self.hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # Use both h_cur and c_cur

        # Concatenate input_tensor and h_cur along channel dimension
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1
        )

        i = torch.sigmoid(cc_i)   # Input gate
        f = torch.sigmoid(cc_f)   # Forget gate
        o = torch.sigmoid(cc_o)   # Output gate
        g = torch.tanh(cc_g)      # Cell gate

        c_next = f * c_cur + i * g  # Next cell state
        h_next = o * torch.tanh(c_next)  # Next hidden state

        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

# ConvLSTM Module
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()

        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()

        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim

            self.cell_list.append(ConvLSTMCell(
                input_dim=cur_input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                bias=bias
            ))

    def forward(self, input_tensor, hidden_state=None):
        # input_tensor: (batch, seq_len, channels, height, width)
        b, seq_len, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, spatial_size=(h, w))

        layer_output_list = []
        last_state_list = []

        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []

            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](
                    input_tensor=cur_layer_input[:, t, :, :, :],
                    cur_state=[h, c]
                )
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, spatial_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, spatial_size))
        return init_states

# Attention Mechanism
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, attn_dim):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_dim = attn_dim

        self.attn_conv = nn.Conv2d(hidden_dim * 2, attn_dim, kernel_size=1)
        self.attn_conv2 = nn.Conv2d(attn_dim, 1, kernel_size=1)

    def forward(self, hidden_state, encoder_outputs):
        # hidden_state: (batch, hidden_dim, H, W)
        # encoder_outputs: (batch, seq_len, hidden_dim, H, W)
        seq_len = encoder_outputs.size(1)
        batch_size, hidden_dim, H, W = hidden_state.size()

        # Repeat hidden_state seq_len times
        hidden_state_expanded = hidden_state.unsqueeze(1).repeat(1, seq_len, 1, 1, 1)
        # Concatenate hidden_state with encoder_outputs
        attn_input = torch.cat((hidden_state_expanded, encoder_outputs), dim=2)  # (batch, seq_len, hidden_dim*2, H, W)
        attn_input = attn_input.view(batch_size * seq_len, hidden_dim * 2, H, W)

        # Compute attention scores
        energy = torch.tanh(self.attn_conv(attn_input))
        energy = self.attn_conv2(energy)  # (batch * seq_len, 1, H, W)
        energy = energy.view(batch_size, seq_len, -1)  # (batch, seq_len, H * W)
        energy = energy.mean(dim=2)  # (batch, seq_len)

        # Compute attention weights
        attn_weights = F.softmax(energy, dim=1)  # (batch, seq_len)

        # Compute context vector
        attn_weights = attn_weights.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # (batch, seq_len, 1, 1, 1)
        context = (attn_weights * encoder_outputs).sum(dim=1)  # (batch, hidden_dim, H, W)

        return context

# Encoder-Decoder ConvLSTM with Attention
class EncoderDecoderConvLSTMWithAttention(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=1, attn_dim=32):
        super(EncoderDecoderConvLSTMWithAttention, self).__init__()

        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers
        )

        self.decoder_cell = ConvLSTMCell(
            input_dim=input_dim + hidden_dim,  # Adjusted input_dim to include concatenated input
            hidden_dim=hidden_dim,
            kernel_size=kernel_size
        )

        self.attention = AttentionBlock(hidden_dim, attn_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
            nn.Sigmoid()  # Assuming output pixel values are between 0 and 1
        )

    def forward(self, input_tensor, target_len):
        # input_tensor: (batch, seq_len, channels, height, width)
        b, seq_len, _, h, w = input_tensor.size()

        # Encoder
        encoder_outputs_list, encoder_states = self.encoder(input_tensor)
        encoder_hidden = encoder_outputs_list[-1]  # From the last layer
        # encoder_hidden: (batch, seq_len, hidden_dim, H, W)

        # Initialize decoder state with the last encoder state
        h, c = encoder_states[-1]

        # Prepare encoder outputs for attention
        encoder_outputs = encoder_hidden  # (batch, seq_len, hidden_dim, H, W)

        # Decoder
        decoder_input = input_tensor[:, -1, :, :, :]  # Last input frame
        outputs = []

        for t in range(target_len):
            # Attention
            context = self.attention(h, encoder_outputs)

            # Combine context with decoder input
            decoder_input_combined = torch.cat([decoder_input, context], dim=1)  # (batch, input_dim + hidden_dim, H, W)

            # Decoder step
            h, c = self.decoder_cell(decoder_input_combined, [h, c])

            # Generate output frame
            frame = self.conv(h)
            outputs.append(frame.unsqueeze(1))  # (batch, 1, channels, H, W)

            # Prepare next input
            decoder_input = frame  # Keep channels dimension

        outputs = torch.cat(outputs, dim=1)  # (batch, target_len, channels, H, W)
        return outputs

# Training setup with model saving
def train_model(model, dataloader, num_epochs=10, lr=1e-3, save_path='./trained_models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # To keep track of the loss per epoch
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            output = model(x, target_len=y.size(1))
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Save the trained model
    model_file = os.path.join(save_path, 'conv_lstm_attention_model.pth')
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


if __name__ == '__main__':
    # Set list of data paths
    NUM_DATA = 50
    IDX_TEST = [0]
    data_paths = []
    for i in range(NUM_DATA):
        if i in IDX_TEST:
            continue
        data_paths.append("./data/imgs/img_{:d}.npy".format(i))

    dataset = ImageSequenceDataset(data_paths, H_prev=20, H_post=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize and train the model
    model = EncoderDecoderConvLSTMWithAttention()
    train_model(model, dataloader)
