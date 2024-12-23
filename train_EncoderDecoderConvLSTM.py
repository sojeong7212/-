import os
import numpy as np
import torch
import torch.nn as nn
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

# ConvLSTM cell definition
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=input_dim + hidden_dim,
            out_channels=4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, combined_conv.size(1) // 4, dim=1
        )
        i = torch.sigmoid(cc_i)   # input gate
        f = torch.sigmoid(cc_f)   # forget gate
        o = torch.sigmoid(cc_o)   # output gate
        g = torch.tanh(cc_g)      # cell gate
        c_next = f * c_cur + i * g  # next cell state
        h_next = o * torch.tanh(c_next)  # next hidden state
        return h_next, c_next

    def init_hidden(self, batch_size, spatial_size):
        height, width = spatial_size
        device = next(self.parameters()).device
        return (
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
            torch.zeros(batch_size, self.hidden_dim, height, width, device=device)
        )

# ConvLSTM module
class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, bias=True):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.cell_list = nn.ModuleList()
        for i in range(num_layers):
            cur_input_dim = input_dim if i == 0 else hidden_dim
            self.cell_list.append(
                ConvLSTMCell(
                    input_dim=cur_input_dim,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    bias=bias
                )
            )

    def forward(self, input_tensor, hidden_state=None):
        # input_tensor: (batch, seq_len, channels, height, width)
        batch_size, seq_len, _, height, width = input_tensor.size()
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))

        layer_output = []
        h, c = hidden_state
        for t in range(seq_len):
            h, c = self.cell_list[0](
                input_tensor=input_tensor[:, t, :, :, :], cur_state=[h, c]
            )
            layer_output.append(h)
        layer_output = torch.stack(layer_output, dim=1)
        return layer_output, (h, c)

    def _init_hidden(self, batch_size, spatial_size):
        return self.cell_list[0].init_hidden(batch_size, spatial_size)

# Encoder-Decoder model using ConvLSTM
class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, kernel_size=3, num_layers=1):
        super(EncoderDecoderConvLSTM, self).__init__()
        self.encoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers
        )
        self.decoder = ConvLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers
        )
        self.conv = nn.Conv2d(hidden_dim, input_dim, kernel_size=1)

    def forward(self, input_tensor, target_len):
        batch_size, seq_len, channels, height, width = input_tensor.size()
        # Encode the input sequence
        encoder_output, encoder_state = self.encoder(input_tensor)
        h, c = encoder_state
        decoder_input = input_tensor[:, -1, :, :, :].unsqueeze(1)  # Last frame
        outputs = []
        # Decode the future sequence
        for _ in range(target_len):
            decoder_output, (h, c) = self.decoder(decoder_input, (h, c))
            frame = self.conv(decoder_output[:, -1, :, :, :])
            outputs.append(frame.unsqueeze(1))
            decoder_input = frame.unsqueeze(1)
        outputs = torch.cat(outputs, dim=1)
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
    model_file = os.path.join(save_path, 'conv_lstm_model.pth')
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
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize and train the model
    model = EncoderDecoderConvLSTM()
    train_model(model, dataloader)