import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class LidarSequenceDataset(Dataset):
    def __init__(self, data_paths, H_prev=20, H_post=10, transform=None):
        self.data = []
        self.H_prev = H_prev
        self.H_post = H_post
        self.transform = transform

        # Load data and create sequences
        for path in data_paths:
            sequence = np.load(path)  # shape: (T, 720)
            T = sequence.shape[0]
            # Generate input-output pairs
            for i in range(T - H_prev - H_post + 1):
                x = sequence[i:i+H_prev]      # Input sequence of length H_prev
                y = sequence[i+H_prev:i+H_prev+H_post]  # Target sequence of length H_post
                self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        # Convert to tensors
        x = torch.from_numpy(x).float()  # Shape: (H_prev, 720)
        y = torch.from_numpy(y).float()  # Shape: (H_post, 720)
        if self.transform:
            x = self.transform(x)
            y = self.transform(y)
        return x, y


class LidarTransformerModel(nn.Module):
    def __init__(self, input_dim=720, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(LidarTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input linear transformation to match d_model
        self.input_fc = nn.Linear(input_dim, d_model)

        # Positional encodings
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # Transformer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout,
                                          batch_first=True)  # Set batch_first=True

        # Output linear transformation back to input_dim
        self.output_fc = nn.Linear(d_model, input_dim)

    def forward(self, src, tgt):
        """
        src: (batch_size, H_prev, input_dim)
        tgt: (batch_size, H_post - 1, input_dim)
        """
        src = self.input_fc(src) * np.sqrt(self.d_model)  # Scale the embeddings
        tgt = self.input_fc(tgt) * np.sqrt(self.d_model)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(src, tgt)

        output = self.output_fc(output)  # (batch_size, H_post - 1, input_dim)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: (batch_size, sequence_length, d_model)
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def train_model(model, dataloader, num_epochs=10, lr=1e-4, save_path='./trained_models'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x, y in dataloader:
            x = x.to(device)  # x: (batch_size, H_prev, 720)
            y = y.to(device)  # y: (batch_size, H_post, 720)

            # Prepare input and target sequences
            y_input = y[:, :-1, :]  # y_input: (batch_size, H_post - 1, 720)
            y_target = y[:, 1:, :]  # y_target: (batch_size, H_post - 1, 720)

            optimizer.zero_grad()
            output = model(x, y_input)  # output: (batch_size, H_post - 1, 720)
            loss = criterion(output, y_target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.6f}")

    # Save the trained model
    model_file = os.path.join(save_path, 'lidar_transformer_model.pth')
    torch.save(model.state_dict(), model_file)
    print(f"Model saved to {model_file}")


if __name__ == '__main__':
    # Set list of data paths
    NUM_DATA = 50
    IDX_TEST = [0]  # Indexes reserved for testing
    data_paths = []
    for i in range(NUM_DATA):
        if i in IDX_TEST:
            continue
        data_paths.append("./data/dists/dist_{:d}.npy".format(i))

    dataset = LidarSequenceDataset(data_paths, H_prev=20, H_post=10)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = LidarTransformerModel(
        input_dim=720,       # Each LIDAR scan has 720 distance measurements
        d_model=512,         # Embedding dimension
        nhead=8,             # Number of attention heads
        num_encoder_layers=6,
        num_decoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1
    )
    train_model(model, dataloader, num_epochs=20, lr=1e-4)
