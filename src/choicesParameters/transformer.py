import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

    def forward(self, x):
        seq_len = x.size(1)
        device = x.device
        position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        x = x + pe
        return x

class TransformerModel(nn.Module):
    def __init__(self, num_output, d_model=128, nhead=8, num_layers=2):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.input_linear = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.output_layer = nn.Linear(d_model, num_output)

    def forward(self, src, src_key_padding_mask=None):
        src = src.unsqueeze(-1)
        src = self.input_linear(src)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = output.mean(dim=0)
        output = self.output_layer(output)
        return output

class CustomDataset(Dataset):
    def __init__(self, input_vectors, target_vectors=None):
        self.input_vectors = input_vectors
        self.target_vectors = target_vectors

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        input_vector = torch.tensor(self.input_vectors[idx], dtype=torch.float32)
        if self.target_vectors is not None:
            target_vector = torch.tensor(self.target_vectors[idx], dtype=torch.float32)
            return input_vector, target_vector
        else:
            return input_vector

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        inputs, targets = zip(*batch)
        targets = torch.stack(targets)
    else:
        inputs = batch
        targets = None

    max_len = max([len(seq) for seq in inputs])

    padded_inputs = [
        torch.nn.functional.pad(seq, (0, max_len - len(seq))) for seq in inputs
    ]
    padded_inputs = torch.stack(padded_inputs)

    src_key_padding_mask = padded_inputs == 0
    if targets is not None:
        return padded_inputs, targets, src_key_padding_mask
    else:
        return padded_inputs, src_key_padding_mask

def train(model, input_vectors, target_vectors, num_epochs=10, batch_size=32, learning_rate=0.001):
    dataset = CustomDataset(input_vectors, target_vectors)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets, src_key_padding_mask in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")


def evaluate(model, input_vectors, target_vectors=None, batch_size=32):
    dataset = CustomDataset(input_vectors, target_vectors)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model.eval()
    predictions = []
    true_targets = []
    with torch.no_grad():
        for batch in dataloader:
            if target_vectors is not None:
                inputs, targets, src_key_padding_mask = batch
                true_targets.extend(targets.numpy())
            else:
                inputs, src_key_padding_mask = batch
            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            predictions.extend(outputs.numpy())

    predictions = np.array(predictions)
    if target_vectors is not None:
        true_targets = np.array(true_targets)
        mse = np.mean((predictions - true_targets) ** 2)
        print(f"MSE: {mse:.4f}")
        return predictions, mse
    else:
        return predictions