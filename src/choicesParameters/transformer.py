import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class PositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model

    def forward(self, x):
        if x.ndim != 3 or x.size(-1) != self.d_model:
            raise ValueError("x must have shape [batch, sequence_length, d_model]")
        seq_len = x.size(1)
        position = torch.arange(seq_len, device=x.device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=x.device, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        angles = position * div_term
        pe = torch.zeros((seq_len, self.d_model), device=x.device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles[:, :self.d_model // 2])
        return x + pe.unsqueeze(0).to(dtype=x.dtype)


class TransformerModel(nn.Module):
    def __init__(self, num_output, d_model=128, nhead=8, num_layers=2, device="cpu"):
        super().__init__()
        for name, value in (("num_output", num_output), ("d_model", d_model),
                            ("nhead", nhead), ("num_layers", num_layers)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.d_model = d_model
        self.device = torch.device(device)
        self.input_linear = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_layer = nn.Linear(d_model, num_output)
        self.to(self.device)

    def forward(self, src, src_key_padding_mask=None):
        if src.ndim != 2 or src.size(1) == 0:
            raise ValueError("src must have shape [batch, sequence_length] and length > 0")
        if not torch.is_floating_point(src):
            src = src.float()
        if not torch.isfinite(src).all().item():
            raise ValueError("src contains NaN or Inf")
        if src_key_padding_mask is not None:
            if (src_key_padding_mask.shape != src.shape
                    or src_key_padding_mask.dtype != torch.bool):
                raise ValueError("The mask must be bool and have the same shape as src")
            if src_key_padding_mask.device != src.device:
                raise ValueError("The mask and src must be on the same device")
            if src_key_padding_mask.all(dim=1).any().item():
                raise ValueError("Completely masked sequences are not allowed")

        encoded = self.pos_encoder(self.input_linear(src.unsqueeze(-1)))
        output = self.transformer_encoder(
            encoded, src_key_padding_mask=src_key_padding_mask
        )
        if src_key_padding_mask is None:
            pooled = output.mean(dim=1)
        else:
            valid = (~src_key_padding_mask).unsqueeze(-1)
            counts = valid.sum(dim=1).clamp_min(1)
            pooled = output.masked_fill(~valid, 0.0).sum(dim=1) / counts
        return self.output_layer(pooled)


class CustomDataset(Dataset):
    def __init__(self, input_vectors, target_vectors=None, device="cpu"):
        self.input_vectors = input_vectors
        self.target_vectors = target_vectors
        self.device = torch.device(device)
        if target_vectors is not None and len(input_vectors) != len(target_vectors):
            raise ValueError("input_vectors and target_vectors must have the same number of examples")

    def __len__(self):
        return len(self.input_vectors)

    def __getitem__(self, idx):
        input_vector = torch.as_tensor(
            self.input_vectors[idx], dtype=torch.float32, device=self.device
        )
        if input_vector.ndim != 1 or input_vector.numel() == 0:
            raise ValueError(f"Input {idx} must be a non-empty 1D vector")
        if not torch.isfinite(input_vector).all().item():
            raise ValueError(f"Input {idx} contains NaN or Inf")
        if self.target_vectors is None:
            return input_vector
        target_vector = torch.as_tensor(
            self.target_vectors[idx], dtype=torch.float32, device=self.device
        )
        if target_vector.ndim != 1 or target_vector.numel() == 0:
            raise ValueError(f"Target {idx} must be a non-empty 1D vector")
        if not torch.isfinite(target_vector).all().item():
            raise ValueError(f"Target {idx} contains NaN or Inf")
        return input_vector, target_vector


def collate_fn(batch):
    if not batch:
        raise ValueError("The batch cannot be empty")
    if isinstance(batch[0], tuple):
        inputs, targets = zip(*batch)
        targets = torch.stack(targets)
    else:
        inputs = batch
        targets = None

    lengths = torch.tensor([len(seq) for seq in inputs], device=inputs[0].device)
    if (lengths == 0).any().item():
        raise ValueError("Empty sequences are not allowed")
    max_len = int(lengths.max().item())
    padded_inputs = torch.stack([
        nn.functional.pad(seq, (0, max_len - len(seq))) for seq in inputs
    ])
    src_key_padding_mask = (
        torch.arange(max_len, device=padded_inputs.device).unsqueeze(0)
        >= lengths.unsqueeze(1)
    )
    if targets is not None:
        return padded_inputs, targets, src_key_padding_mask
    return padded_inputs, src_key_padding_mask


def train(model, input_vectors, target_vectors, num_epochs=10, batch_size=32, learning_rate=0.001):
    if target_vectors is None:
        raise ValueError("target_vectors is required for training")
    if len(input_vectors) == 0:
        raise ValueError("At least one example is required for training")
    if not isinstance(num_epochs, int) or isinstance(num_epochs, bool) or num_epochs <= 0:
        raise ValueError("num_epochs must be a positive integer")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")

    device = next(model.parameters()).device
    dataset = CustomDataset(input_vectors, target_vectors, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    criterion = nn.MSELoss()

    optimizer = getattr(model, "_training_optimizer", None)
    if optimizer is None or (
        {id(p) for group in optimizer.param_groups for p in group["params"]}
        != {id(p) for p in model.parameters()}
    ):
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model._training_optimizer = optimizer
    else:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    for epoch in range(num_epochs):
        model.train()
        total_squared_error = 0.0
        total_values = 0
        for inputs, targets, src_key_padding_mask in dataloader:
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, src_key_padding_mask=src_key_padding_mask)
            if outputs.shape != targets.shape:
                raise ValueError(
                    f"Output {tuple(outputs.shape)} and "
                    f"target {tuple(targets.shape)} do not match"
                )
            loss = criterion(outputs, targets)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Loss became non-finite")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            total_squared_error += loss.item() * targets.numel()
            total_values += targets.numel()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_squared_error / total_values:.4f}")


def evaluate(model, input_vectors, target_vectors=None, batch_size=32):
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if len(input_vectors) == 0:
        raise ValueError("At least one example is required for evaluation")
    device = next(model.parameters()).device
    dataset = CustomDataset(input_vectors, target_vectors, device=device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    model.eval()
    predictions = []
    squared_error = 0.0
    total_values = 0
    with torch.inference_mode():
        for batch in dataloader:
            if target_vectors is not None:
                inputs, targets, mask = batch
            else:
                inputs, mask = batch
            outputs = model(inputs, src_key_padding_mask=mask)
            if target_vectors is not None:
                if outputs.shape != targets.shape:
                    raise ValueError(
                        f"Output dimensions {tuple(outputs.shape)} and "
                        f"target {tuple(targets.shape)} do not match"
                    )
                squared_error += ((outputs - targets) ** 2).sum().item()
                total_values += targets.numel()
            predictions.extend(outputs.cpu().numpy())
    predictions = np.array(predictions)
    if target_vectors is not None:
        mse = np.float64(squared_error / total_values)
        print(f"MSE: {mse:.4f}")
        return predictions, mse
    return predictions
