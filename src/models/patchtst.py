
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.signal import square, sawtooth
import math
import itertools

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Positional Encoding (Sinusoidal) ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


# ==== PatchTST Variants ====
class PatchTSTMinimal(nn.Module):
    def __init__(
        self, patch_length, horizon, d_model, num_heads, dim_feedforward, num_layers
    ):
        super().__init__()
        self.embedding = nn.Linear(patch_length, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class PatchTSTStandard(nn.Module):
    def __init__(
        self, patch_length, horizon, d_model, num_heads, dim_feedforward, num_layers
    ):
        super().__init__()
        self.embedding = nn.Linear(patch_length, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x)


class PatchTSTFull(nn.Module):
    def __init__(
        self,
        patch_length,
        horizon,
        d_model,
        num_heads,
        dim_feedforward,
        num_encoder_layers,
    ):
        super().__init__()
        self.encoder_input_proj = nn.Linear(patch_length, d_model)
        self.decoder_input_proj = nn.Linear(patch_length, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, dim_feedforward, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, dim_feedforward, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.fc = nn.Linear(d_model, 1)
        self.horizon = horizon

    def forward(self, x):
        enc = self.encoder_input_proj(x)
        enc = self.pos_encoder(enc)
        enc = self.encoder(enc)
        dec_input = x[:, -1:, :].repeat(1, self.horizon, 1)
        dec = self.decoder_input_proj(dec_input)
        dec = self.pos_decoder(dec)
        dec = self.decoder(dec, enc)
        return self.fc(dec).squeeze(-1)
