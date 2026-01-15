import torch
import torch.nn as nn
from torch.utils.data import Dataset 
import math

from src.models.sda_blocks import SDAMultiheadAttention, SDATransformerDecoderLayer, SDATransformerEncoderLayer, TransformerEncoder, TransformerDecoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==== Dataset ====
class TimeSeriesDataset(Dataset):
    def __init__(self, data, patch_length, horizon):
        self.X, self.y = self.create_sequences(data, patch_length, horizon)

    def create_sequences(self, data, patch_length, horizon):
        X, y = [], []
        for i in range(len(data) - patch_length - horizon):
            X.append(data[i : i + patch_length])
            y.append(data[i + patch_length : i + patch_length + horizon])
        return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==== Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# ==== Series Decomposition ====
class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        trend = self.moving_avg(x.transpose(1, 2)).transpose(1, 2)
        seasonal = x - trend
        return trend, seasonal


# ==== Autoformer Variants ====
class SDAStandardAutoformer(nn.Module):
    def __init__(self, patch_length, horizon, d_model=8, num_heads=2):
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size=3)
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=32, batch_first=True
            ),
            num_layers=2,
        )
        self.fc_seasonal = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(patch_length, horizon)

    def forward(self, x):
        trend, seasonal = self.decomposition(x)
        seasonal = self.embedding(seasonal)
        seasonal = self.pos_encoder(seasonal)
        seasonal = self.encoder(seasonal)
        seasonal_out = self.fc_seasonal(seasonal.mean(dim=1))
        trend_out = self.fc_trend(trend.squeeze(-1))
        return seasonal_out + trend_out


class SDAFullAutoformer(nn.Module):
    def __init__(self, patch_length, horizon, d_model=8, num_heads=2):
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size=25)
        self.enc_embedding = nn.Linear(1, d_model)
        self.dec_embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_decoder = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(
            SDATransformerEncoderLayer(d_model, num_heads), num_layers=2
        )
        self.decoder = TransformerDecoder(
            SDATransformerDecoderLayer(d_model, num_heads), num_layers=1
        )
        self.projection = nn.Linear(d_model, 1)
        self.horizon = horizon

    def forward(self, x):
        trend, seasonal = self.decomposition(x)
        enc_in = self.pos_encoder(self.enc_embedding(seasonal))
        enc_out = self.encoder(enc_in.transpose(0, 1)).transpose(0, 1)
        B = x.size(0)
        dec_input = torch.zeros(B, self.horizon, 1).to(x.device)
        dec_in = self.pos_decoder(self.dec_embedding(dec_input))
        dec_out = self.decoder(
            dec_in.transpose(0, 1), enc_out.transpose(0, 1)
        ).transpose(0, 1)
        return self.projection(dec_out).squeeze(-1)


class SDAMinimalAutoformer(nn.Module):
    def __init__(self, patch_length, horizon, d_model=8, num_heads=2):
        super().__init__()
        self.decomposition = SeriesDecomposition(kernel_size=3)
        self.embedding = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = SDATransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=32, batch_first=True
        )
        self.encoder = TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_seasonal = nn.Linear(d_model, horizon)
        self.fc_trend = nn.Linear(patch_length, horizon)

    def forward(self, x):
        trend, seasonal = self.decomposition(x)
        seasonal = self.embedding(seasonal)
        seasonal = self.pos_encoder(seasonal)
        seasonal = self.encoder(seasonal)
        seasonal_out = self.fc_seasonal(seasonal.mean(dim=1))
        trend_out = self.fc_trend(trend.squeeze(-1))
        return seasonal_out + trend_out
