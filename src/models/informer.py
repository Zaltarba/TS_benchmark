
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import math
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==== Positional Encoding ====
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)].to(x.device)


# ==== Informer Variants ====
class InformerMinimal(nn.Module):
    def __init__(self, patch_length, d_model=8, num_heads=2, horizon=6):
        super().__init__()
        self.embedding = nn.Linear(patch_length, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=32
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x.transpose(1, 2))
        x = self.pos_encoding(x.transpose(1, 2))
        x = self.encoder(x.transpose(0, 1)).transpose(0, 1)
        x = x.mean(dim=1)
        return self.fc_out(x)


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, L, _ = x.size()
        Q = self.q_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k**0.5
        u = min(L, max(1, int(torch.log(torch.tensor(L, dtype=torch.float32)).item())))
        top_u = torch.topk(scores.max(dim=-1).values, u, dim=-1).indices

        sparse_output = []
        for b in range(B):
            per_batch = []
            for h in range(self.n_heads):
                idx = top_u[b, h]
                q = Q[b, h][idx]
                attn = torch.matmul(q, K[b, h].transpose(0, 1)) / self.d_k**0.5
                attn_weights = torch.softmax(attn, dim=-1)
                context = torch.matmul(attn_weights, V[b, h])
                padded_context = torch.zeros((L, self.d_k), device=x.device)
                padded_context[idx] = context
                per_batch.append(padded_context)
            sparse_output.append(torch.cat(per_batch, dim=-1))
        output = torch.stack(sparse_output, dim=0)
        return self.out_proj(output)


class ProbSparseTransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class InformerStandard(nn.Module):
    def __init__(self, patch_length, d_model=8, num_heads=2, horizon=6):
        super().__init__()
        self.embedding = nn.Linear(patch_length, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [
                ProbSparseTransformerLayer(d_model, num_heads, dim_feedforward=32)
                for _ in range(2)
            ]
        )
        self.fc_out = nn.Linear(d_model, horizon)

    def forward(self, x):
        x = self.embedding(x.transpose(1, 2))
        x = self.pos_encoding(x.transpose(1, 2))
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.fc_out(x)


class FullInformer(nn.Module):
    def __init__(
        self,
        patch_length,
        horizon,
        d_model=8,
        n_heads=2,
        d_ff=32,
        e_layers=2,
        d_layers=1,
    ):
        super().__init__()
        self.enc_embedding = nn.Linear(1, d_model)
        self.dec_embedding = nn.Linear(1, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder = nn.ModuleList(
            [
                ProbSparseTransformerLayer(d_model, n_heads, d_ff)
                for _ in range(e_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                ProbSparseTransformerLayer(d_model, n_heads, d_ff)
                for _ in range(d_layers)
            ]
        )
        self.projection = nn.Linear(d_model, 1)
        self.horizon = horizon
        self.patch_length = patch_length

    def forward(self, x):
        B = x.size(0)
        enc = self.enc_embedding(x)
        dec_input = (
            x[:, -1:, :].repeat(1, self.horizon, 1)
            if self.horizon > self.patch_length
            else x[:, -self.horizon :, :]
        )
        dec = self.dec_embedding(dec_input)
        enc = self.pos_encoding(enc)
        dec = self.pos_encoding(dec)
        for layer in self.encoder:
            enc = layer(enc)
        for layer in self.decoder:
            dec = layer(dec)
        out = self.projection(dec).squeeze(-1)
        return out
