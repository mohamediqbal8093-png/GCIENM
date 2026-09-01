from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from gcienm import CausalConv1d, PositionalEncoding


class CNNBaseline(nn.Module):
    def __init__(self, n_features, horizon, hidden_dim=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(n_features, hidden_dim, kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        z = self.net(x.transpose(1, 2)).squeeze(-1)
        return self.head(z)


class TemporalBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, x):
        z = F.gelu(self.conv1(x))
        z = self.dropout(z)
        z = self.conv2(z)
        return self.norm(x + self.dropout(F.gelu(z)))


class TCNBaseline(nn.Module):
    def __init__(self, n_features, horizon, hidden_dim=64, kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(n_features, hidden_dim, 1)
        self.blocks = nn.Sequential(
            TemporalBlock(hidden_dim, kernel_size, 1, dropout),
            TemporalBlock(hidden_dim, kernel_size, 2, dropout),
            TemporalBlock(hidden_dim, kernel_size, 4, dropout),
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        z = self.blocks(self.in_proj(x.transpose(1, 2)))
        return self.head(z[:, :, -1])


class RecurrentBaseline(nn.Module):
    def __init__(self, cell, n_features, horizon, hidden_dim=64, dropout=0.1):
        super().__init__()
        cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = cls(
            n_features,
            hidden_dim,
            num_layers=2,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        z, _ = self.rnn(x)
        return self.head(z[:, -1])


class TransformerBaseline(nn.Module):
    def __init__(
        self, n_features, horizon, hidden_dim=64, num_heads=4, dropout=0.1
    ):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.pos = PositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x):
        z = self.pos(self.input_proj(x))
        mask = torch.triu(
            torch.ones(z.size(1), z.size(1), device=z.device, dtype=torch.bool),
            diagonal=1,
        )
        z = self.encoder(z, mask=mask)
        return self.head(z[:, -1])


def build_baseline(name: str, cfg: dict, n_features: int):
    m = cfg["model"]
    h = int(cfg["split"]["horizon"])
    hidden = int(m["hidden_dim"])
    kernel = int(m["kernel_size"])
    dropout = float(m["dropout"])
    heads = int(m["num_heads"])

    name = name.lower()
    if name == "cnn":
        return CNNBaseline(n_features, h, hidden, kernel, dropout)
    if name == "tcn":
        return TCNBaseline(n_features, h, hidden, kernel, dropout)
    if name in {"rnn", "gru", "lstm"}:
        return RecurrentBaseline(name, n_features, h, hidden, dropout)
    if name == "transformer":
        return TransformerBaseline(n_features, h, hidden, heads, dropout)
    raise ValueError(f"Unsupported baseline: {name}")
