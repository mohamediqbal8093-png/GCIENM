from __future__ import annotations

import math
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) > self.pe.size(1):
            raise ValueError("Sequence length exceeds positional encoding capacity.")
        return x + self.pe[:, : x.size(1)]


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.left_padding, 0))
        return self.conv(x)


class DilatedCausalBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(channels, channels, kernel_size, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        residual = x
        z = x.transpose(1, 2)
        z = F.gelu(self.conv1(z))
        z = self.dropout(z)
        z = self.conv2(z)
        z = z.transpose(1, 2)
        return self.norm(residual + self.dropout(F.gelu(z)))


class MultiHeadNonlinearAttention(nn.Module):
    """
    Nonlinear tanh attention:
        s_ij = tanh(w_q^T q_i + w_k^T k_j + b)
        a_ij = softmax(s_ij)
        c_i  = sum_j a_ij v_j
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_score = nn.Parameter(torch.empty(num_heads, self.d_head))
        self.k_score = nn.Parameter(torch.empty(num_heads, self.d_head))
        self.bias = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.q_score)
        nn.init.xavier_uniform_(self.k_score)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        causal: bool = False,
        need_weights: bool = False,
    ):
        key = query if key is None else key
        value = key if value is None else value
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        q_term = torch.einsum("bhld,hd->bhl", q, self.q_score).unsqueeze(-1)
        k_term = torch.einsum("bhmd,hd->bhm", k, self.k_score).unsqueeze(-2)
        scores = torch.tanh(q_term + k_term + self.bias)

        if causal:
            lq, lk = scores.shape[-2:]
            mask = torch.triu(
                torch.ones(lq, lk, device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(query.size(0), query.size(1), self.d_model)
        out = self.out_proj(context)
        return (out, weights) if need_weights else out


class MultiHeadScaledDotAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        causal: bool = False,
        need_weights: bool = False,
    ):
        key = query if key is None else key
        value = key if value is None else value
        mask = None
        if causal:
            lq, lk = query.size(1), key.size(1)
            mask = torch.triu(
                torch.ones(lq, lk, device=query.device, dtype=torch.bool), diagonal=1
            )
        out, weights = self.attn(
            query, key, value, attn_mask=mask,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        return (out, weights) if need_weights else out


def build_attention(mode: str, d_model: int, num_heads: int, dropout: float):
    mode = mode.lower()
    if mode == "nonlinear":
        return MultiHeadNonlinearAttention(d_model, num_heads, dropout)
    if mode == "scaled_dot":
        return MultiHeadScaledDotAttention(d_model, num_heads, dropout)
    raise ValueError(f"Unknown attention mode: {mode}")


class MVPNN(nn.Module):
    """1x1 point-wise convolutions that preserve temporal organization."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels * 2, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 2, channels, kernel_size=1),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + z)


class GCIENMEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        attention_mode: str,
        use_atm: bool = True,
        use_mvpnn: bool = True,
        use_dilated_conv: bool = True,
    ):
        super().__init__()
        self.use_atm = use_atm
        self.use_mvpnn = use_mvpnn
        self.use_dilated_conv = use_dilated_conv
        self.attn = build_attention(attention_mode, d_model, num_heads, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.mvpnn = MVPNN(d_model, dropout)
        self.conv = DilatedCausalBlock(d_model, kernel_size, dilation, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_atm:
            z = self.attn(x, causal=True)
            x = self.attn_norm(x + self.dropout(z))
        if self.use_mvpnn:
            x = self.mvpnn(x)
        if self.use_dilated_conv:
            x = self.conv(x)
        return x


class GCIENMDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
        attention_mode: str,
        use_atm: bool = True,
        use_mvpnn: bool = True,
        use_dilated_conv: bool = True,
    ):
        super().__init__()
        self.use_atm = use_atm
        self.use_mvpnn = use_mvpnn
        self.use_dilated_conv = use_dilated_conv
        self.self_attn = build_attention(attention_mode, d_model, num_heads, dropout)
        self.cross_attn = build_attention(attention_mode, d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mvpnn = MVPNN(d_model, dropout)
        self.conv = DilatedCausalBlock(d_model, kernel_size, dilation, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        if self.use_atm:
            z = self.self_attn(x, causal=True)
            x = self.norm1(x + self.dropout(z))
            z = self.cross_attn(x, key=memory, value=memory, causal=False)
            x = self.norm2(x + self.dropout(z))
        if self.use_mvpnn:
            x = self.mvpnn(x)
        if self.use_dilated_conv:
            x = self.conv(x)
        return x


class GCIENM(nn.Module):
    def __init__(
        self,
        n_features: int,
        horizon: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        kernel_size: int = 3,
        dilation_rates=(1, 2, 4),
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dropout: float = 0.1,
        attention_mode: str = "nonlinear",
        use_atm: bool = True,
        use_mvpnn: bool = True,
        use_dilated_conv: bool = True,
    ):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        self.horizon = horizon
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.positional = PositionalEncoding(hidden_dim)
        dilations = list(dilation_rates) or [1]

        self.encoder = nn.ModuleList([
            GCIENMEncoderLayer(
                d_model=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                dropout=dropout,
                attention_mode=attention_mode,
                use_atm=use_atm,
                use_mvpnn=use_mvpnn,
                use_dilated_conv=use_dilated_conv,
            )
            for i in range(num_encoder_layers)
        ])

        self.horizon_tokens = nn.Parameter(torch.randn(1, horizon, hidden_dim) * 0.02)
        self.decoder = nn.ModuleList([
            GCIENMDecoderLayer(
                d_model=hidden_dim,
                num_heads=num_heads,
                kernel_size=kernel_size,
                dilation=dilations[i % len(dilations)],
                dropout=dropout,
                attention_mode=attention_mode,
                use_atm=use_atm,
                use_mvpnn=use_mvpnn,
                use_dilated_conv=use_dilated_conv,
            )
            for i in range(num_decoder_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        memory = self.positional(self.input_proj(x))
        for layer in self.encoder:
            memory = layer(memory)

        dec = self.horizon_tokens.expand(x.size(0), -1, -1)
        dec = self.positional(dec)
        for layer in self.decoder:
            dec = layer(dec, memory)
        return self.head(dec).squeeze(-1)


def model_from_config(cfg: dict, n_features: int):
    m = cfg["model"]
    return GCIENM(
        n_features=n_features,
        horizon=int(cfg["split"]["horizon"]),
        hidden_dim=int(m["hidden_dim"]),
        num_heads=int(m["num_heads"]),
        kernel_size=int(m["kernel_size"]),
        dilation_rates=tuple(int(v) for v in m["dilation_rates"]),
        num_encoder_layers=int(m["num_encoder_layers"]),
        num_decoder_layers=int(m["num_decoder_layers"]),
        dropout=float(m["dropout"]),
        attention_mode=str(m["attention_mode"]),
        use_atm=bool(m.get("use_atm", True)),
        use_mvpnn=bool(m.get("use_mvpnn", True)),
        use_dilated_conv=bool(m.get("use_dilated_conv", True)),
    )


if __name__ == "__main__":
    model = GCIENM(n_features=5, horizon=72)
    x = torch.randn(4, 48, 5)
    y = model(x)
    print("Input:", tuple(x.shape))
    print("Output:", tuple(y.shape))
