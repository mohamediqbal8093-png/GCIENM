from __future__ import annotations

import torch

from gcienm import (
    CausalConv1d,
    GCIENM,
    MultiHeadNonlinearAttention,
    MultiHeadScaledDotAttention,
)


def test_attention_dimensions():
    b, lq, lk, d, h = 3, 12, 20, 32, 4
    q = torch.randn(b, lq, d)
    k = torch.randn(b, lk, d)
    v = torch.randn(b, lk, d)

    for cls in (MultiHeadNonlinearAttention, MultiHeadScaledDotAttention):
        attn = cls(d, h, 0.0)
        out, weights = attn(q, k, v, need_weights=True)
        assert out.shape == (b, lq, d)
        assert weights.shape == (b, h, lq, lk), (
            cls.__name__,
            weights.shape,
        )


def test_gicienm_output():
    model = GCIENM(
        n_features=5,
        horizon=72,
        hidden_dim=32,
        num_heads=4,
        kernel_size=3,
        dilation_rates=(1, 2, 4),
        num_encoder_layers=2,
        num_decoder_layers=1,
        dropout=0.0,
    )
    x = torch.randn(2, 48, 5)
    y = model(x)
    assert y.shape == (2, 72)


def test_causal_conv_no_future_dependency():
    torch.manual_seed(1)
    conv = CausalConv1d(1, 1, kernel_size=3, dilation=1)
    x1 = torch.randn(1, 1, 10)
    x2 = x1.clone()
    x2[:, :, 7:] += 1000.0

    y1 = conv(x1)
    y2 = conv(x2)
    # Outputs through index 6 must be identical because only future values changed.
    assert torch.allclose(y1[:, :, :7], y2[:, :, :7], atol=1e-6)


def test_nonlinear_attention_causal_mask():
    torch.manual_seed(2)
    attn = MultiHeadNonlinearAttention(16, 4, 0.0)
    x = torch.randn(1, 8, 16)
    _, weights = attn(x, causal=True, need_weights=True)
    upper = torch.triu(weights[0, 0], diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper), atol=1e-7)


if __name__ == "__main__":
    tests = [
        test_attention_dimensions,
        test_gicienm_output,
        test_causal_conv_no_future_dependency,
        test_nonlinear_attention_causal_mask,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print("All shape/causality tests passed.")
