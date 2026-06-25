import torch
from pocketlm.attention import scaled_dot_product_attention

def _qkv(T=4, hs=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(1, T, hs, generator=g),
            torch.randn(1, T, hs, generator=g),
            torch.randn(1, T, hs, generator=g))

def test_output_shape_matches_v():
    q, k, v = _qkv()
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == v.shape
    assert weights.shape == (1, 4, 4)

def test_weights_are_row_stochastic():
    q, k, v = _qkv()
    _, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 4), atol=1e-6)

def test_causal_masks_future():
    q, k, v = _qkv()
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    # strict upper triangle (future positions) must be exactly zero
    upper = torch.triu(torch.ones(4, 4), diagonal=1).bool()
    assert torch.all(weights[0][upper] == 0)

def test_causal_no_future_leak():
    q, k, v = _qkv()
    out_a, _ = scaled_dot_product_attention(q, k, v, causal=True)
    k2, v2 = k.clone(), v.clone()
    k2[:, 1:] += 5.0           # perturb every position after the first
    v2[:, 1:] += 5.0
    out_b, _ = scaled_dot_product_attention(q, k2, v2, causal=True)
    assert torch.allclose(out_a[:, 0], out_b[:, 0], atol=1e-6)
