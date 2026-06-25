import torch
from pocketlm.modern import RMSNorm, SwiGLU, precompute_rope, apply_rope

def test_rmsnorm_normalizes():
    norm = RMSNorm(8)
    x = torch.randn(2, 4, 8) * 5 + 3
    out = norm(x)
    assert out.shape == x.shape
    # with unit weight, each vector's RMS is ~1
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)

def test_swiglu_preserves_shape():
    mlp = SwiGLU(16)
    x = torch.randn(2, 5, 16)
    assert mlp(x).shape == x.shape

def test_rope_preserves_norm():
    # Rotation is orthogonal, so it never changes a vector's length.
    cos, sin = precompute_rope(head_size=8, seq_len=10)
    x = torch.randn(1, 2, 6, 8)   # [B, n_head, T, head_size]
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    assert torch.allclose(out.norm(dim=-1), x.norm(dim=-1), atol=1e-5)

def test_rope_position_zero_is_identity():
    cos, sin = precompute_rope(head_size=8, seq_len=10)
    x = torch.randn(1, 1, 3, 8)
    out = apply_rope(x, cos, sin)
    # position 0 has angle 0, so it is unrotated
    assert torch.allclose(out[:, :, 0], x[:, :, 0], atol=1e-6)

def test_rope_requires_even_head_size():
    import pytest
    with pytest.raises(ValueError):
        precompute_rope(head_size=7, seq_len=4)
