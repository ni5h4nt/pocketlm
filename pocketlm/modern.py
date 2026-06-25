"""Modern transformer components (u6): RMSNorm, SwiGLU, and rotary position
embeddings (RoPE). Each is a drop-in replacement for a part of PocketLM v1 and is
selectable through PocketLMConfig."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    """Root-mean-square norm: like LayerNorm but without centering, so it is
    cheaper (no mean subtraction, no bias) and works just as well in practice."""
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """Gated MLP: silu(gate(x)) * up(x), then project down. The multiplicative
    gate is more expressive than a plain GELU MLP at the same parameter budget."""
    def __init__(self, n_embd: int, hidden: int | None = None) -> None:
        super().__init__()
        hidden = hidden or 4 * n_embd
        self.w_gate = nn.Linear(n_embd, hidden, bias=False)
        self.w_up = nn.Linear(n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


def precompute_rope(head_size: int, seq_len: int, base: float = 10000.0
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape [seq_len, head_size//2] for RoPE."""
    if head_size % 2 != 0:
        raise ValueError("head_size must be even for RoPE")
    theta = 1.0 / (base ** (torch.arange(0, head_size, 2).float() / head_size))
    freqs = torch.outer(torch.arange(seq_len).float(), theta)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the feature pairs of x by position-dependent angles. x is
    [..., T, head_size]; cos/sin are [>=T, head_size//2]. Encodes position as a
    rotation, so attention scores depend on *relative* position."""
    t = x.size(-2)
    cos = cos[:t].view(1, 1, t, -1)
    sin = sin[:t].view(1, 1, t, -1)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack([rx1, rx2], dim=-1).flatten(-2)
