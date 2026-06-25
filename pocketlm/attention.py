"""The attention primitive, taught from scratch in u3. Scaled dot-product
attention with an optional causal mask, returning the attention weights so they
can be visualized and the no-future-leak invariant can be checked."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """q, k, v: [..., T, head_size]. Returns (out[..., T, head_size],
    weights[..., T, T]). The 1/sqrt(head_size) scale keeps the softmax from
    saturating as head_size grows. With causal=True, position t attends only to
    positions <= t."""
    head_size = q.size(-1)
    att = (q @ k.transpose(-2, -1)) / (head_size ** 0.5)
    if causal:
        t = q.size(-2)
        allowed = torch.tril(torch.ones(t, t, device=q.device, dtype=torch.bool))
        att = att.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(att, dim=-1)
    return weights @ v, weights
