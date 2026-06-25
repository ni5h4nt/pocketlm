"""KV caching (u8). During generation the keys and values of past tokens never
change, so caching them turns each step from O(T) work into O(1) new work. This
module is the cache plus the parity guarantee: cached step-by-step attention
reproduces full causal attention exactly."""
from __future__ import annotations
import torch
from .attention import scaled_dot_product_attention


class KVCache:
    """Appends per-step keys/values along the time axis and returns the full
    history. k/v are [..., T, head_size]."""
    def __init__(self) -> None:
        self.k: torch.Tensor | None = None
        self.v: torch.Tensor | None = None

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.k = k if self.k is None else torch.cat([self.k, k], dim=-2)
        self.v = v if self.v is None else torch.cat([self.v, v], dim=-2)
        return self.k, self.v

    def __len__(self) -> int:
        return 0 if self.k is None else self.k.size(-2)


def cached_step(q_new: torch.Tensor, k_new: torch.Tensor, v_new: torch.Tensor,
                cache: KVCache) -> torch.Tensor:
    """One decode step: append the new token's k/v, then attend the new query
    over the whole history. The new token is the latest position, so all cached
    keys are in its past and no causal mask is needed."""
    k, v = cache.append(k_new, v_new)
    out, _ = scaled_dot_product_attention(q_new, k, v, causal=False)
    return out
