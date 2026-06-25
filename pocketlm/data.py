"""Batching helpers: contiguous next-token batches for training, and ragged
sequence padding with a boolean mask for variable-length inputs."""
from __future__ import annotations
import torch


def make_batch(data: torch.Tensor, block_size: int, batch_size: int, *,
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `batch_size` windows of length `block_size`; y is x shifted by one."""
    if len(data) <= block_size + 1:
        raise ValueError("data must be longer than block_size + 1")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + block_size] for i in ix])
    return x, y


def pad_batch(seqs: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad ragged sequences to the longest. Returns (padded[B,Tmax],
    mask[B,Tmax]) where mask is True on real tokens, False on padding."""
    if not seqs:
        raise ValueError("seqs must be non-empty")
    width = max(len(s) for s in seqs)
    padded = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.bool)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = True
    return padded, mask
