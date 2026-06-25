"""Autoregressive sampling with temperature / top-k / top-p, plus a
distribution-inspection helper for the probability lessons."""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .model import PocketLM
from .tokenizer import CharTokenizer


def _filter(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    # logits: [B, vocab]. Returns a filtered copy (removed entries -> -inf).
    logits = logits.clone()
    if top_k is not None and top_k < logits.size(-1):
        kth = torch.topk(logits, top_k).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove_sorted = cdf > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()  # always keep top-1
        remove_sorted[..., 0] = False
        # scatter the per-rank removal mask back to original vocab positions
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


@torch.no_grad()
def generate(model: PocketLM, tokenizer: CharTokenizer, prompt: str, *,
             max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             seed: int | None = None) -> str:
    if not prompt:
        raise ValueError("prompt must be non-empty (need at least one context token)")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be >= 1")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")
    if top_p is not None and not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")
    if seed is not None:
        torch.manual_seed(seed)
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    block = model.cfg.block_size
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -block:])
        logits = logits[:, -1, :]
        if temperature == 0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = _filter(logits / temperature, top_k, top_p)
            nxt = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        idx = torch.cat([idx, nxt], dim=1)
    return tokenizer.decode(idx[0].tolist())


@torch.no_grad()
def next_token_probs(model: PocketLM, tokenizer: CharTokenizer,
                     context: str) -> dict[str, float]:
    if not context:
        raise ValueError("context must be non-empty")
    device = next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long, device=device)
    logits, _ = model(idx[:, -model.cfg.block_size:])
    probs = F.softmax(logits[0, -1, :], dim=-1)
    return {tok: probs[i].item() for i, tok in enumerate(tokenizer.itos)}
