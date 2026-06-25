"""Training entry point + device-aware config presets."""
from __future__ import annotations
from dataclasses import replace
from typing import Any
import torch
from .model import PocketLM, PocketLMConfig
from .tokenizer import CharTokenizer

# values are heterogeneous (the config also has bool fields), so type as Any to
# keep the **unpack into PocketLMConfig well-typed.
_PRESETS: dict[str, dict[str, Any]] = {
    "gpu":   dict(n_layer=4, n_head=4, n_embd=128, block_size=128, steps=2000, batch_size=32),
    "cpu":   dict(n_layer=2, n_head=2, n_embd=64,  block_size=64,  steps=500,  batch_size=16),
    "micro": dict(n_layer=2, n_head=2, n_embd=64,  block_size=64,  steps=20,   batch_size=16),
}


def pick_config(device: str | None, vocab_size: int) -> PocketLMConfig:
    key = "gpu" if device == "cuda" else ("micro" if device == "micro" else "cpu")
    return PocketLMConfig(vocab_size=vocab_size, **_PRESETS[key])


def _get_batch(data: torch.Tensor, cfg: PocketLMConfig, device: str):
    ix = torch.randint(len(data) - cfg.block_size - 1, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + cfg.block_size] for i in ix])
    return x.to(device), y.to(device)


def train_tiny(corpus, cfg=None, *, seed=1337, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    tok = CharTokenizer.from_text(corpus)
    cfg = cfg or pick_config(device, tok.vocab_size)
    cfg = replace(cfg, vocab_size=tok.vocab_size)   # do not mutate caller's config
    data = torch.tensor(tok.encode(corpus), dtype=torch.long)
    if len(data) <= cfg.block_size + 1:
        raise ValueError(
            f"corpus has {len(data)} tokens; need > block_size+1 ({cfg.block_size + 1})")
    model = PocketLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    for _ in range(cfg.steps):
        xb, yb = _get_batch(data, cfg, device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return model, tok
