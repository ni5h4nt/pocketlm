"""Training toolkit for u5: weight init, LR schedule, train/val split, loss/
perplexity evaluation, and checkpoint save/resume. Each piece is one tested
utility a lesson drives."""
from __future__ import annotations
import math
import torch
import torch.nn as nn
from .data import make_batch


def init_weights(model: nn.Module, std: float = 0.02) -> nn.Module:
    """GPT-style init: small normal on Linear/Embedding weights, zero biases.
    A bad init gives NaN loss or stalled training; this one trains stably."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=std)
    return model


def cosine_lr(step: int, *, warmup: int, total: int,
              base_lr: float, min_lr: float = 0.0) -> float:
    """Linear warmup for `warmup` steps, then cosine decay to `min_lr` at `total`."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= total:
        return min_lr
    ratio = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * ratio))


def split_data(data: torch.Tensor, val_frac: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a token stream into (train, val). Keeping val separate is how you
    detect overfitting and data leakage."""
    if not 0.0 < val_frac < 1.0:
        raise ValueError("val_frac must be in (0, 1)")
    n = int(len(data) * (1 - val_frac))
    return data[:n], data[n:]


@torch.no_grad()
def estimate_loss(model: nn.Module, data: torch.Tensor, block_size: int, batch_size: int,
                  *, iters: int = 20, generator: torch.Generator | None = None) -> float:
    """Mean next-token loss over `iters` random batches (eval mode, no grad)."""
    was_training = model.training
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = make_batch(data, block_size, batch_size, generator=generator)
        _, loss = model(x, y)
        total += loss.item()
    if was_training:
        model.train()
    return total / iters


def perplexity(loss: float) -> float:
    """Perplexity = exp(loss): the effective number of equally-likely next
    tokens. A uniform model over V tokens scores exactly V."""
    return math.exp(loss)


def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer,
                    step: int) -> None:
    torch.save({"model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "step": step}, path)


def load_checkpoint(path: str, model: nn.Module,
                    optimizer: torch.optim.Optimizer | None = None) -> int:
    """Restore model (and optionally optimizer) state. Returns the saved step so
    training resumes exactly where it stopped."""
    # weights_only=True refuses arbitrary pickled objects; our checkpoints are
    # only tensors, ints, and plain containers, so this stays safe to load.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optim"])
    return int(ckpt["step"])
