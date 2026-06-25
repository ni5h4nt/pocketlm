"""The simplest language model: a bigram. Each token id indexes a row of logits
over the next token, so prediction depends only on the current token. Used in u1
to ground the training loop before transformer mechanics."""
from __future__ import annotations
import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLM(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        # row t = the logits over the next token given current token t.
        self.emb = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        logits = self.emb(idx)              # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
