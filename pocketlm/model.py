"""The PocketLM decoder-only transformer. Config-driven so u6 can swap in modern
components (RMSNorm, SwiGLU, RoPE, grouped-query attention); the defaults reproduce
the PocketLM v1 architecture exactly."""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from .modern import RMSNorm, SwiGLU, precompute_rope, apply_rope


@dataclass
class PocketLMConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    steps: int = 2000
    batch_size: int = 32
    tie_weights: bool = True
    norm: str = "layernorm"        # "layernorm" | "rmsnorm"
    mlp: str = "gelu"              # "gelu" | "swiglu"
    pos: str = "learned"          # "learned" | "rope"
    n_kv_head: int | None = None  # None -> full multi-head; < n_head -> grouped-query


def _make_norm(cfg: PocketLMConfig) -> nn.Module:
    if cfg.norm == "layernorm":
        return nn.LayerNorm(cfg.n_embd)
    if cfg.norm == "rmsnorm":
        return RMSNorm(cfg.n_embd)
    raise ValueError(f"unknown norm {cfg.norm!r}")


def _make_mlp(cfg: PocketLMConfig) -> nn.Module:
    if cfg.mlp == "gelu":
        return nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )
    if cfg.mlp == "swiglu":
        return SwiGLU(cfg.n_embd)
    raise ValueError(f"unknown mlp {cfg.mlp!r}")


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: PocketLMConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head or cfg.n_head
        if cfg.n_head % self.n_kv != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        self.head_size = cfg.n_embd // cfg.n_head
        self.use_rope = cfg.pos == "rope"
        self.c_q = nn.Linear(cfg.n_embd, cfg.n_head * self.head_size)
        self.c_kv = nn.Linear(cfg.n_embd, 2 * self.n_kv * self.head_size)
        self.c_proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("mask", mask.view(1, 1, cfg.block_size, cfg.block_size))
        if self.use_rope:
            cos, sin = precompute_rope(self.head_size, cfg.block_size)
            self.register_buffer("rope_cos", cos)
            self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        kv = self.c_kv(x).view(B, T, 2, self.n_kv, self.head_size)
        k = kv[:, :, 0].transpose(1, 2)        # [B, n_kv, T, head_size]
        v = kv[:, :, 1].transpose(1, 2)
        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)
        if self.n_kv != self.n_head:           # grouped-query: share KV across query heads
            repeat = self.n_head // self.n_kv
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class Block(nn.Module):
    def __init__(self, cfg: PocketLMConfig) -> None:
        super().__init__()
        self.ln1 = _make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = _make_norm(cfg)
        self.mlp = _make_mlp(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class PocketLM(nn.Module):
    def __init__(self, cfg: PocketLMConfig) -> None:
        super().__init__()
        if cfg.pos not in ("learned", "rope"):
            raise ValueError(f"unknown pos {cfg.pos!r}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.use_rope = cfg.pos == "rope"
        if not self.use_rope:
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = _make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            # Share the embedding matrix with the output projection: same
            # [vocab, n_embd] weight, fewer parameters, a common GPT trick.
            self.head.weight = self.tok_emb.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)
        if not self.use_rope:   # RoPE injects position inside attention instead
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
