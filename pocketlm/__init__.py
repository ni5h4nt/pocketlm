"""PocketLM, a tiny decoder-only language model you build from scratch."""
from .tokenizer import CharTokenizer
from .bpe import BPETokenizer
from .bigram import BigramLM
from .attention import scaled_dot_product_attention
from .modern import RMSNorm, SwiGLU, precompute_rope, apply_rope
from .model import PocketLM, PocketLMConfig
from .train import train_tiny, pick_config
from .generate import generate, next_token_probs
from .data import make_batch, pad_batch
from .trainer import (init_weights, cosine_lr, split_data, estimate_loss,
                      perplexity, save_checkpoint, load_checkpoint)

__version__ = "0.1.0"
__all__ = ["CharTokenizer", "BPETokenizer", "BigramLM", "scaled_dot_product_attention",
           "PocketLM", "PocketLMConfig", "train_tiny", "pick_config", "generate",
           "next_token_probs", "make_batch", "pad_batch", "init_weights", "cosine_lr",
           "split_data", "estimate_loss", "perplexity", "save_checkpoint",
           "load_checkpoint", "RMSNorm", "SwiGLU", "precompute_rope", "apply_rope"]
