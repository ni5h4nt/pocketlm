"""PocketLM, a tiny decoder-only language model you build from scratch."""
from .tokenizer import CharTokenizer
from .model import PocketLM, PocketLMConfig
from .train import train_tiny, pick_config
from .generate import generate, next_token_probs

__version__ = "0.1.0"
__all__ = ["CharTokenizer", "PocketLM", "PocketLMConfig", "train_tiny",
           "pick_config", "generate", "next_token_probs"]
