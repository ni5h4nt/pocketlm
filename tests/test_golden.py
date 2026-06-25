import torch
from pocketlm.train import train_tiny, pick_config
from pocketlm.generate import generate

CORPUS = "the quick brown fox jumps over the lazy dog. " * 100

def _train():
    torch.use_deterministic_algorithms(True, warn_only=True)
    cfg = pick_config("micro", vocab_size=1)
    return train_tiny(CORPUS, cfg, seed=1234, device="cpu")

def test_generation_properties():
    model, tok = _train()
    out = generate(model, tok, "the ", max_new_tokens=40, temperature=0, seed=0)
    assert len(out) == 4 + 40
    assert set(out) <= set(tok.itos)        # never emits non-vocab chars

def test_greedy_is_reproducible():
    model, tok = _train()
    a = generate(model, tok, "the ", max_new_tokens=20, temperature=0)
    b = generate(model, tok, "the ", max_new_tokens=20, temperature=0)
    assert a == b
