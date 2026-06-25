import math
import torch
from pocketlm.train import train_tiny, pick_config

CORPUS = "to be or not to be that is the question " * 50

def test_training_beats_uniform_baseline():
    # Wiring check: a working model drives loss well below the uniform baseline
    # ln(vocab); a broken one (detached grads, bad mask) stays ~uniform. Fast and
    # not sensitive to exact convergence.
    cfg = pick_config("cpu", vocab_size=1)         # vocab_size overwritten internally
    cfg.steps, cfg.batch_size, cfg.block_size = 200, 16, 32
    model, tok = train_tiny(CORPUS, cfg, seed=0, device="cpu")
    data = torch.tensor(tok.encode(CORPUS))
    x = data[: cfg.block_size].unsqueeze(0)
    y = data[1 : cfg.block_size + 1].unsqueeze(0)
    with torch.no_grad():
        _, loss = model(x, y)
    assert loss.item() < 0.6 * math.log(tok.vocab_size)

def test_deterministic():
    cfg = pick_config("micro", vocab_size=1)
    m1, _ = train_tiny(CORPUS, cfg, seed=42, device="cpu")
    m2, _ = train_tiny(CORPUS, cfg, seed=42, device="cpu")
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)

def test_returns_model_and_tokenizer():
    model, tok = train_tiny(CORPUS, pick_config("micro", vocab_size=1), device="cpu")
    assert tok.decode(tok.encode("be")) == "be"
    assert model.cfg.vocab_size == tok.vocab_size
