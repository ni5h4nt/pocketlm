import math
import torch
from pocketlm.model import PocketLM, PocketLMConfig
from pocketlm.trainer import (init_weights, cosine_lr, split_data, estimate_loss,
                              perplexity, save_checkpoint, load_checkpoint)

def _tiny():
    return PocketLMConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16)

def test_init_weights_std_and_zero_bias():
    m = init_weights(PocketLM(_tiny()), std=0.02)
    lin = [mod for mod in m.modules() if isinstance(mod, torch.nn.Linear)][0]
    assert abs(lin.weight.std().item() - 0.02) < 0.01
    if lin.bias is not None:
        assert torch.all(lin.bias == 0)

def test_cosine_lr_shape():
    base = 1.0
    assert cosine_lr(0, warmup=10, total=100, base_lr=base) < base       # warming up
    assert cosine_lr(9, warmup=10, total=100, base_lr=base) == base      # peak at end of warmup
    mid = cosine_lr(55, warmup=10, total=100, base_lr=base)
    assert 0 < mid < base                                                # decaying
    assert cosine_lr(100, warmup=10, total=100, base_lr=base, min_lr=0.0) == 0.0

def test_split_data():
    data = torch.arange(100)
    tr, va = split_data(data, 0.1)
    assert len(tr) == 90 and len(va) == 10
    assert torch.equal(torch.cat([tr, va]), data)

def test_perplexity():
    assert abs(perplexity(0.0) - 1.0) < 1e-9
    assert abs(perplexity(math.log(11)) - 11.0) < 1e-6

def test_estimate_loss_runs_and_restores_mode():
    m = PocketLM(_tiny())   # nn.Module defaults to training mode
    data = torch.randint(0, 11, (500,))
    g = torch.Generator().manual_seed(0)
    loss = estimate_loss(m, data, block_size=8, batch_size=4, iters=5, generator=g)
    assert loss > 0
    assert m.training is True   # eval was temporary; training mode restored

def test_checkpoint_roundtrip(tmp_path):
    m = PocketLM(_tiny())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    path = str(tmp_path / "ckpt.pt")
    save_checkpoint(path, m, opt, step=42)
    m2 = PocketLM(_tiny())
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    step = load_checkpoint(path, m2, opt2)
    assert step == 42
    for p1, p2 in zip(m.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)
