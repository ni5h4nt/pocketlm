import torch
from pocketlm.model import PocketLM, PocketLMConfig

def _tiny():
    return PocketLMConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16)

def test_forward_shapes():
    m = PocketLM(_tiny())
    idx = torch.randint(0, 11, (3, 8))
    logits, loss = m(idx)
    assert logits.shape == (3, 8, 11)
    assert loss is None

def test_loss_when_targets_given():
    m = PocketLM(_tiny())
    idx = torch.randint(0, 11, (3, 8))
    _, loss = m(idx, idx)
    assert loss.ndim == 0 and loss.item() > 0

def test_weight_tying_default():
    m = PocketLM(_tiny())
    assert m.head.weight is m.tok_emb.weight   # tied by default (PocketLM v1)

def test_weight_tying_can_be_disabled():
    cfg = PocketLMConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16,
                         tie_weights=False)
    m = PocketLM(cfg)
    assert m.head.weight is not m.tok_emb.weight

def test_causality():
    # Changing the LAST token must not change earlier positions' logits.
    torch.manual_seed(0)
    m = PocketLM(_tiny()).eval()
    a = torch.randint(0, 11, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 11
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[:, :-1, :], lb[:, :-1, :], atol=1e-5)
