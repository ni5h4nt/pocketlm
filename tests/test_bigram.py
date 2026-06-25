import torch
from pocketlm.bigram import BigramLM

def test_forward_shapes():
    m = BigramLM(7)
    idx = torch.randint(0, 7, (3, 5))
    logits, loss = m(idx)
    assert logits.shape == (3, 5, 7)
    assert loss is None

def test_loss_when_targets_given():
    m = BigramLM(7)
    idx = torch.randint(0, 7, (3, 5))
    _, loss = m(idx, idx)
    assert loss.ndim == 0 and loss.item() > 0

def test_position_independent():
    # A bigram model's next-token logits depend only on the CURRENT token,
    # never on earlier context or position.
    m = BigramLM(7).eval()
    a = torch.tensor([[1, 2, 3]])
    b = torch.tensor([[4, 5, 3]])   # different context, same last token (3)
    with torch.no_grad():
        la, _ = m(a)
        lb, _ = m(b)
    assert torch.allclose(la[0, -1], lb[0, -1])
