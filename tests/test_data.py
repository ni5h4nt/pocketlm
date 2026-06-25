import torch
from pocketlm.data import make_batch, pad_batch

def test_make_batch_shapes_and_shift():
    data = torch.arange(100)
    g = torch.Generator().manual_seed(0)
    x, y = make_batch(data, block_size=8, batch_size=4, generator=g)
    assert x.shape == (4, 8) and y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])   # y is x shifted by one

def test_make_batch_too_short():
    import pytest
    with pytest.raises(ValueError):
        make_batch(torch.arange(5), block_size=8, batch_size=2)

def test_pad_batch():
    padded, mask = pad_batch([[1, 2, 3], [4, 5]], pad_id=0)
    assert padded.shape == (2, 3)
    assert padded[1].tolist() == [4, 5, 0]
    assert mask.tolist() == [[True, True, True], [True, True, False]]

def test_pad_batch_empty():
    import pytest
    with pytest.raises(ValueError):
        pad_batch([], pad_id=0)
