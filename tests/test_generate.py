import torch
from pocketlm.model import PocketLM, PocketLMConfig
from pocketlm.tokenizer import CharTokenizer
from pocketlm.generate import generate, next_token_probs

def _setup():
    tok = CharTokenizer.from_text("abcde ")
    cfg = PocketLMConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=2,
                         n_head=2, n_embd=16)
    torch.manual_seed(0)
    return PocketLM(cfg).eval(), tok

def test_length_and_charset():
    m, tok = _setup()
    out = generate(m, tok, "ab", max_new_tokens=10, seed=0)
    assert len(out) == 2 + 10
    assert set(out) <= set(tok.itos)

def test_temperature_zero_is_deterministic():
    m, tok = _setup()
    a = generate(m, tok, "ab", max_new_tokens=10, temperature=0)
    b = generate(m, tok, "ab", max_new_tokens=10, temperature=0)
    assert a == b

def test_next_token_probs_is_distribution():
    m, tok = _setup()
    probs = next_token_probs(m, tok, "ab")
    assert set(probs) == set(tok.itos)
    assert abs(sum(probs.values()) - 1.0) < 1e-5

def test_top_k_one_equals_argmax():
    m, tok = _setup()
    greedy = generate(m, tok, "ab", max_new_tokens=5, temperature=0)
    topk1 = generate(m, tok, "ab", max_new_tokens=5, top_k=1, temperature=1.0)
    assert greedy == topk1

def test_invalid_args_raise():
    import pytest
    m, tok = _setup()
    with pytest.raises(ValueError):
        generate(m, tok, "", max_new_tokens=5)            # empty prompt
    with pytest.raises(ValueError):
        generate(m, tok, "ab", max_new_tokens=0)          # non-positive length
    with pytest.raises(ValueError):
        generate(m, tok, "ab", max_new_tokens=5, top_p=1.5)  # out of range
