import torch
from pocketlm.attention import scaled_dot_product_attention
from pocketlm.kvcache import KVCache, cached_step
from pocketlm.model import PocketLM, PocketLMConfig
from pocketlm.tokenizer import CharTokenizer
from pocketlm.serve import InferenceService

def test_kvcache_matches_full_attention():
    # Feeding tokens one at a time through the cache must reproduce, row by row,
    # the output of full causal attention over the whole sequence.
    torch.manual_seed(0)
    B, H, T, hs = 1, 2, 6, 8
    q = torch.randn(B, H, T, hs)
    k = torch.randn(B, H, T, hs)
    v = torch.randn(B, H, T, hs)
    full, _ = scaled_dot_product_attention(q, k, v, causal=True)
    cache = KVCache()
    for t in range(T):
        out_t = cached_step(q[:, :, t:t + 1], k[:, :, t:t + 1], v[:, :, t:t + 1], cache)
        assert torch.allclose(out_t[:, :, 0], full[:, :, t], atol=1e-5)
    assert len(cache) == T

def _service():
    tok = CharTokenizer.from_text("abcde ")
    cfg = PocketLMConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=2,
                         n_head=2, n_embd=16)
    torch.manual_seed(0)
    return InferenceService(PocketLM(cfg), tok)

def test_service_handle_shape():
    svc = _service()
    resp = svc.handle({"prompt": "ab", "max_new_tokens": 6, "seed": 0})
    assert resp["prompt"] == "ab"
    assert len(resp["completion"]) == 2 + 6
    assert set(resp["completion"]) <= set(svc.tokenizer.itos)

def test_service_requires_prompt():
    import pytest
    with pytest.raises(ValueError):
        _service().handle({"max_new_tokens": 5})
