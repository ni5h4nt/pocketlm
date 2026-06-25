import torch
from pocketlm.model import PocketLM, PocketLMConfig
from pocketlm.lora import add_lora, trainable_parameters, LoRALinear
from pocketlm.posttrain import dpo_loss, fake_quantize

def _tiny():
    return PocketLMConfig(vocab_size=11, block_size=8, n_layer=2, n_head=2, n_embd=16)

def test_lora_is_initially_no_op():
    torch.manual_seed(0)
    m = PocketLM(_tiny()).eval()
    idx = torch.randint(0, 11, (1, 8))
    before, _ = m(idx)
    add_lora(m, rank=2)
    after, _ = m.eval()(idx)
    assert torch.allclose(before, after, atol=1e-6)   # B is zero-init

def test_lora_trains_few_parameters():
    m = PocketLM(_tiny())
    total = sum(p.numel() for p in m.parameters())
    add_lora(m, rank=2)
    trainable = sum(p.numel() for p in trainable_parameters(m))
    assert 0 < trainable < total * 0.2          # adapters are a small fraction
    assert any(isinstance(mod, LoRALinear) for mod in m.modules())

def test_dpo_prefers_larger_margin():
    t = torch.tensor
    big = dpo_loss(t(2.0), t(0.0), t(0.5), t(0.5))    # policy margin 2 vs ref 0
    flat = dpo_loss(t(0.5), t(0.5), t(0.5), t(0.5))   # policy margin 0 vs ref 0
    assert big < flat

def test_fake_quantize_more_bits_less_error():
    torch.manual_seed(0)
    w = torch.randn(64, 64)
    err4 = (fake_quantize(w, bits=4) - w).abs().mean()
    err8 = (fake_quantize(w, bits=8) - w).abs().mean()
    assert err8 < err4               # more bits, finer grid, less error
    assert err8 > 0                  # but quantization still loses something
