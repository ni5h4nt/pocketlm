"""Post-training pieces for u7: a DPO preference loss and a CPU-runnable
fake-quantization (the concept behind QLoRA's 4-bit base, without a GPU kernel)."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def dpo_loss(policy_chosen_logp: torch.Tensor, policy_rejected_logp: torch.Tensor,
             ref_chosen_logp: torch.Tensor, ref_rejected_logp: torch.Tensor,
             beta: float = 0.1) -> torch.Tensor:
    """Direct Preference Optimization loss. Pushes the policy to prefer the chosen
    response over the rejected one, relative to a frozen reference model. Lower
    loss = a larger preference margin than the reference."""
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp
    return -F.logsigmoid(beta * (policy_margin - ref_margin)).mean()


def fake_quantize(w: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Symmetric per-tensor quantize-then-dequantize to `bits`. This is the
    accuracy effect of low-bit weights (what QLoRA's frozen base pays) without a
    real packed kernel, so it runs anywhere."""
    if bits < 2:
        raise ValueError("bits must be >= 2")
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.clamp(torch.round(w / scale), -qmax, qmax)
    return q * scale
