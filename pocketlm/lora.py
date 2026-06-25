"""LoRA (u7): freeze the base weights and train a small low-rank update instead.
A LoRALinear adds `B @ A` (rank r) to a frozen Linear, so fine-tuning touches a
tiny fraction of the parameters."""
from __future__ import annotations
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.b.weight)        # start as a no-op: output == base(x)
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scaling * self.b(self.a(x))


def add_lora(model: nn.Module, rank: int = 4, alpha: int = 8,
             targets: tuple[str, ...] = ("c_q", "c_kv", "c_proj")) -> nn.Module:
    """Freeze the whole model, then wrap the named Linear submodules with LoRA so
    only the adapters train. Returns the same model, modified in place."""
    for p in model.parameters():
        p.requires_grad = False
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and name in targets:
                setattr(module, name, LoRALinear(child, rank, alpha))
    return model


def trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    """The parameters left trainable, i.e. the LoRA adapters after add_lora()."""
    return [p for p in model.parameters() if p.requires_grad]
