from __future__ import annotations

from torch import nn
from torch.optim import AdamW, Optimizer

from .config import OptimizerConfig


def build_adamw(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Build AdamW over trainable parameters only.

    Matrix-like parameters receive weight decay; vectors/scalars such as biases
    and normalization parameters do not.
    """

    decay = []
    no_decay = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim >= 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)

    if not decay and not no_decay:
        raise ValueError("Model has no trainable parameters")

    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": config.weight_decay})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0})

    return AdamW(
        groups,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
    )
