from __future__ import annotations

from torch import nn
from torch.optim import AdamW, Optimizer

from .config import OptimizerConfig


def build_adamw(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Build AdamW over trainable parameters only."""

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("Model has no trainable parameters")

    return AdamW(
        trainable_parameters,
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
