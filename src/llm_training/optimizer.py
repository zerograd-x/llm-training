from __future__ import annotations

from typing import Any

from torch import nn
from torch.optim import AdamW, Optimizer

from .config import OptimizerConfig, TrainingConfig


def _trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise ValueError("Model has no trainable parameters")
    return parameters


def build_adamw(model: nn.Module, config: OptimizerConfig) -> Optimizer:
    """Build AdamW over trainable parameters only."""

    return AdamW(
        _trainable_parameters(model),
        lr=config.learning_rate,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


def build_training_optimizer(model: nn.Module, config: TrainingConfig) -> Any:
    """Build the real torch optimizer or a DeepSpeed optimizer placeholder."""

    ds = config.deepspeed
    if ds is None or ds.optimizer is None:
        return build_adamw(model, config.optimizer)

    from accelerate.utils.deepspeed import DummyOptim

    return DummyOptim(
        _trainable_parameters(model),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
