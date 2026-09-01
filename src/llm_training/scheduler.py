from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

from .config import SchedulerConfig


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    num_training_steps: int,
) -> LRScheduler:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    warmup_steps = config.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
    warmup_steps = min(warmup_steps, num_training_steps)

    return get_scheduler(
        config.name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
