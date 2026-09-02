from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

from .config import SchedulerConfig, TrainingConfig


def resolve_warmup_steps(
    config: SchedulerConfig,
    *,
    num_training_steps: int,
) -> int:
    if num_training_steps <= 0:
        raise ValueError("num_training_steps must be > 0")

    warmup_steps = config.warmup_steps
    if warmup_steps is None:
        warmup_steps = int(num_training_steps * config.warmup_ratio)
    return min(warmup_steps, num_training_steps)


def build_scheduler(
    optimizer: Optimizer,
    config: SchedulerConfig,
    *,
    num_training_steps: int,
) -> LRScheduler:
    warmup_steps = resolve_warmup_steps(
        config,
        num_training_steps=num_training_steps,
    )

    return get_scheduler(
        config.name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def build_training_scheduler(
    optimizer: Any,
    config: TrainingConfig,
    *,
    num_training_steps: int,
) -> Any:
    """Build the real scheduler or a DeepSpeed scheduler placeholder."""

    ds = config.deepspeed
    if ds is None or ds.scheduler is None:
        return build_scheduler(
            optimizer,
            config.scheduler,
            num_training_steps=num_training_steps,
        )

    warmup_steps = resolve_warmup_steps(
        config.scheduler,
        num_training_steps=num_training_steps,
    )
    scheduler_type = ds.scheduler.type.lower()
    if scheduler_type in {"warmuplr", "warmupdecaylr"} and warmup_steps <= 0:
        raise ValueError(
            f"DeepSpeed {ds.scheduler.type} requires at least one warmup step; "
            "set scheduler.warmup_steps >= 1 or choose a warmup_ratio that "
            "resolves to at least one step"
        )

    from accelerate.utils.deepspeed import DummyScheduler

    return DummyScheduler(
        optimizer,
        total_num_steps=num_training_steps,
        warmup_num_steps=warmup_steps,
    )
