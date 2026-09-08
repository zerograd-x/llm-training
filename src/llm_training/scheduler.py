from __future__ import annotations

from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler

from .config import SchedulerConfig, TrainingConfig


def bind_scheduler_optimizer(scheduler: Any, prepared_optimizer: Any) -> None:
    """Keep a Torch scheduler attached to the optimizer that owns updates.

    Accelerate may replace AdamW with CPUAdam for DeepSpeed offload. Their
    parameter-group dictionaries initially alias, masking the old reference,
    until checkpoint loading replaces CPUAdam's groups. Follow the optimizer
    object itself so restored and future groups remain visible to the scheduler.
    DeepSpeed-owned scheduler placeholders require no rebinding.
    """
    if not isinstance(scheduler, LRScheduler):
        return
    seen = set()
    optimizer = prepared_optimizer
    while hasattr(optimizer, "optimizer"):
        if id(optimizer) in seen:
            raise TypeError("Cannot resolve the prepared optimizer for a Torch scheduler")
        seen.add(id(optimizer))
        optimizer = optimizer.optimizer
    if not isinstance(optimizer, Optimizer):
        raise TypeError("A Torch scheduler requires a prepared Torch optimizer")
    scheduler.optimizer = optimizer


def resolve_warmup_steps(
    config: SchedulerConfig,
    *,
    num_training_steps: int,
) -> int:
    """Resolve warmup in user-visible optimizer-step units.

    Every public ``*_steps`` setting in llm-training is expressed in synchronized
    optimizer updates. Backend-specific scheduler stepping is translated later
    and never changes the meaning of ``SchedulerConfig.warmup_steps``.
    """
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
    scheduler_step_scale: int = 1,
) -> LRScheduler:
    """Build a scheduler from optimizer-step semantics.

    ``num_training_steps`` and configured warmup are optimizer updates.
    ``scheduler_step_scale`` is an internal Accelerate/backend translation for
    schedulers that advance more than once per synchronized optimizer update.
    """
    if scheduler_step_scale <= 0:
        raise ValueError("scheduler_step_scale must be > 0")
    warmup_steps = resolve_warmup_steps(
        config,
        num_training_steps=num_training_steps,
    )

    return get_scheduler(
        config.name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * scheduler_step_scale,
        num_training_steps=num_training_steps * scheduler_step_scale,
    )


def build_training_scheduler(
    optimizer: Any,
    config: TrainingConfig,
    *,
    num_training_steps: int,
    scheduler_step_scale: int = 1,
) -> Any:
    """Build the real scheduler or a DeepSpeed scheduler placeholder.

    Public step counts remain optimizer updates; backend-specific scaling is
    applied only to the underlying scheduler horizon.
    """

    ds = config.deepspeed
    if ds is None or ds.scheduler is None:
        return build_scheduler(
            optimizer,
            config.scheduler,
            num_training_steps=num_training_steps,
            scheduler_step_scale=scheduler_step_scale,
        )

    if scheduler_step_scale <= 0:
        raise ValueError("scheduler_step_scale must be > 0")
    warmup_steps = resolve_warmup_steps(
        config.scheduler,
        num_training_steps=num_training_steps,
    )
    scheduler_type = ds.scheduler.type.lower()
    if scheduler_type in {"warmuplr", "warmupdecaylr"} and warmup_steps <= 0:
        raise ValueError(
            f"DeepSpeed {ds.scheduler.type} requires at least one warmup step; "
            "set scheduler.warmup_steps >= 1 or choose a warmup_ratio that "
            "resolves to at least one optimizer step"
        )

    from accelerate.utils.deepspeed import DummyScheduler

    return DummyScheduler(
        optimizer,
        total_num_steps=num_training_steps * scheduler_step_scale,
        warmup_num_steps=warmup_steps * scheduler_step_scale,
    )
