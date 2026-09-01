from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FreezeConfig:
    """Controls which base-model parameters remain trainable."""

    train_last_n_layers: int | None = None
    train_embeddings: bool = True
    train_lm_head: bool = True
    trainable_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class OptimizerConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8


@dataclass(frozen=True)
class SchedulerConfig:
    name: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int | None = None


@dataclass(frozen=True)
class CheckpointConfig:
    save_every_steps: int | None = None
    keep_last: int | None = 3
    resume_from: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    output_dir: str
    num_epochs: int = 1
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    max_grad_norm: float | None = 1.0
    seed: int = 42
    log_every_steps: int = 10
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.per_device_batch_size <= 0:
            raise ValueError("per_device_batch_size must be > 0")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be > 0")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: no, fp16, bf16")
        if self.max_grad_norm is not None and self.max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be > 0 when set")
        if self.optimizer.learning_rate <= 0:
            raise ValueError("optimizer.learning_rate must be > 0")
        if self.optimizer.weight_decay < 0:
            raise ValueError("optimizer.weight_decay must be >= 0")
        if not 0 <= self.scheduler.warmup_ratio <= 1:
            raise ValueError("scheduler.warmup_ratio must be in [0, 1]")
        if self.scheduler.warmup_steps is not None and self.scheduler.warmup_steps < 0:
            raise ValueError("scheduler.warmup_steps must be >= 0")
        if self.checkpoint.save_every_steps is not None and self.checkpoint.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be > 0 when set")
        if self.checkpoint.keep_last is not None and self.checkpoint.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be > 0 when set")
