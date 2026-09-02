from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


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
class LossConfig:
    """Selects the default loss or a custom BaseLoss implementation.

    ``module_path=None`` uses ``DefaultCrossEntropyLoss``. ``kwargs`` are passed
    to whichever loss constructor is selected.
    """

    module_path: str | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GradientCheckpointingConfig:
    """Controls model-native activation checkpointing."""

    enabled: bool = False
    use_reentrant: bool | None = None


@dataclass(frozen=True)
class DeepSpeedConfig:
    """Small, backend-agnostic surface for common DeepSpeed ZeRO settings.

    Optimizer and scheduler ownership stays in ``llm-training``. This config is
    intentionally limited to parameter/gradient/state partitioning and offload.
    """

    zero_stage: Literal[1, 2, 3] = 2
    offload_optimizer_device: Literal["none", "cpu", "nvme"] = "none"
    offload_param_device: Literal["none", "cpu", "nvme"] = "none"
    offload_optimizer_nvme_path: str | None = None
    offload_param_nvme_path: str | None = None
    zero3_save_16bit_model: bool = True


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
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )
    deepspeed: DeepSpeedConfig | None = None
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
        if self.loss.module_path is not None and not self.loss.module_path.strip():
            raise ValueError("loss.module_path must be non-empty when set")

        if self.deepspeed is not None:
            ds = self.deepspeed
            if ds.zero_stage not in {1, 2, 3}:
                raise ValueError("deepspeed.zero_stage must be one of: 1, 2, 3")
            if ds.offload_optimizer_device not in {"none", "cpu", "nvme"}:
                raise ValueError(
                    "deepspeed.offload_optimizer_device must be one of: none, cpu, nvme"
                )
            if ds.offload_param_device not in {"none", "cpu", "nvme"}:
                raise ValueError(
                    "deepspeed.offload_param_device must be one of: none, cpu, nvme"
                )
            if ds.zero_stage < 2 and ds.offload_optimizer_device != "none":
                raise ValueError("optimizer offload requires DeepSpeed ZeRO stage 2 or 3")
            if ds.zero_stage != 3 and ds.offload_param_device != "none":
                raise ValueError("parameter offload requires DeepSpeed ZeRO stage 3")
            if (
                ds.offload_optimizer_device == "nvme"
                and not ds.offload_optimizer_nvme_path
            ):
                raise ValueError(
                    "deepspeed.offload_optimizer_nvme_path is required for NVMe optimizer offload"
                )
            if ds.offload_param_device == "nvme" and not ds.offload_param_nvme_path:
                raise ValueError(
                    "deepspeed.offload_param_nvme_path is required for NVMe parameter offload"
                )

        if self.checkpoint.save_every_steps is not None and self.checkpoint.save_every_steps <= 0:
            raise ValueError("checkpoint.save_every_steps must be > 0 when set")
        if self.checkpoint.keep_last is not None and self.checkpoint.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be > 0 when set")
