from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class FreezeConfig:
    """Controls which base-model parameters remain fully trainable."""

    train_last_n_layers: int | None = None
    train_embeddings: bool = True
    train_lm_head: bool = True
    trainable_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LoRAConfig:
    """Configures LoRA adapter placement and low-rank parameterization.

    ``target_modules`` contains exact module names or suffixes. When omitted,
    the runtime uses a conservative allow-list of common transformer
    projection names and resolves them to fully-qualified module names inside
    transformer blocks. ``target_last_n_layers`` limits adapter placement and
    is intentionally separate from ``FreezeConfig.train_last_n_layers``:
    FreezeConfig controls full-training of base parameters, while this field
    controls where adapter parameters are inserted.
    """

    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] | None = None
    target_last_n_layers: int | None = None
    adapter_name: str = "default"


@dataclass(frozen=True)
class TrainableTokensConfig:
    """Train only newly appended vocabulary rows through PEFT TrainableTokens.

    The model must already have been resized to the expanded vocabulary before
    it is passed to ``Trainer``. ``n_tokens_added`` identifies the final N rows
    of that resized vocabulary as trainable. The runtime deliberately uses
    PEFT's TrainableTokens implementation rather than gradient masking a full
    embedding matrix.
    """

    n_tokens_added: int
    adapter_name: str = "default"


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
class DeepSpeedOptimizerConfig:
    """Explicitly opt in to a DeepSpeed-owned optimizer.

    ``type`` is the optimizer name understood by DeepSpeed. ``params`` is for
    backend-specific options; common optimizer hyperparameters remain sourced
    from ``OptimizerConfig`` so they are not duplicated across configurations.
    """

    type: str = "AdamW"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepSpeedSchedulerConfig:
    """Explicitly opt in to a DeepSpeed-owned learning-rate scheduler."""

    type: str = "WarmupDecayLR"
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeepSpeedConfig:
    """Small, general surface for DeepSpeed ZeRO and optional DS ownership."""

    zero_stage: Literal[1, 2, 3] = 2
    offload_optimizer_device: Literal["none", "cpu", "nvme"] = "none"
    offload_param_device: Literal["none", "cpu", "nvme"] = "none"
    offload_optimizer_nvme_path: str | None = None
    offload_param_nvme_path: str | None = None
    zero3_save_16bit_model: bool = True
    optimizer: DeepSpeedOptimizerConfig | None = None
    scheduler: DeepSpeedSchedulerConfig | None = None


@dataclass(frozen=True)
class DPOConfig:
    """Direct Preference Optimization parameters owned by TRL."""

    beta: float = 0.1
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    label_smoothing: float = 0.0
    max_length: int = 1024
    max_prompt_length: int = 512
    reference_free: bool = False
    system_prompt: str | None = None

    def validate(self) -> None:
        if self.beta <= 0:
            raise ValueError("dpo.beta must be > 0")
        if self.loss_type not in {"sigmoid", "hinge", "ipo"}:
            raise ValueError("dpo.loss_type must be one of: sigmoid, hinge, ipo")
        if not 0 <= self.label_smoothing < 0.5:
            raise ValueError("dpo.label_smoothing must be in [0, 0.5)")
        if self.max_length <= 0:
            raise ValueError("dpo.max_length must be > 0")
        if self.max_prompt_length <= 0:
            raise ValueError("dpo.max_prompt_length must be > 0")
        if self.max_prompt_length > self.max_length:
            raise ValueError("dpo.max_prompt_length must be <= dpo.max_length")
        if self.system_prompt is not None and not self.system_prompt.strip():
            raise ValueError("dpo.system_prompt must be non-empty when set")


@dataclass(frozen=True)
class CheckpointConfig:
    save_every_steps: int | None = None
    keep_last: int | None = 3
    resume_from: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    batch_size: int | None = None
    every_steps: int | None = None
    every_epochs: int | None = 1


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
    freeze: FreezeConfig | None = None
    lora: LoRAConfig | None = None
    trainable_tokens: TrainableTokensConfig | None = None
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    gradient_checkpointing: GradientCheckpointingConfig = field(
        default_factory=GradientCheckpointingConfig
    )
    deepspeed: DeepSpeedConfig | None = None
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    max_steps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def validate(self) -> None:
        for name in ("batch_size", "every_steps", "every_epochs"):
            value = getattr(self.evaluation, name)
            if value is not None and value <= 0:
                raise ValueError(f"evaluation.{name} must be > 0 when set")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be > 0")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError("max_steps must be > 0 when set")
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

        if self.freeze is not None:
            if (
                self.freeze.train_last_n_layers is not None
                and self.freeze.train_last_n_layers < 0
            ):
                raise ValueError("freeze.train_last_n_layers must be >= 0 or None")
            if any(not prefix.strip() for prefix in self.freeze.trainable_prefixes):
                raise ValueError(
                    "freeze.trainable_prefixes must not contain empty prefixes"
                )

        if self.lora is not None:
            lora = self.lora
            if lora.rank <= 0:
                raise ValueError("lora.rank must be > 0")
            if lora.alpha <= 0:
                raise ValueError("lora.alpha must be > 0")
            if not 0 <= lora.dropout < 1:
                raise ValueError("lora.dropout must be in [0, 1)")
            if lora.bias not in {"none", "all", "lora_only"}:
                raise ValueError("lora.bias must be one of: none, all, lora_only")
            if not lora.adapter_name.strip():
                raise ValueError("lora.adapter_name must be non-empty")
            if lora.target_last_n_layers is not None and lora.target_last_n_layers <= 0:
                raise ValueError("lora.target_last_n_layers must be > 0 when set")
            if lora.target_modules is not None:
                if not lora.target_modules:
                    raise ValueError("lora.target_modules must not be empty when set")
                if any(not target.strip() for target in lora.target_modules):
                    raise ValueError("lora.target_modules must not contain empty names")
                if len(set(lora.target_modules)) != len(lora.target_modules):
                    raise ValueError("lora.target_modules must not contain duplicates")

        if self.trainable_tokens is not None:
            trainable_tokens = self.trainable_tokens
            if trainable_tokens.n_tokens_added <= 0:
                raise ValueError("trainable_tokens.n_tokens_added must be > 0")
            if not trainable_tokens.adapter_name.strip():
                raise ValueError("trainable_tokens.adapter_name must be non-empty")
            if self.lora is not None:
                raise ValueError(
                    "lora and trainable_tokens cannot be enabled together yet; "
                    "use one PEFT adaptation method per training run"
                )
            if self.freeze is not None:
                raise ValueError(
                    "freeze and trainable_tokens cannot be combined yet: standalone "
                    "TrainableTokens means only the selected vocabulary rows are trainable"
                )

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
                raise ValueError(
                    "optimizer offload requires DeepSpeed ZeRO stage 2 or 3"
                )
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

            if ds.optimizer is not None:
                if not ds.optimizer.type.strip():
                    raise ValueError("deepspeed.optimizer.type must be non-empty")
                reserved = {"lr", "weight_decay"}
                if ds.optimizer.type.lower() in {"adam", "adamw"}:
                    reserved |= {"betas", "eps"}
                overlap = sorted(reserved.intersection(ds.optimizer.params))
                if overlap:
                    raise ValueError(
                        "deepspeed.optimizer.params must not override shared optimizer "
                        f"settings: {', '.join(overlap)}"
                    )
                if ds.scheduler is None:
                    raise ValueError(
                        "DeepSpeed-owned optimizer requires a DeepSpeed-owned scheduler"
                    )

            if ds.scheduler is not None:
                if not ds.scheduler.type.strip():
                    raise ValueError("deepspeed.scheduler.type must be non-empty")
                reserved_scheduler = {
                    "warmup_min_lr",
                    "warmup_max_lr",
                    "warmup_num_steps",
                    "total_num_steps",
                }
                overlap = sorted(reserved_scheduler.intersection(ds.scheduler.params))
                if overlap:
                    raise ValueError(
                        "deepspeed.scheduler.params must not override shared scheduler "
                        f"settings: {', '.join(overlap)}"
                    )

        if (
            self.checkpoint.save_every_steps is not None
            and self.checkpoint.save_every_steps <= 0
        ):
            raise ValueError("checkpoint.save_every_steps must be > 0 when set")
        if self.checkpoint.keep_last is not None and self.checkpoint.keep_last <= 0:
            raise ValueError("checkpoint.keep_last must be > 0 when set")
