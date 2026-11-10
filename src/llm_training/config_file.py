from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import (
    CheckpointConfig,
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    DPOConfig,
    EvaluationConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    GRPOConfig,
    GRPORewardConfig,
    LoRAConfig,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainableTokensConfig,
    TrainingConfig,
)


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _strict_dataclass(cls, value: Any, path: str):
    payload = _mapping(value, path)
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"Unknown {path} fields: {', '.join(unknown)}")
    try:
        return cls(**payload)
    except TypeError as exc:
        raise ValueError(f"Invalid {path}: {exc}") from exc


def load_config_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if source.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValueError("Config file must use .json, .yaml, or .yml")
    try:
        text = source.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read config file {source}: {exc}") from exc
    try:
        payload = json.loads(text) if source.suffix.lower() == ".json" else yaml.safe_load(text)
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"Invalid config file {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Config root must be a mapping")
    return dict(payload)


@dataclass(frozen=True)
class ModelConfig:
    name_or_path: str
    tokenizer_name_or_path: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = False
    model_kwargs: Mapping[str, Any] | None = None
    tokenizer_kwargs: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.name_or_path.strip():
            raise ValueError("model.name_or_path must be non-empty")
        if self.dtype not in {"auto", "float16", "bfloat16", "float32"}:
            raise ValueError("model.dtype must be one of: auto, float16, bfloat16, float32")
        model_kwargs = dict(self.model_kwargs or {})
        tokenizer_kwargs = dict(self.tokenizer_kwargs or {})
        reserved_model = sorted({"torch_dtype", "trust_remote_code"}.intersection(model_kwargs))
        if reserved_model:
            raise ValueError(
                "model.model_kwargs must not override explicit settings: "
                + ", ".join(reserved_model)
            )
        if "trust_remote_code" in tokenizer_kwargs:
            raise ValueError(
                "model.tokenizer_kwargs must not override trust_remote_code"
            )
        object.__setattr__(self, "model_kwargs", model_kwargs)
        object.__setattr__(self, "tokenizer_kwargs", tokenizer_kwargs)


@dataclass(frozen=True)
class DataConfig:
    train_file: str
    format: str
    validation_file: str | None = None
    prompt_field: str = "prompt"
    completion_field: str = "completion"
    chosen_field: str = "chosen"
    rejected_field: str = "rejected"
    text_field: str = "text"
    max_sequence_length: int = 2048

    def __post_init__(self) -> None:
        if not self.train_file.strip():
            raise ValueError("data.train_file must be non-empty")
        if self.format not in {"sft", "causal_lm", "dpo", "grpo"}:
            raise ValueError(
                "data.format must be one of: sft, causal_lm, dpo, grpo"
            )
        if self.max_sequence_length <= 0:
            raise ValueError("data.max_sequence_length must be > 0")
        if self.format == "sft":
            required = (self.prompt_field, self.completion_field)
        elif self.format == "dpo":
            required = (self.prompt_field, self.chosen_field, self.rejected_field)
        elif self.format == "grpo":
            required = (self.prompt_field,)
        else:
            required = (self.text_field,)
        if any(not name.strip() for name in required):
            raise ValueError("data field names must be non-empty")


@dataclass(frozen=True)
class ArtifactConfig:
    save: str = "auto"

    def __post_init__(self) -> None:
        if self.save not in {"auto", "model", "adapter", "none"}:
            raise ValueError("artifacts.save must be one of: auto, model, adapter, none")


@dataclass(frozen=True)
class TrainingJobConfig:
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    artifacts: ArtifactConfig
    dpo: DPOConfig | None = None
    grpo: GRPOConfig | None = None


_TRAINING_SCALARS = {
    "output_dir",
    "num_epochs",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "mixed_precision",
    "max_grad_norm",
    "seed",
    "log_every_steps",
    "max_steps",
}


def _deepspeed(value: Any) -> DeepSpeedConfig | None:
    if value is None:
        return None
    payload = _mapping(value, "deepspeed")
    optimizer = payload.pop("optimizer", None)
    scheduler = payload.pop("scheduler", None)
    if optimizer is not None:
        payload["optimizer"] = _strict_dataclass(
            DeepSpeedOptimizerConfig, optimizer, "deepspeed.optimizer"
        )
    if scheduler is not None:
        payload["scheduler"] = _strict_dataclass(
            DeepSpeedSchedulerConfig, scheduler, "deepspeed.scheduler"
        )
    return _strict_dataclass(DeepSpeedConfig, payload, "deepspeed")


def training_job_from_dict(value: Mapping[str, Any]) -> TrainingJobConfig:
    payload = dict(value)
    sections = {
        "model", "data", "training", "freeze", "lora", "trainable_tokens",
        "optimizer", "scheduler", "loss", "gradient_checkpointing",
        "deepspeed", "checkpoint", "evaluation", "artifacts", "dpo", "grpo",
    }
    unknown = sorted(set(payload) - sections)
    if unknown:
        raise ValueError(f"Unknown top-level fields: {', '.join(unknown)}")
    missing = sorted(name for name in ("model", "data", "training") if name not in payload)
    if missing:
        raise ValueError(f"Missing top-level fields: {', '.join(missing)}")

    training = _mapping(payload["training"], "training")
    unknown_training = sorted(set(training) - _TRAINING_SCALARS)
    if unknown_training:
        raise ValueError(f"Unknown training fields: {', '.join(unknown_training)}")
    if "output_dir" not in training:
        raise ValueError("training.output_dir is required")

    freeze = payload.get("freeze")
    if freeze is not None:
        freeze = _mapping(freeze, "freeze")
        if "trainable_prefixes" in freeze:
            freeze["trainable_prefixes"] = tuple(freeze["trainable_prefixes"])
        freeze = _strict_dataclass(FreezeConfig, freeze, "freeze")
    lora = payload.get("lora")
    if lora is not None:
        lora = _mapping(lora, "lora")
        if lora.get("target_modules") is not None:
            lora["target_modules"] = tuple(lora["target_modules"])
        lora = _strict_dataclass(LoRAConfig, lora, "lora")

    config = TrainingConfig(
        **training,
        freeze=freeze,
        lora=lora,
        trainable_tokens=(
            None if payload.get("trainable_tokens") is None else
            _strict_dataclass(TrainableTokensConfig, payload["trainable_tokens"], "trainable_tokens")
        ),
        optimizer=_strict_dataclass(OptimizerConfig, payload.get("optimizer"), "optimizer"),
        scheduler=_strict_dataclass(SchedulerConfig, payload.get("scheduler"), "scheduler"),
        loss=_strict_dataclass(LossConfig, payload.get("loss"), "loss"),
        gradient_checkpointing=_strict_dataclass(
            GradientCheckpointingConfig,
            payload.get("gradient_checkpointing"),
            "gradient_checkpointing",
        ),
        deepspeed=_deepspeed(payload.get("deepspeed")),
        checkpoint=_strict_dataclass(CheckpointConfig, payload.get("checkpoint"), "checkpoint"),
        evaluation=_strict_dataclass(EvaluationConfig, payload.get("evaluation"), "evaluation"),
    )
    config.validate()
    if config.lora is not None and config.freeze is not None:
        raise ValueError(
            "Config-driven LoRA does not support simultaneously training base-model "
            "parameters with freeze; use the Python API for a custom artifact lifecycle"
        )
    data = _strict_dataclass(DataConfig, payload["data"], "data")
    raw_dpo = payload.get("dpo")
    if data.format == "dpo":
        dpo = _strict_dataclass(DPOConfig, raw_dpo, "dpo")
        dpo.validate()
        if config.trainable_tokens is not None:
            raise ValueError("DPO does not support trainable_tokens yet")
        if config.loss.module_path is not None or config.loss.kwargs:
            raise ValueError("DPO owns its loss; custom loss configuration is not supported")
        if config.deepspeed is not None and (
            config.deepspeed.optimizer is not None
            or config.deepspeed.scheduler is not None
        ):
            raise ValueError(
                "DPO uses the Hugging Face optimizer and scheduler; "
                "deepspeed.optimizer/deepspeed.scheduler are not supported"
            )
        if config.evaluation.every_epochs not in {None, 1}:
            raise ValueError(
                "DPO supports evaluation.every_epochs only as 1; "
                "use evaluation.every_steps for another cadence"
            )
    else:
        if raw_dpo is not None:
            raise ValueError("The dpo section requires data.format='dpo'")
        dpo = None

    raw_grpo = payload.get("grpo")
    if data.format == "grpo":
        grpo_payload = _mapping(raw_grpo, "grpo")
        raw_rewards = grpo_payload.pop("rewards", None)
        if not isinstance(raw_rewards, list):
            raise ValueError("grpo.rewards must be a list")
        rewards = tuple(
            _strict_dataclass(GRPORewardConfig, item, f"grpo.rewards[{index}]")
            for index, item in enumerate(raw_rewards)
        )
        grpo = _strict_dataclass(
            GRPOConfig,
            {**grpo_payload, "rewards": rewards},
            "grpo",
        )
        grpo.validate()
        if config.trainable_tokens is not None:
            raise ValueError("GRPO does not support trainable_tokens yet")
        if config.loss.module_path is not None or config.loss.kwargs:
            raise ValueError("GRPO owns its loss; custom loss configuration is not supported")
        if config.deepspeed is not None and (
            config.deepspeed.optimizer is not None
            or config.deepspeed.scheduler is not None
        ):
            raise ValueError(
                "GRPO uses the Hugging Face optimizer and scheduler; "
                "deepspeed.optimizer/deepspeed.scheduler are not supported"
            )
        if config.evaluation.every_epochs not in {None, 1}:
            raise ValueError(
                "GRPO supports evaluation.every_epochs only as 1; "
                "use evaluation.every_steps for another cadence"
            )
        effective_generation_steps = (
            grpo.steps_per_generation
            if grpo.steps_per_generation is not None
            else config.gradient_accumulation_steps
        )
        if (
            config.per_device_batch_size * effective_generation_steps
            % grpo.num_generations
        ):
            raise ValueError(
                "training.per_device_batch_size * effective "
                "steps_per_generation must be divisible by "
                "grpo.num_generations"
            )
    else:
        if raw_grpo is not None:
            raise ValueError("The grpo section requires data.format='grpo'")
        grpo = None

    artifacts = _strict_dataclass(ArtifactConfig, payload.get("artifacts"), "artifacts")
    if artifacts.save == "adapter" and config.lora is None and config.trainable_tokens is None:
        raise ValueError("artifacts.save='adapter' requires lora or trainable_tokens")
    if artifacts.save == "model" and (config.lora is not None or config.trainable_tokens is not None):
        raise ValueError("artifacts.save='model' is not supported for adapter training; use 'adapter'")
    return TrainingJobConfig(
        model=_strict_dataclass(ModelConfig, payload["model"], "model"),
        data=data,
        training=config,
        artifacts=artifacts,
        dpo=dpo,
        grpo=grpo,
    )


def load_training_job(path: str | Path) -> TrainingJobConfig:
    return training_job_from_dict(load_config_mapping(path))
