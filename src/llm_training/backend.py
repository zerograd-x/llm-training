from __future__ import annotations

import importlib.util
from typing import Any

from accelerate import Accelerator, DeepSpeedPlugin
from torch import nn

from .config import GradientCheckpointingConfig, TrainingConfig


def configure_gradient_checkpointing(
    model: nn.Module,
    config: GradientCheckpointingConfig,
) -> None:
    """Enable model-native gradient checkpointing before backend preparation."""

    if not config.enabled:
        return

    enable = getattr(model, "gradient_checkpointing_enable", None)
    if enable is None:
        raise TypeError(
            "gradient checkpointing was requested, but the model does not expose "
            "gradient_checkpointing_enable()"
        )

    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "use_cache"):
        model_config.use_cache = False

    if config.use_reentrant is None:
        enable()
        return

    try:
        enable(
            gradient_checkpointing_kwargs={
                "use_reentrant": config.use_reentrant,
            }
        )
    except TypeError as exc:
        raise TypeError(
            "This model does not accept gradient_checkpointing_kwargs, but "
            "gradient_checkpointing.use_reentrant was explicitly configured"
        ) from exc


def _offload_config(device: str, nvme_path: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"device": device}
    if device == "nvme":
        result["nvme_path"] = nvme_path
    return result


def build_deepspeed_config(config: TrainingConfig) -> dict[str, Any]:
    """Translate the general training config into one DeepSpeed config dict."""

    if config.deepspeed is None:
        raise ValueError("DeepSpeed config requested without TrainingConfig.deepspeed")

    ds = config.deepspeed
    zero: dict[str, Any] = {
        "stage": ds.zero_stage,
    }
    if ds.zero_stage >= 2:
        zero["offload_optimizer"] = _offload_config(
            ds.offload_optimizer_device,
            ds.offload_optimizer_nvme_path,
        )
    if ds.zero_stage == 3:
        zero["offload_param"] = _offload_config(
            ds.offload_param_device,
            ds.offload_param_nvme_path,
        )
        zero["stage3_gather_16bit_weights_on_model_save"] = (
            ds.zero3_save_16bit_model
        )

    result: dict[str, Any] = {
        "zero_optimization": zero,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
    }
    if config.max_grad_norm is not None:
        result["gradient_clipping"] = config.max_grad_norm

    if ds.optimizer is not None:
        params = dict(ds.optimizer.params)
        params["lr"] = "auto"
        params["weight_decay"] = "auto"
        if ds.optimizer.type.lower() in {"adam", "adamw"}:
            params["betas"] = [config.optimizer.beta1, config.optimizer.beta2]
            params["eps"] = config.optimizer.eps
        result["optimizer"] = {
            "type": ds.optimizer.type,
            "params": params,
        }

    if ds.scheduler is not None:
        params = dict(ds.scheduler.params)
        scheduler_type = ds.scheduler.type.lower()
        if scheduler_type in {"warmuplr", "warmupdecaylr"}:
            params.setdefault("warmup_min_lr", 0.0)
            params["warmup_max_lr"] = "auto"
            params["warmup_num_steps"] = "auto"
        if scheduler_type == "warmupdecaylr":
            params["total_num_steps"] = "auto"
        result["scheduler"] = {
            "type": ds.scheduler.type,
            "params": params,
        }

    return result


def build_accelerator(config: TrainingConfig) -> Accelerator:
    """Build the Accelerate runtime for the configured training backend."""

    deepspeed_plugin = None
    if config.deepspeed is not None:
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "DeepSpeed was requested but is not installed. "
                "Install llm-training with the 'deepspeed' extra."
            )

        deepspeed_plugin = DeepSpeedPlugin(
            hf_ds_config=build_deepspeed_config(config),
            # The model is currently constructed before Trainer/Accelerator, so
            # ZeRO-3 init-time partitioning is intentionally not claimed here.
            zero3_init_flag=False,
        )

    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )
