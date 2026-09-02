from __future__ import annotations

import importlib.util

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


def build_accelerator(config: TrainingConfig) -> Accelerator:
    """Build the Accelerate runtime for the configured training backend."""

    deepspeed_plugin = None
    if config.deepspeed is not None:
        if importlib.util.find_spec("deepspeed") is None:
            raise ImportError(
                "DeepSpeed was requested but is not installed. "
                "Install llm-training with the 'deepspeed' extra."
            )

        ds = config.deepspeed
        deepspeed_plugin = DeepSpeedPlugin(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            zero_stage=ds.zero_stage,
            offload_optimizer_device=ds.offload_optimizer_device,
            offload_param_device=ds.offload_param_device,
            # The model is currently constructed before Trainer/Accelerator, so
            # ZeRO-3 init-time partitioning is intentionally not claimed here.
            zero3_init_flag=False,
            zero3_save_16bit_model=(
                ds.zero3_save_16bit_model if ds.zero_stage == 3 else None
            ),
        )

    return Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        deepspeed_plugin=deepspeed_plugin,
    )
