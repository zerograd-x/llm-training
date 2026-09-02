from types import SimpleNamespace

import pytest
import torch
from torch import nn

import llm_training.backend as backend
from llm_training import (
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    GradientCheckpointingConfig,
    TrainingConfig,
)


class CheckpointableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.config = SimpleNamespace(use_cache=True)
        self.calls = []

    def gradient_checkpointing_enable(self, **kwargs) -> None:
        self.calls.append(kwargs)


def test_gradient_checkpointing_disables_cache_and_forwards_reentrant() -> None:
    model = CheckpointableModel()

    backend.configure_gradient_checkpointing(
        model,
        GradientCheckpointingConfig(enabled=True, use_reentrant=False),
    )

    assert model.config.use_cache is False
    assert model.calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]


def test_gradient_checkpointing_uses_model_default_when_reentrant_unspecified() -> None:
    model = CheckpointableModel()

    backend.configure_gradient_checkpointing(
        model,
        GradientCheckpointingConfig(enabled=True),
    )

    assert model.calls == [{}]


def test_gradient_checkpointing_rejects_unsupported_model() -> None:
    with pytest.raises(TypeError, match="gradient_checkpointing_enable"):
        backend.configure_gradient_checkpointing(
            nn.Linear(2, 2),
            GradientCheckpointingConfig(enabled=True),
        )


def test_deepspeed_config_validates_offload_requirements(tmp_path) -> None:
    with pytest.raises(ValueError, match="parameter offload requires"):
        TrainingConfig(
            output_dir=str(tmp_path),
            deepspeed=DeepSpeedConfig(
                zero_stage=2,
                offload_param_device="cpu",
            ),
        ).validate()

    with pytest.raises(ValueError, match="NVMe optimizer offload"):
        TrainingConfig(
            output_dir=str(tmp_path),
            deepspeed=DeepSpeedConfig(
                zero_stage=3,
                offload_optimizer_device="nvme",
            ),
        ).validate()


def test_deepspeed_owned_optimizer_requires_deepspeed_scheduler(tmp_path) -> None:
    with pytest.raises(ValueError, match="requires a DeepSpeed-owned scheduler"):
        TrainingConfig(
            output_dir=str(tmp_path),
            deepspeed=DeepSpeedConfig(
                optimizer=DeepSpeedOptimizerConfig(type="AdamW"),
            ),
        ).validate()


def test_deepspeed_owned_configs_cannot_override_shared_settings(tmp_path) -> None:
    with pytest.raises(ValueError, match="must not override shared optimizer"):
        TrainingConfig(
            output_dir=str(tmp_path),
            deepspeed=DeepSpeedConfig(
                optimizer=DeepSpeedOptimizerConfig(
                    type="AdamW",
                    params={"lr": 1e-3},
                ),
                scheduler=DeepSpeedSchedulerConfig(),
            ),
        ).validate()

    with pytest.raises(ValueError, match="must not override shared scheduler"):
        TrainingConfig(
            output_dir=str(tmp_path),
            deepspeed=DeepSpeedConfig(
                scheduler=DeepSpeedSchedulerConfig(
                    params={"warmup_num_steps": 123},
                ),
            ),
        ).validate()


def test_build_deepspeed_config_maps_owned_optimizer_and_scheduler(tmp_path) -> None:
    config = TrainingConfig(
        output_dir=str(tmp_path),
        gradient_accumulation_steps=4,
        max_grad_norm=0.5,
        deepspeed=DeepSpeedConfig(
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
            optimizer=DeepSpeedOptimizerConfig(
                type="AdamW",
                params={"torch_adam": True},
            ),
            scheduler=DeepSpeedSchedulerConfig(
                type="WarmupDecayLR",
                params={"warmup_type": "linear"},
            ),
        ),
    )
    config.validate()

    result = backend.build_deepspeed_config(config)

    assert result["zero_optimization"] == {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
        "stage3_gather_16bit_weights_on_model_save": True,
    }
    assert result["gradient_accumulation_steps"] == 4
    assert result["gradient_clipping"] == 0.5
    assert result["optimizer"] == {
        "type": "AdamW",
        "params": {
            "torch_adam": True,
            "lr": "auto",
            "weight_decay": "auto",
            "betas": [0.9, 0.95],
            "eps": 1e-8,
        },
    }
    assert result["scheduler"] == {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_type": "linear",
            "warmup_min_lr": 0.0,
            "warmup_max_lr": "auto",
            "warmup_num_steps": "auto",
            "total_num_steps": "auto",
        },
    }


def test_build_accelerator_uses_single_deepspeed_config(monkeypatch, tmp_path) -> None:
    captured = {}

    class FakePlugin:
        def __init__(self, **kwargs) -> None:
            captured["plugin"] = kwargs

    class FakeAccelerator:
        def __init__(self, **kwargs) -> None:
            captured["accelerator"] = kwargs

    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(backend, "DeepSpeedPlugin", FakePlugin)
    monkeypatch.setattr(backend, "Accelerator", FakeAccelerator)

    config = TrainingConfig(
        output_dir=str(tmp_path),
        gradient_accumulation_steps=4,
        mixed_precision="bf16",
        deepspeed=DeepSpeedConfig(
            zero_stage=3,
            offload_optimizer_device="cpu",
            offload_param_device="cpu",
        ),
    )

    result = backend.build_accelerator(config)

    assert isinstance(result, FakeAccelerator)
    assert captured["plugin"]["hf_ds_config"] == backend.build_deepspeed_config(config)
    assert captured["plugin"]["zero3_init_flag"] is False
    accelerator_kwargs = captured["accelerator"]
    assert accelerator_kwargs["gradient_accumulation_steps"] == 4
    assert accelerator_kwargs["mixed_precision"] == "bf16"
    assert isinstance(accelerator_kwargs["deepspeed_plugin"], FakePlugin)


def test_deepspeed_request_fails_early_when_extra_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="deepspeed"):
        backend.build_accelerator(
            TrainingConfig(
                output_dir=str(tmp_path),
                deepspeed=DeepSpeedConfig(zero_stage=2),
            )
        )
