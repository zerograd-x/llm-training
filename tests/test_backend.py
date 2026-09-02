from types import SimpleNamespace

import pytest
from torch import nn

import llm_training.backend as backend
from llm_training import (
    DeepSpeedConfig,
    GradientCheckpointingConfig,
    TrainingConfig,
)


class CheckpointableModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(__import__("torch").ones(1))
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


def test_build_accelerator_maps_general_deepspeed_config(monkeypatch, tmp_path) -> None:
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
            zero3_save_16bit_model=True,
        ),
    )

    result = backend.build_accelerator(config)

    assert isinstance(result, FakeAccelerator)
    assert captured["plugin"] == {
        "gradient_accumulation_steps": 4,
        "zero_stage": 3,
        "offload_optimizer_device": "cpu",
        "offload_param_device": "cpu",
        "offload_optimizer_nvme_path": None,
        "offload_param_nvme_path": None,
        "zero3_init_flag": False,
        "zero3_save_16bit_model": True,
    }
    assert captured["accelerator"] == {
        "gradient_accumulation_steps": 4,
        "mixed_precision": "bf16",
        "deepspeed_plugin": captured["accelerator"]["deepspeed_plugin"],
    }
    assert isinstance(captured["accelerator"]["deepspeed_plugin"], FakePlugin)


def test_deepspeed_request_fails_early_when_extra_is_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(backend.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="deepspeed"):
        backend.build_accelerator(
            TrainingConfig(
                output_dir=str(tmp_path),
                deepspeed=DeepSpeedConfig(zero_stage=2),
            )
        )
