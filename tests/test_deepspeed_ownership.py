from copy import deepcopy

import pytest
import torch
from accelerate.utils.deepspeed import DummyOptim, DummyScheduler
from torch import nn
from torch.optim import AdamW

from llm_training import (
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from llm_training.optimizer import build_training_optimizer
from llm_training.scheduler import bind_scheduler_optimizer, build_training_scheduler


def test_default_deepspeed_path_keeps_torch_optimizer_and_scheduler(tmp_path) -> None:
    model = nn.Linear(4, 2)
    config = TrainingConfig(
        output_dir=str(tmp_path),
        optimizer=OptimizerConfig(learning_rate=3e-4, weight_decay=0.2),
        scheduler=SchedulerConfig(name="linear", warmup_steps=2),
        deepspeed=DeepSpeedConfig(zero_stage=3),
    )

    optimizer = build_training_optimizer(model, config)
    assert isinstance(optimizer, AdamW)
    assert not isinstance(optimizer, DummyOptim)
    assert optimizer.defaults["lr"] == 3e-4

    scheduler = build_training_scheduler(
        optimizer,
        config,
        num_training_steps=10,
    )

    assert not isinstance(scheduler, DummyScheduler)


def test_deepspeed_owned_optimizer_and_scheduler_use_dummy_placeholders(tmp_path) -> None:
    model = nn.Linear(4, 2)
    config = TrainingConfig(
        output_dir=str(tmp_path),
        optimizer=OptimizerConfig(learning_rate=2e-5, weight_decay=0.1),
        scheduler=SchedulerConfig(warmup_ratio=0.2),
        deepspeed=DeepSpeedConfig(
            zero_stage=3,
            optimizer=DeepSpeedOptimizerConfig(type="AdamW"),
            scheduler=DeepSpeedSchedulerConfig(type="WarmupDecayLR"),
        ),
    )
    config.validate()

    optimizer = build_training_optimizer(model, config)
    scheduler = build_training_scheduler(
        optimizer,
        config,
        num_training_steps=20,
    )

    assert isinstance(optimizer, DummyOptim)
    assert optimizer.lr == 2e-5
    assert optimizer.weight_decay == 0.1
    assert isinstance(scheduler, DummyScheduler)
    assert scheduler.optimizer is optimizer
    assert scheduler.warmup_num_steps == 4
    assert scheduler.total_num_steps == 20


def test_torch_optimizer_can_use_deepspeed_owned_scheduler(tmp_path) -> None:
    model = nn.Linear(4, 2)
    config = TrainingConfig(
        output_dir=str(tmp_path),
        optimizer=OptimizerConfig(learning_rate=1e-4),
        scheduler=SchedulerConfig(warmup_steps=3),
        deepspeed=DeepSpeedConfig(
            zero_stage=2,
            scheduler=DeepSpeedSchedulerConfig(type="WarmupLR"),
        ),
    )
    config.validate()

    optimizer = build_training_optimizer(model, config)
    scheduler = build_training_scheduler(
        optimizer,
        config,
        num_training_steps=12,
    )

    assert isinstance(optimizer, torch.optim.Optimizer)
    assert not isinstance(optimizer, DummyOptim)
    assert isinstance(scheduler, DummyScheduler)
    assert scheduler.optimizer is optimizer
    assert scheduler.warmup_num_steps == 3
    assert scheduler.total_num_steps == 12


def test_deepspeed_warmup_scheduler_rejects_zero_warmup(tmp_path) -> None:
    model = nn.Linear(4, 2)
    config = TrainingConfig(
        output_dir=str(tmp_path),
        scheduler=SchedulerConfig(warmup_steps=0),
        deepspeed=DeepSpeedConfig(
            zero_stage=2,
            scheduler=DeepSpeedSchedulerConfig(type="WarmupDecayLR"),
        ),
    )

    optimizer = build_training_optimizer(model, config)
    with pytest.raises(ValueError, match="requires at least one warmup step"):
        build_training_scheduler(
            optimizer,
            config,
            num_training_steps=2,
        )


def test_scheduler_updates_replacement_optimizer_after_state_restore(tmp_path) -> None:
    parameter = nn.Parameter(torch.ones(2))
    original = AdamW([parameter], lr=0.01)
    config = TrainingConfig(
        output_dir=str(tmp_path), scheduler=SchedulerConfig(name="linear", warmup_steps=0)
    )
    scheduler = build_training_scheduler(original, config, num_training_steps=2)
    replacement = AdamW(original.param_groups, lr=0.01)
    assert original.param_groups[0] is replacement.param_groups[0]
    bind_scheduler_optimizer(scheduler, replacement)

    parameter.grad = torch.ones_like(parameter)
    replacement.step()
    scheduler.step()
    assert replacement.param_groups[0]["lr"] == 0.005

    saved = deepcopy(replacement.state_dict())
    previous_group = replacement.param_groups[0]
    replacement.load_state_dict(saved)
    assert replacement.param_groups[0] is not previous_group
    parameter.grad = torch.ones_like(parameter)
    replacement.step()
    scheduler.step()
    assert replacement.param_groups[0]["lr"] == 0

