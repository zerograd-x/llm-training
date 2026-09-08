from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from llm_training import (
    BaseLoss,
    CheckpointConfig,
    DeepSpeedConfig,
    DistributedRuntime,
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    build_training_plan,
)
from llm_training.checkpoint import find_latest_checkpoint


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])},
            {"input_ids": torch.tensor([2, 3, 4]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([3, 4, 5]), "labels": torch.tensor([3, 4, 5])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([4, 5, 6])},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


class FullyMaskedDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int):
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "labels": torch.tensor([-100, -100, -100]),
        }


def stack_collator(rows):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in rows]),
        "labels": torch.stack([row["labels"] for row in rows]),
    }


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16)

    def forward(self, input_ids):
        logits = self.head(self.embed(input_ids))
        return SimpleNamespace(logits=logits)


class CountingLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, logits, labels):
        self.calls += 1
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
        )


def build_config(tmp_path, *, loss=None) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        per_device_batch_size=2,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        loss=loss or LossConfig(),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2),
    )


def test_training_plan_matches_distributed_step_accounting(tmp_path) -> None:
    config = TrainingConfig(
        output_dir=str(tmp_path / "run"),
        num_epochs=1,
        per_device_batch_size=8,
        gradient_accumulation_steps=1,
        scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.03),
        checkpoint=CheckpointConfig(save_every_steps=500, keep_last=2),
        deepspeed=DeepSpeedConfig(zero_stage=3),
    )

    plan = build_training_plan(config, train_examples=50_000, world_size=4)

    assert plan.global_batch_size == 32
    assert plan.batches_per_rank == 1563
    assert plan.optimizer_steps == 1563
    assert plan.scheduler_steps == 6252
    assert plan.warmup_optimizer_steps == 47
    assert plan.expected_checkpoints == 4
    assert plan.retained_checkpoints == 2
    assert {warning.code for warning in plan.warnings} == {
        "zero3_initial_load_unpartitioned"
    }


def test_trainer_runs_and_saves_checkpoint(tmp_path) -> None:
    trainer = Trainer(
        model=TinyLM(),
        train_dataset=TinyDataset(),
        collate_fn=stack_collator,
        config=build_config(tmp_path),
    )

    state = trainer.train()

    assert state.global_step == 2
    assert state.epoch == 1
    assert state.batch_in_epoch == 0
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-00000002"


def test_trainer_reuses_explicit_distributed_runtime(tmp_path) -> None:
    config = build_config(tmp_path)
    runtime = DistributedRuntime(config)
    trainer = Trainer(
        model=TinyLM(),
        train_dataset=TinyDataset(),
        collate_fn=stack_collator,
        config=config,
        runtime=runtime,
    )

    assert trainer.runtime is runtime
    assert trainer.accelerator is runtime.accelerator


def test_trainer_rejects_runtime_built_from_different_config(tmp_path) -> None:
    runtime_config = build_config(tmp_path / "runtime")
    trainer_config = build_config(tmp_path / "trainer")
    runtime = DistributedRuntime(runtime_config)

    with pytest.raises(ValueError, match="must match"):
        Trainer(
            model=TinyLM(),
            train_dataset=TinyDataset(),
            collate_fn=stack_collator,
            config=trainer_config,
            runtime=runtime,
        )


def test_trainer_uses_runtime_custom_loss(tmp_path) -> None:
    loss_fn = CountingLoss()
    trainer = Trainer(
        model=TinyLM(),
        train_dataset=TinyDataset(),
        collate_fn=stack_collator,
        config=build_config(tmp_path),
        loss_fn=loss_fn,
    )

    trainer.train()

    assert loss_fn.calls == 2


def test_trainer_rejects_runtime_and_config_loss_together(tmp_path) -> None:
    config = build_config(
        tmp_path,
        loss=LossConfig(
            module_path="llm_training.loss.DefaultCrossEntropyLoss",
        ),
    )

    with pytest.raises(ValueError, match="either Trainer"):
        Trainer(
            model=TinyLM(),
            train_dataset=TinyDataset(),
            collate_fn=stack_collator,
            config=config,
            loss_fn=CountingLoss(),
        )


def test_trainer_fails_before_update_on_nonfinite_loss(tmp_path) -> None:
    trainer = Trainer(
        model=TinyLM(),
        train_dataset=FullyMaskedDataset(),
        collate_fn=stack_collator,
        config=TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=1,
            per_device_batch_size=1,
            optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
            scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        ),
    )

    with pytest.raises(FloatingPointError, match="Non-finite loss"):
        trainer.train()
