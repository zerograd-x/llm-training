"""Resume must preserve stochastic training, not only reload checkpoint tensors."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_training import CheckpointConfig, OptimizerConfig, SchedulerConfig, Trainer, TrainingConfig


class DropoutLM(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8, 16)
        self.seen = []

    def forward(self, input_ids):
        self.seen.extend(input_ids.detach().cpu().tolist())
        return SimpleNamespace(logits=self.head(self.dropout(self.embedding(input_ids))))


class Interrupted(RuntimeError):
    pass


class InterruptAtCheckpoint(Trainer):
    stop_step = 1

    def save_checkpoint(self):
        path = super().save_checkpoint()
        if self.state.global_step == self.stop_step:
            raise Interrupted()
        return path


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


@pytest.mark.parametrize("dropout", [0.0, 0.4])
@pytest.mark.parametrize("stop_step", [1, 3])
def test_mid_epoch_resume_matches_uninterrupted_dropout_training(tmp_path, dropout, stop_step):
    torch.manual_seed(17)
    initial = DropoutLM(dropout)
    rows = [{"input_ids": torch.tensor([1, i, 2]), "labels": torch.tensor([1, i, 2])}
            for i in range(3, 7)]

    def make(kind, output, resume=None):
        return kind(model=deepcopy(initial), train_dataset=rows, collate_fn=collate,
                    config=TrainingConfig(
                        output_dir=str(output), seed=123, num_epochs=2,
                        per_device_batch_size=1, gradient_accumulation_steps=2,
                        optimizer=OptimizerConfig(learning_rate=0.01, weight_decay=0),
                        scheduler=SchedulerConfig(name="linear", warmup_steps=0),
                        checkpoint=CheckpointConfig(save_every_steps=1, resume_from=resume)))

    continuous = make(Trainer, tmp_path / "continuous")
    continuous.train()
    expected = {name: p.detach().cpu().clone() for name, p in
                continuous.accelerator.unwrap_model(continuous.model).named_parameters()}
    interrupted = make(InterruptAtCheckpoint, tmp_path / "resume")
    interrupted.stop_step = stop_step
    with pytest.raises(Interrupted):
        interrupted.train()
    resumed = make(Trainer, tmp_path / "resume", "latest")
    assert (resumed.state.global_step, resumed.state.batch_in_epoch) == (stop_step, 2)
    resumed.train()
    assert resumed.state.global_step == continuous.state.global_step == 4
    unwrap = resumed.accelerator.unwrap_model
    assert unwrap(continuous.model).seen == unwrap(interrupted.model).seen + unwrap(resumed.model).seen
    for name, parameter in resumed.accelerator.unwrap_model(resumed.model).named_parameters():
        torch.testing.assert_close(parameter.detach().cpu(), expected[name], rtol=0, atol=0)
