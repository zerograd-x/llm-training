from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from llm_training import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
)
from llm_training.loss import DefaultCrossEntropyLoss, shift_causal_labels
from llm_training.trainer import EpochSeededRandomSampler


class ToyLM(nn.Module):
    def __init__(self, dropout=0.0):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(8, 16)
        self.seen = []

    def forward(self, input_ids):
        self.seen.extend(input_ids.detach().cpu().tolist())
        return SimpleNamespace(
            logits=self.head(self.dropout(self.embedding(input_ids)))
        )


def rows():
    return [
        dict(
            input_ids=torch.tensor([1, i + 3, 2]),
            labels=torch.tensor([1, i + 3 if i % 2 else -100, 2]),
        )
        for i in range(5)
    ]


def collate(batch):
    return {key: torch.stack([row[key] for row in batch]) for key in batch[0]}


def test_partial_accumulation_matches_mean_of_microbatch_gradients(tmp_path):
    torch.manual_seed(31)
    reference = ToyLM()
    model = deepcopy(reference)
    data = rows()
    config = TrainingConfig(
        output_dir=str(tmp_path),
        seed=123,
        gradient_accumulation_steps=2,
        max_grad_norm=None,
        scheduler=SchedulerConfig(name="constant", warmup_steps=0),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        train_dataset=data,
        collate_fn=collate,
        optimizer=optimizer,
        config=config,
    )
    trainer.train()
    manual = torch.optim.SGD(reference.parameters(), lr=0.1)
    order = list(EpochSeededRandomSampler(data, seed=123))
    loss_fn = DefaultCrossEntropyLoss()
    for offset in range(0, len(order), 2):
        group = order[offset : offset + 2]
        manual.zero_grad()
        for index in group:
            batch = collate([data[index]])
            loss = loss_fn(
                reference(batch["input_ids"]).logits,
                shift_causal_labels(batch["labels"]),
            )
            (loss / len(group)).backward()
        manual.step()
    assert trainer.state.global_step == 3
    for actual, expected in zip(
        trainer.accelerator.unwrap_model(trainer.model).parameters(),
        reference.parameters(),
    ):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


class Stop(RuntimeError):
    pass


class Interrupt(Trainer):
    stop_step = 1

    def save_checkpoint(self):
        path = super().save_checkpoint()
        if self.state.global_step == self.stop_step:
            raise Stop()
        return path


@pytest.mark.parametrize("stop_step", [1, 3, 4, 6])
def test_uneven_three_epoch_resume(tmp_path, stop_step):
    torch.manual_seed(8)
    initial = ToyLM(0.3)

    def make(cls, name, resume=None):
        return cls(
            model=deepcopy(initial),
            train_dataset=rows(),
            collate_fn=collate,
            config=TrainingConfig(
                output_dir=str(tmp_path / name),
                num_epochs=3,
                seed=123,
                gradient_accumulation_steps=2,
                optimizer=OptimizerConfig(learning_rate=0.01, weight_decay=0),
                scheduler=SchedulerConfig(name="linear", warmup_steps=0),
                checkpoint=CheckpointConfig(save_every_steps=1, resume_from=resume),
            ),
        )

    baseline = make(Trainer, "baseline")
    baseline.train()
    interrupted = make(Interrupt, "resumed")
    interrupted.stop_step = stop_step
    with pytest.raises(Stop):
        interrupted.train()
    resumed = make(Trainer, "resumed", "latest")
    resumed.train()
    unwrap = resumed.accelerator.unwrap_model
    assert resumed.state == baseline.state
    assert resumed.state.global_step == 9
    assert (
        unwrap(baseline.model).seen
        == unwrap(interrupted.model).seen + unwrap(resumed.model).seen
    )
    for a, b in zip(
        unwrap(resumed.model).parameters(), unwrap(baseline.model).parameters()
    ):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert resumed.scheduler.state_dict() == baseline.scheduler.state_dict()
