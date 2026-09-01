from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from llm_training import CheckpointConfig, OptimizerConfig, SchedulerConfig, Trainer, TrainingConfig
from llm_training.checkpoint import find_latest_checkpoint


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([2, 3, 4])},
            {"input_ids": torch.tensor([2, 3, 4]), "labels": torch.tensor([3, 4, 5])},
            {"input_ids": torch.tensor([3, 4, 5]), "labels": torch.tensor([4, 5, 6])},
            {"input_ids": torch.tensor([4, 5, 6]), "labels": torch.tensor([5, 6, 7])},
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        return self.rows[index]


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

    def forward(self, input_ids, labels):
        logits = self.head(self.embed(input_ids))
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        return SimpleNamespace(loss=loss, logits=logits)


def test_trainer_runs_and_saves_checkpoint(tmp_path) -> None:
    config = TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        per_device_batch_size=2,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2),
    )
    trainer = Trainer(
        model=TinyLM(),
        train_dataset=TinyDataset(),
        collate_fn=stack_collator,
        config=config,
    )

    state = trainer.train()

    assert state.global_step == 2
    assert state.epoch == 1
    assert state.batch_in_epoch == 0
    latest = find_latest_checkpoint(tmp_path)
    assert latest is not None
    assert latest.name == "checkpoint-00000002"
