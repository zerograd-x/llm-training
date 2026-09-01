import pytest
import torch
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    load_causal_lm,
)


class TensorRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def stack_collator(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


class IntentionalInterruption(RuntimeError):
    pass


class FailOnSecondBatch:
    """Let step 1 finish and checkpoint, then simulate a training-job failure."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, rows):
        self.calls += 1
        if self.calls == 2:
            raise IntentionalInterruption("simulated worker failure after the first step")
        return stack_collator(rows)


def build_rows(tokenizer):
    encoded = tokenizer(
        [
            "alpha beta",
            "gamma delta",
            "epsilon zeta",
            "eta theta",
        ],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    return [
        {
            "input_ids": encoded["input_ids"][index],
            "attention_mask": encoded["attention_mask"][index],
            "labels": labels[index],
        }
        for index in range(encoded["input_ids"].shape[0])
    ]


def build_config(output_dir, *, resume_from=None):
    return TrainingConfig(
        output_dir=str(output_dir),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        checkpoint=CheckpointConfig(
            save_every_steps=1,
            keep_last=2,
            resume_from=resume_from,
        ),
    )


def snapshot_parameters(trainer):
    model = trainer.accelerator.unwrap_model(trainer.model)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


@pytest.mark.integration
def test_real_hf_training_resumes_mid_epoch(tmp_path) -> None:
    first = load_causal_lm("sshleifer/tiny-gpt2")
    rows = build_rows(first.tokenizer)

    interrupted = Trainer(
        model=first.model,
        tokenizer=first.tokenizer,
        train_dataset=TensorRows(rows),
        collate_fn=FailOnSecondBatch(),
        config=build_config(tmp_path),
    )

    with pytest.raises(IntentionalInterruption):
        interrupted.train()

    assert interrupted.state.global_step == 1
    assert interrupted.state.epoch == 0
    assert interrupted.state.batch_in_epoch == 1
    assert (tmp_path / "checkpoint-00000001" / "trainer_state.json").is_file()

    checkpoint_parameters = snapshot_parameters(interrupted)

    fresh = load_causal_lm("sshleifer/tiny-gpt2")
    resumed = Trainer(
        model=fresh.model,
        tokenizer=fresh.tokenizer,
        train_dataset=TensorRows(rows),
        collate_fn=stack_collator,
        config=build_config(tmp_path, resume_from="latest"),
    )

    assert resumed.state.global_step == 1
    assert resumed.state.epoch == 0
    assert resumed.state.batch_in_epoch == 1

    restored_parameters = snapshot_parameters(resumed)
    for name, expected in checkpoint_parameters.items():
        torch.testing.assert_close(restored_parameters[name], expected)

    before_second_step = snapshot_parameters(resumed)
    final_state = resumed.train()
    after_second_step = snapshot_parameters(resumed)

    assert final_state.global_step == 2
    assert final_state.epoch == 1
    assert final_state.batch_in_epoch == 0
    assert (tmp_path / "checkpoint-00000002" / "trainer_state.json").is_file()
    assert any(
        not torch.equal(before_second_step[name], after_second_step[name])
        for name in before_second_step
    )
