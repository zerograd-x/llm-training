import pytest
import torch
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    DefaultCrossEntropyLoss,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    load_causal_lm,
    shift_causal_labels,
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


@pytest.mark.integration
def test_real_hf_model_trains_one_step(tmp_path) -> None:
    components = load_causal_lm("sshleifer/tiny-gpt2")
    encoded = components.tokenizer(
        ["hello world", "a tiny training example"],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100

    components.model.eval()
    with torch.no_grad():
        internal_loss = components.model(**encoded, labels=labels).loss
        outputs = components.model(**encoded)
        external_loss = DefaultCrossEntropyLoss()(
            outputs.logits,
            shift_causal_labels(labels),
        )
    torch.testing.assert_close(external_loss, internal_loss.float())

    rows = []
    for index in range(encoded["input_ids"].shape[0]):
        rows.append(
            {
                "input_ids": encoded["input_ids"][index],
                "attention_mask": encoded["attention_mask"][index],
                "labels": labels[index],
            }
        )

    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=TensorRows(rows),
        collate_fn=stack_collator,
        config=TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=1,
            per_device_batch_size=2,
            log_every_steps=1,
            optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
            scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
            checkpoint=CheckpointConfig(save_every_steps=1, keep_last=1),
        ),
    )

    state = trainer.train()

    assert state.global_step == 1
    assert (tmp_path / "checkpoint-00000001" / "trainer_state.json").is_file()
