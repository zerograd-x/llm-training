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


def build_rows(tokenizer):
    encoded = tokenizer(
        ["gpu bf16 smoke test", "tiny causal lm training"],
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


@pytest.mark.gpu
def test_real_hf_model_trains_on_cuda_bf16(tmp_path) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support BF16")

    components = load_causal_lm("sshleifer/tiny-gpt2")
    rows = build_rows(components.tokenizer)

    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=TensorRows(rows),
        collate_fn=stack_collator,
        config=TrainingConfig(
            output_dir=str(tmp_path),
            num_epochs=1,
            per_device_batch_size=2,
            mixed_precision="bf16",
            log_every_steps=1,
            optimizer=OptimizerConfig(learning_rate=1e-4, weight_decay=0.0),
            scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
            checkpoint=CheckpointConfig(save_every_steps=1, keep_last=1),
        ),
    )

    assert trainer.accelerator.device.type == "cuda"
    assert next(trainer.model.parameters()).device.type == "cuda"
    assert trainer.accelerator.mixed_precision == "bf16"

    before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.accelerator.unwrap_model(trainer.model).named_parameters()
    }

    state = trainer.train()

    after = {
        name: parameter.detach().cpu().clone()
        for name, parameter in trainer.accelerator.unwrap_model(trainer.model).named_parameters()
    }

    assert state.global_step == 1
    assert (tmp_path / "checkpoint-00000001" / "trainer_state.json").is_file()
    assert any(not torch.equal(before[name], after[name]) for name in before)
