import pytest
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
    load_causal_lm,
)


llm_training_data = pytest.importorskip(
    "llm_training_data",
    reason=(
        "Install the sibling llm-training-data repository to run the cross-repo "
        "integration test, e.g. `pip install -e ../llm-training-data`."
    ),
)


class PromptCompletionRows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@pytest.mark.integration
def test_llm_training_data_collator_trains_real_hf_model(tmp_path) -> None:
    components = load_causal_lm("sshleifer/tiny-gpt2")

    rows = [
        {"prompt": "Question: 2 + 2?", "completion": " 4"},
        {"prompt": "Question: capital of France?", "completion": " Paris"},
    ]

    data_collator = llm_training_data.build_sft_collator(
        components.tokenizer,
        max_sequence_length=32,
        pad_to_multiple_of=8,
    )
    collate_fn = adapt_column_batch_collator(data_collator)

    batch = collate_fn(rows)
    assert batch["input_ids"].shape == batch["labels"].shape
    assert (batch["labels"] == -100).any()
    assert (batch["labels"] != -100).any()

    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=PromptCompletionRows(rows),
        collate_fn=collate_fn,
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
