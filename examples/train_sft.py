from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    DistributedRuntime,
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
)
from llm_training_data import build_sft_collator


class PromptDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


config = TrainingConfig(
    output_dir="./outputs/sft-demo",
    num_epochs=1,
    per_device_batch_size=2,
    checkpoint=CheckpointConfig(save_every_steps=10, keep_last=2),
)
runtime = DistributedRuntime(config)
components = runtime.load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")

rows = [
    {"prompt": "What is 2 + 2?", "completion": "4"},
    {"prompt": "Name a primary color.", "completion": "Blue"},
]

data_collator = build_sft_collator(
    components.tokenizer,
    max_sequence_length=128,
)

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=PromptDataset(rows),
    collate_fn=adapt_column_batch_collator(data_collator),
    config=config,
    runtime=runtime,
)

trainer.train()
trainer.export_model()
