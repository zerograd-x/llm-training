# llm-training

A small, opinionated training runtime for causal language models.

The project is intentionally split from [`llm-training-data`](https://github.com/zerograd-x/llm-training-data):

- **llm-training-data** owns rendering, tokenization, label construction, padding, and sequence packing.
- **llm-training** owns model loading, trainable-parameter policy, optimization, checkpoint/resume, and the training loop.
- **model-artifact-transfer** can publish completed model artifacts after training.

Dependency direction is one-way: `llm-training` may consume `llm-training-data`; `llm-training-data` must not depend on this package.

## v0 scope

The first version focuses on one golden path:

- Hugging Face decoder-only causal LMs
- PyTorch + Hugging Face Accelerate
- full fine-tuning or selective layer freezing
- AdamW + Hugging Face schedulers
- deterministic epoch sampling
- checkpoint/save/resume of model, optimizer, scheduler, RNG, and trainer position
- a thin adapter for `llm-training-data` column-batch collators

Not in the core: Ray, serving, vLLM, Spark, retrieval, semantic IDs, constrained decoding, or MTP.

## Install

```bash
pip install -e .
```

For the SFT example, install the data package alongside this repository:

```bash
pip install -e ../llm-training-data
```

## Architecture

```text
raw examples
    |
    v
llm-training-data
(render/tokenize/mask/pad/pack)
    |
    v
llm-training
(model/freeze/optimizer/scheduler/train/checkpoint)
    |
    v
trained checkpoint
    |
    v
model-artifact-transfer   # optional publishing layer
```

## Minimal usage

```python
from llm_training import (
    FreezeConfig,
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
    apply_freeze_policy,
    load_causal_lm,
)
from llm_training_data import build_sft_collator

components = load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")
apply_freeze_policy(components.model, FreezeConfig())

data_collator = build_sft_collator(
    components.tokenizer,
    max_sequence_length=512,
)

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=train_dataset,
    collate_fn=adapt_column_batch_collator(data_collator),
    config=TrainingConfig(output_dir="./outputs"),
)
trainer.train()
```

## Checkpoint semantics

A checkpoint is a **training checkpoint**, not merely exported model weights. Accelerate owns the runtime state (model, optimizer, scheduler, mixed-precision scaler when present, and RNG state), while `llm-training` stores the logical trainer position and configuration next to it.

```text
checkpoint-00001000/
├── ... Accelerate runtime state ...
├── trainer_state.json
├── training_config.json
└── tokenizer/
```

`TrainerState` records the next epoch/batch position, so a resumed single-process run can continue at the next batch rather than restarting the epoch.
