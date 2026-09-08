# llm-training

A small, opinionated training runtime for decoder-only causal language models.

The project is intentionally split from [`llm-training-data`](https://github.com/zerograd-x/llm-training-data):

- **llm-training-data** owns rendering, tokenization, label construction, padding, and packing.
- **llm-training** owns model loading, trainable-parameter policy, loss, optimization, checkpoint/resume, evaluation, and export.
- **model-artifact-transfer** can publish completed artifacts after training.

Dependency direction is one-way: `llm-training` may consume `llm-training-data`; `llm-training-data` must not depend on this package.

## Scope

The core runtime supports:

- Hugging Face decoder-only causal LMs
- PyTorch + Hugging Face Accelerate
- full fine-tuning and selective layer freezing
- LoRA and TrainableTokens
- custom causal-LM losses
- AdamW + Hugging Face schedulers
- gradient checkpointing
- DeepSpeed ZeRO-1/2/3
- deterministic epoch sampling
- resumable checkpoints
- scheduled validation
- standard Hugging Face model export

Serving, data systems, remote orchestration, retrieval, and application-specific model wrappers stay outside the core runtime.

## Install

```bash
pip install -e .
```

Optional features:

```bash
pip install -e ".[peft]"
pip install -e ".[deepspeed]"
```

For SFT data preparation and the high-level local training API:

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
(model/adaptation/loss/optimizer/train/checkpoint/eval/export)
    |
    v
checkpoint / adapter / exported model
```

## Direct training

Use `Trainer` when you want explicit control, especially for multi-GPU jobs.

```python
from llm_training import (
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
    load_causal_lm,
)
from llm_training_data import build_sft_collator

components = load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")
collator = build_sft_collator(components.tokenizer, max_sequence_length=512)

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=train_dataset,
    collate_fn=adapt_column_batch_collator(collator),
    config=TrainingConfig(output_dir="./outputs"),
)
trainer.train()
```

## High-level local training

Use `run_training(...)` for a single-process end-to-end local path with optional vocabulary expansion, adapter merge, and fresh-process inference.

```python
from llm_training import TrainingConfig, run_training

result = run_training(
    base_path="./base-model",
    train_data="./train.jsonl",
    validation_data="./validation.jsonl",
    config=TrainingConfig(output_dir="./run"),
)

print(result.model_path)
print(result.checkpoint_path)
```

See [`docs/training.md`](docs/training.md).

## DeepSpeed

```python
from llm_training import DeepSpeedConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    mixed_precision="bf16",
    deepspeed=DeepSpeedConfig(zero_stage=3),
)
```

ZeRO-3 partitions parameters after `accelerator.prepare(...)`. The current loading path does not partition model parameters during `from_pretrained()` itself.

See [`docs/deepspeed.md`](docs/deepspeed.md) and [`docs/distributed-runtime.md`](docs/distributed-runtime.md).

## Checkpoints vs exported models

A checkpoint is resumable training state: model/runtime state, optimizer, scheduler, RNG state, trainer position, configuration, and tokenizer state as required by the backend.

An exported model is an inference artifact. Do not treat checkpoint directories and exported model directories as interchangeable.

## Custom losses

The trainer owns causal next-token alignment. A loss implementation receives aligned `logits` and `labels`; labels equal to `-100` are ignored.

Inject a `BaseLoss` instance directly or configure one through `LossConfig`.

## Tests

Unit tests:

```bash
pytest -m "not integration and not gpu"
```

CPU integration tests:

```bash
pytest -m "integration and not gpu" tests/integration
```

GPU checks are opt-in and are not part of normal push/PR CI.

## Documentation

Long-lived documentation is intentionally small:

- [`training.md`](docs/training.md) — training entry points and artifact lifecycle
- [`deepspeed.md`](docs/deepspeed.md) — ZeRO and DeepSpeed ownership
- [`distributed-runtime.md`](docs/distributed-runtime.md) — distributed runtime contracts
- [`evaluation.md`](docs/evaluation.md) — validation behavior and metrics
- [`peft.md`](docs/peft.md) — LoRA and PEFT behavior
- [`trainable_tokens.md`](docs/trainable_tokens.md) — TrainableTokens behavior
- [`qwen35_aws_training_runbook.md`](docs/qwen35_aws_training_runbook.md) — Qwen3.5 AWS operations
