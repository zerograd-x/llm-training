# llm-training

A small, opinionated training runtime for decoder-only causal language models.

The project is intentionally split from [`llm-training-data`](https://github.com/zerograd-x/llm-training-data):

- **llm-training-data** owns rendering, tokenization, label construction, padding, and packing.
- **llm-training** owns model loading, trainable-parameter policy, loss, optimization, checkpoint/resume, evaluation, run metadata, and export.
- **model-artifact-transfer** can publish completed artifacts after training.

Dependency direction is one-way: `llm-training` may consume `llm-training-data`; `llm-training-data` must not depend on this package.

To run SFT, causal-LM, or DPO jobs directly from YAML/JSON with the installed
wheel, see [Config-driven training](docs/config-driven.md) and
[Config-driven DPO](docs/dpo.md).

## Scope

The core runtime supports:

- Hugging Face decoder-only causal LMs
- PyTorch + Hugging Face Accelerate
- full fine-tuning and selective layer freezing
- LoRA and TrainableTokens
- custom causal-LM losses
- TRL-backed Direct Preference Optimization (DPO)
- AdamW + Hugging Face schedulers
- gradient checkpointing
- DeepSpeed ZeRO-1/2/3
- deterministic epoch sampling
- resumable checkpoints
- scheduled validation
- effective training-plan inspection
- stable run identity and persisted run metadata
- standard Hugging Face model export

Serving, data systems, remote orchestration, retrieval, and application-specific model wrappers stay outside the core runtime.

The design goal is not to expose every backend knob. It is to make training configuration hard to misinterpret: shared hyperparameters have one source of truth, user-facing `*_steps` values always mean synchronized optimizer updates, and the runtime records the effective plan that actually executes.

## Install

```bash
pip install -e .
```

Optional features:

```bash
pip install -e ".[peft]"
pip install -e ".[deepspeed]"
pip install -e ".[dpo,peft]"
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
    +--> run.json
    +--> training-plan.json
    +--> checkpoints / adapter / exported model
```

## Inspect before training

A config should explain what it will actually do before a large job starts.

```bash
python -m llm_training.inspect training_config.json \
  --train-examples 50000 \
  --world-size 4 \
  --model Qwen/Qwen3.5-4B \
  --max-sequence-length 256
```

The effective plan reports:

- trainable-parameter policy and, after model adaptation, exact trainable parameter counts
- per-device batch, gradient accumulation, global batch, and optimizer-step count
- explicit epoch- or optimizer-step-based training horizon
- optimizer and scheduler ownership plus LR, betas, epsilon, weight decay, and warmup
- DeepSpeed ZeRO/offload/clipping settings and the generated effective DeepSpeed config
- checkpoint/evaluation cadence and retention
- run identity, output path, durable artifact URI, and configuration warnings

Supported training paths persist the same information to `training-plan.json`.

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

For step-based training, set `max_steps`. It is the authoritative stopping
horizon and may span multiple passes over a finite map-style dataset:

```python
config = TrainingConfig(
    output_dir="./outputs",
    max_steps=100_000,
)
```

When `max_steps` is set, `num_epochs` no longer limits stopping. Epochs remain
the unit for deterministic data reshuffling, resume position, and optional
epoch-based evaluation. Warmup, logging, evaluation, and checkpoint step settings
continue to mean synchronized optimizer updates.

## High-level local training

Use `run_training(...)` for a single-process end-to-end local path with optional vocabulary expansion, adapter merge, and fresh-process inference.

```python
from llm_training import TrainingConfig, run_training

result = run_training(
    base_path="./base-model",
    train_data="./train.jsonl",
    validation_data="./validation.jsonl",
    config=TrainingConfig(output_dir="./run"),
    run_id="experiment-001",
    artifact_root_uri="s3://my-bucket/training",
)

print(result.run_id)
print(result.model_path)
print(result.checkpoint_path)
print(result.artifact_uri)
```

A run owns its output directory. `run.json` records that identity, and durable artifacts use `artifact_root_uri/run_id/` instead of ad-hoc path construction.

See [`docs/training.md`](docs/training.md).

## Optimizer and DeepSpeed ownership

`OptimizerConfig` is the single source of truth for common optimizer hyperparameters:

```python
OptimizerConfig(
    learning_rate=2e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
)
```

With `DeepSpeedConfig(zero_stage=3)` alone, the optimizer and scheduler are Torch/Hugging Face configured and DeepSpeed prepares/wraps their runtime state. DeepSpeed-owned optimizer/scheduler implementations are explicit opt-ins; they cannot silently override shared optimizer or scheduler settings.

Batch size also has one source of truth in `TrainingConfig`. DeepSpeed receives automatic batch values rather than requiring a second manually synchronized micro-batch setting.

```python
from llm_training import DeepSpeedConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    mixed_precision="bf16",
    deepspeed=DeepSpeedConfig(zero_stage=3),
)
```

All public step settings—warmup, logging, evaluation, and checkpoint cadence—mean synchronized optimizer updates regardless of world size. Backend-specific scheduler ticks are translated internally.

ZeRO-3 currently partitions parameters after `accelerator.prepare(...)`; the loading path still materializes a complete model on each rank during `from_pretrained()`.

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

- [`training.md`](docs/training.md) — training entry points, effective plans, run identity, and artifact lifecycle
- [`deepspeed.md`](docs/deepspeed.md) — optimizer/scheduler ownership, step semantics, ZeRO, offload, and clipping
- [`dpo.md`](docs/dpo.md) — preference data, DPO parameters, reference policy, LoRA, and multi-GPU launch
- [`distributed-runtime.md`](docs/distributed-runtime.md) — distributed runtime contracts
- [`evaluation.md`](docs/evaluation.md) — validation behavior and metrics
- [`peft.md`](docs/peft.md) — LoRA and PEFT behavior
- [`trainable_tokens.md`](docs/trainable_tokens.md) — TrainableTokens behavior
- [`qwen35_aws_training_runbook.md`](docs/qwen35_aws_training_runbook.md) — Qwen3.5 AWS operations
