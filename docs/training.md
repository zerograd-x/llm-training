# Training

`llm-training` has two supported entry points:

1. `Trainer` for direct control over distributed training.
2. `run_training(...)` for a local, single-process orchestration path that prepares data splits, optionally expands vocabulary, trains, exports/merges artifacts, and can run fresh-process inference.

Use `Trainer` for multi-GPU jobs. Use `run_training(...)` when you want a compact end-to-end local pipeline.

## Effective plan before a run

A config is not useful unless its runtime consequences are clear. Inspect it before model loading:

```bash
python -m llm_training.inspect training_config.json \
  --train-data train.jsonl \
  --world-size 4 \
  --model Qwen/Qwen3.5-4B \
  --max-sequence-length 256 \
  --run-id qwen35-category-20260908 \
  --artifact-root-uri s3://BUCKET/training \
  --has-eval-data
```

You can pass `--train-examples 50000` instead of a JSONL path. If dataset size is omitted, step-dependent values remain unknown but model, optimizer, backend, adaptation, and output settings are still shown.

The effective plan reports:

- run id, local output directory, and optional durable artifact URI;
- model, precision, backend, sequence-length ceiling, and gradient checkpointing;
- requested/effective adaptation policy and exact trainable parameter counts after model adaptation;
- per-device batch, world size, gradient accumulation, global batch, and optimizer-step count;
- optimizer owner/type plus LR, betas, epsilon, and weight decay;
- scheduler owner/type, warmup, and the backend's internal scheduler horizon;
- DeepSpeed ZeRO stage, optimizer/parameter offload, gradient clipping ownership, and the generated DeepSpeed config;
- checkpoint cadence/retention and scheduled evaluation points;
- warnings for easy-to-miss runtime consequences.

All user-facing `*_steps` values mean **synchronized optimizer updates**. This includes:

- `scheduler.warmup_steps`;
- `checkpoint.save_every_steps`;
- `evaluation.every_steps`;
- `log_every_steps`.

Accelerate may advance an underlying scheduler more than once per optimizer update in distributed mode. That backend detail is translated internally and does not change user-facing step semantics. For example, on four ranks, `warmup_steps=100` still means 100 optimizer updates even if the underlying scheduler receives 400 ticks.

Warnings include dominant warmup, final-checkpoint-only runs, a single validation point, no periodic loss line, output collisions, ZeRO-3's full-model initial load, offload that cannot help initial loading, and gradient checkpointing on a small trainable fraction.

## Run identity and persisted metadata

Every supported high-level run can have one durable identity:

```python
result = run_training(
    ...,
    config=TrainingConfig(output_dir="./runs/category-sft"),
    run_id="category-sft-20260908",
    artifact_root_uri="s3://my-bucket/llm-training",
)
```

This resolves to:

```text
run_id       = category-sft-20260908
local output = ./runs/category-sft
artifact URI = s3://my-bucket/llm-training/category-sft-20260908/
```

The run directory contains:

```text
RUN_DIR/
  run.json
  training-plan.json
  training-result.json
  checkpoints...
  artifacts...
  stages/...
```

`run.json` is the stable identity record. A non-empty output directory without `run.json` cannot be silently claimed by a new run. If `run.json` already exists, its identity must match.

`training-plan.json` persists the effective execution plan, including optimizer/backend ownership and resolved step counts, so a completed checkpoint can be understood without reconstructing the original command line.

For programmatic inspection:

```python
from llm_training import (
    build_training_plan,
    format_training_plan,
    resolve_run_identity,
    save_training_plan,
)

identity = resolve_run_identity(
    config.output_dir,
    run_id="category-sft-20260908",
    artifact_root_uri="s3://my-bucket/llm-training",
)
plan = build_training_plan(
    config,
    train_examples=len(train_dataset),
    world_size=4,
    model_name="Qwen/Qwen3.5-4B",
    max_sequence_length=256,
    run_identity=identity,
)
print(format_training_plan(plan))
save_training_plan(plan)
```

After constructing a `Trainer`, pass `trainer.adaptation_summary` to replace the requested adaptation description with exact trainable/total parameter counts.

## Minimal direct training

```python
from llm_training import Trainer, TrainingConfig, load_causal_lm

components = load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=train_dataset,
    collate_fn=collate_fn,
    config=TrainingConfig(output_dir="./outputs"),
)
trainer.train()
```

`llm-training-data` owns rendering, tokenization, masking, padding, and packing. The core training package owns model adaptation, loss, optimizer/scheduler, checkpointing, evaluation, and export.

## High-level local training

```python
from llm_training import TrainingConfig, run_training

result = run_training(
    base_path="./base-model",
    train_data="./train.jsonl",
    validation_data="./validation.jsonl",
    config=TrainingConfig(output_dir="./run"),
    run_id="local-smoke",
)

print(result.run_id)
print(result.model_path)
print(result.checkpoint_path)
```

Input rows require string `prompt` and `completion` fields. JSONL paths and in-memory lists are accepted. Duplicate sample IDs and exact overlap across train/validation/test splits are rejected before model loading.

`run_training(...)` uses fresh subprocesses for lifecycle stages so training, adapter merge, and inference do not share stale distributed/model state. Stage logs are written under `RUN_DIR/stages/`.

## Checkpoints and exported models

A checkpoint is resumable training state: model/runtime state, optimizer, scheduler, RNG state, trainer position, configuration, and tokenizer files as required by the backend.

An exported model is an inference artifact. Full training exports a Hugging Face model. LoRA and TrainableTokens paths may first save an adapter and then merge it into a standalone model.

Do not treat a checkpoint directory and an exported model directory as interchangeable.

## Distributed training

For multi-GPU training, launch with `torchrun` and construct `Trainer` normally:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

ZeRO-3 partitions model state after `accelerator.prepare(...)`. The current model-loading path still materializes a full checkpoint per rank before that point.

See:

- `deepspeed.md` for ZeRO and optimizer/scheduler ownership;
- `distributed-runtime.md` for distributed runtime contracts;
- `evaluation.md` for evaluation semantics;
- `peft.md` and `trainable_tokens.md` for parameter-efficient training;
- `qwen35_aws_training_runbook.md` for the Qwen3.5 single-node AWS operational path.
