# Training

`llm-training` has two supported entry points:

1. `Trainer` for direct control over distributed training.
2. `run_training(...)` for a local, single-process orchestration path that prepares data splits, optionally expands vocabulary, trains, exports/merges artifacts, and can run fresh-process inference.

Use `Trainer` for multi-GPU jobs. Use `run_training(...)` when you want a compact end-to-end local pipeline.

## Inspect the effective plan before a run

A training configuration is not useful by itself unless its runtime consequences are clear. `llm-training` can resolve the config into an effective execution plan before model loading:

```bash
python -m llm_training.inspect training_config.json \
  --train-data train.jsonl \
  --world-size 4 \
  --model Qwen/Qwen3.5-4B \
  --max-sequence-length 256 \
  --has-eval-data
```

You can pass `--train-examples 50000` instead of a JSONL path. If the dataset size is omitted, model/backend/adaptation/output settings are still shown, but step-dependent fields remain unknown.

The plan reports:

- requested model, precision, backend, sequence-length ceiling, and gradient checkpointing;
- effective adaptation policy and, when a model has already been adapted, exact trainable/total parameter counts;
- per-device batch, world size, gradient accumulation, effective global batch, batches per rank, and optimizer steps;
- warmup in user-visible optimizer steps plus the internal scheduler-step horizon used by Accelerate;
- checkpoint cadence, expected distinct checkpoints, retention count, and evaluation points;
- output location and warnings for configurations whose runtime consequences are easy to miss.

Warnings currently include dominant warmup, final-checkpoint-only runs, a single validation point, no periodic loss line, an existing output directory without resume, ZeRO-3's full-model initial load, and gradient checkpointing on a small trainable fraction.

`run_training(...)` prints the effective plan after the model adaptation policy has been applied, so its log contains the exact trainable parameter count. The Qwen3.5 category SFT example does the same before the first optimizer step.

For programmatic inspection:

```python
from llm_training import build_training_plan, format_training_plan

plan = build_training_plan(
    config,
    train_examples=len(train_dataset),
    world_size=4,
    model_name="Qwen/Qwen3.5-4B",
    max_sequence_length=256,
)
print(format_training_plan(plan))
```

After constructing a `Trainer`, pass `trainer.adaptation_summary` to `build_training_plan(...)` to replace the requested adaptation description with the effective trainable parameter counts.

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

`llm-training-data` is responsible for rendering, tokenization, masking, padding, and packing. The core training package owns model adaptation, loss, optimizer/scheduler, checkpointing, evaluation, and export.

## High-level local training

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

Input rows require string `prompt` and `completion` fields. JSONL paths and in-memory lists are both accepted. Duplicate sample IDs and exact overlap across train/validation/test splits are rejected before model loading.

`run_training(...)` uses fresh subprocesses for lifecycle stages so training, adapter merge, and inference do not share stale distributed/model state. Stage logs are written under:

```text
RUN_DIR/stages/
  train.log
  train-request.json
  train-result.json
  merge.log          # when an adapter must be merged
  infer.log          # when inference=True
```

The final summary is written to `RUN_DIR/training-result.json` and returned as `TrainingResult`.

## Checkpoints and exported models

A checkpoint is resumable training state. It includes model/runtime state, optimizer, scheduler, RNG state, trainer position, configuration, and tokenizer files as required by the backend.

An exported model is an inference artifact. Full training exports a Hugging Face model. LoRA and TrainableTokens paths may first save an adapter and then merge it into a standalone model.

Do not treat a checkpoint directory and an exported model directory as interchangeable.

## Distributed training

For multi-GPU training, launch your script with `torchrun` and construct `Trainer` normally. `DistributedRuntime` and Accelerate/DeepSpeed own process coordination.

Example:

```bash
torchrun --standalone --nproc_per_node=4 train.py
```

ZeRO-3 partitions model state after `accelerator.prepare(...)`. Model loading before that point still requires each rank to load the checkpoint according to the current backend implementation.

See:

- `deepspeed.md` for ZeRO and optimizer/scheduler ownership.
- `distributed-runtime.md` for distributed runtime contracts.
- `evaluation.md` for evaluation semantics.
- `peft.md` and `trainable_tokens.md` for parameter-efficient training.
- `qwen35_aws_training_runbook.md` for the Qwen3.5 single-node AWS operational path.
