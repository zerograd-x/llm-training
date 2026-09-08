# Training

`llm-training` has two supported entry points:

1. `Trainer` for direct control over distributed training.
2. `run_training(...)` for a local, single-process orchestration path that prepares data splits, optionally expands vocabulary, trains, exports/merges artifacts, and can run fresh-process inference.

Use `Trainer` for multi-GPU jobs. Use `run_training(...)` when you want a compact end-to-end local pipeline.

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
