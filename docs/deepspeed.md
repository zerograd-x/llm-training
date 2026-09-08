# DeepSpeed backend

`llm-training` treats DeepSpeed as a backend, not as a second source of training hyperparameters.

## Optimizer and scheduler ownership

The common optimizer settings always live in `OptimizerConfig`:

```python
OptimizerConfig(
    learning_rate=2e-5,
    weight_decay=0.1,
    beta1=0.9,
    beta2=0.95,
    eps=1e-8,
)
```

With `DeepSpeedConfig(zero_stage=...)` alone, `llm-training` constructs the Torch optimizer and Hugging Face scheduler, then DeepSpeed prepares/wraps them during `accelerator.prepare(...)`.

To let DeepSpeed create the optimizer and scheduler, opt in explicitly:

```python
config = TrainingConfig(
    output_dir="./outputs",
    optimizer=OptimizerConfig(
        learning_rate=2e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    ),
    scheduler=SchedulerConfig(warmup_ratio=0.03),
    deepspeed=DeepSpeedConfig(
        zero_stage=3,
        optimizer=DeepSpeedOptimizerConfig(
            type="AdamW",
            params={"torch_adam": True},
        ),
        scheduler=DeepSpeedSchedulerConfig(
            type="WarmupDecayLR",
            params={"warmup_type": "linear"},
        ),
    ),
)
```

`DeepSpeedOptimizerConfig.params` may contain backend-specific options, but it cannot override shared `lr`, `weight_decay`, `betas`, or `eps`. Likewise, DeepSpeed scheduler params cannot override the shared warmup or total-step values. This keeps one source of truth.

Supported ownership combinations:

| Optimizer | Scheduler | Status |
| --- | --- | --- |
| Torch-configured | Torch/HF-configured | supported; default |
| Torch-configured | DeepSpeed-owned | supported |
| DeepSpeed-owned | DeepSpeed-owned | supported |
| DeepSpeed-owned | Torch/HF-configured | not yet supported |

The effective training plan reports both configuration ownership and backend preparation, for example `torch-configured / deepspeed-prepared`.

## Step semantics

Every public `*_steps` setting means **synchronized optimizer updates**:

- `scheduler.warmup_steps`
- `checkpoint.save_every_steps`
- `evaluation.every_steps`
- `log_every_steps`

This meaning does not change with world size.

Accelerate's underlying scheduler may advance more than once for one optimizer update when `split_batches=False`. `llm-training` translates optimizer-step counts to those internal scheduler ticks at the backend boundary. For example, with four ranks:

```text
warmup_steps = 100          # user meaning: 100 optimizer updates
internal scheduler warmup = 400 ticks
```

`warmup_ratio` is also resolved against the optimizer-update horizon before backend scaling.

## Batch size

`TrainingConfig` is the only source for batch accounting:

```python
per_device_batch_size=8
gradient_accumulation_steps=1
```

The effective global batch is:

```text
per-device batch × gradient accumulation × world size
```

DeepSpeed receives:

```json
{
  "train_micro_batch_size_per_gpu": "auto",
  "train_batch_size": "auto"
}
```

so users do not have to repeat the same micro-batch size in two configs.

## ZeRO and offload

```python
DeepSpeedConfig(
    zero_stage=3,
    offload_optimizer_device="none",
    offload_param_device="none",
    zero3_save_16bit_model=True,
)
```

Supported rules:

- ZeRO-1/2/3 are supported.
- optimizer offload requires ZeRO-2 or ZeRO-3.
- parameter offload requires ZeRO-3.
- NVMe offload requires an explicit NVMe path.

The generated DeepSpeed config is included in `training-plan.json`, so the exact effective backend settings remain available after the run.

## Gradient clipping

`TrainingConfig.max_grad_norm` remains the single user-facing clipping setting.

Without DeepSpeed, `Trainer` performs clipping through Accelerate before the optimizer update.

With DeepSpeed, the value is translated to DeepSpeed's `gradient_clipping`, and the DeepSpeed engine owns clipping as part of its engine step. `Trainer` does not issue a second manual clip on the DeepSpeed path.

```text
non-DeepSpeed:
backward -> accelerator.clip_grad_norm_ -> optimizer.step

DeepSpeed:
backward -> DeepSpeed engine step
              ├─ gradient clipping
              ├─ optimizer update
              └─ scheduler update
```

The effective plan reports the clipping owner explicitly.

## Gradient checkpointing

Gradient checkpointing is independent of DeepSpeed:

```python
GradientCheckpointingConfig(
    enabled=True,
    use_reentrant=False,
)
```

When enabled, the runtime disables `model.config.use_cache` when available and calls the model's native `gradient_checkpointing_enable()` before backend preparation.

For a run that trains only a small fraction of the model, the effective plan warns that gradient checkpointing may cost more recomputation than it saves; this is a performance warning, not a correctness error.

## ZeRO-3 model-loading limitation

`DistributedRuntime` exists before model loading, but true ZeRO-3 initialization-time partitioning is still disabled. The current lifecycle is:

```text
DistributedRuntime
    -> Accelerator + DeepSpeedPlugin
    -> each rank loads a complete model
    -> Trainer applies freeze/PEFT policy
    -> accelerator.prepare(...)
    -> ZeRO-3 partitions parameters/state
```

Therefore ZeRO-3 reduces persistent training-state memory but does not yet remove the full-model-per-rank materialization requirement during `from_pretrained()`.

Parameter offload also starts after DeepSpeed preparation, so it does not solve this initial-load requirement. Both limitations are surfaced in the effective training plan.

## Checkpoint and export ownership

Checkpointing is collective under DeepSpeed because ranks may own different parameter, gradient, and optimizer shards. All ranks participate in `accelerator.save_state(...)`; rank 0 writes small logical metadata after the collective save.

Distributed full-model export uses `accelerator.get_state_dict(model)` before Hugging Face `save_pretrained(...)`, so ZeRO-3 shards are gathered through the backend rather than assuming every rank already has a full state dict.

LoRA and TrainableTokens merges are single-process operations after distributed training exits.

## Inspect the exact settings

Before a run:

```bash
python -m llm_training.inspect training_config.json \
  --train-examples 50000 \
  --world-size 4 \
  --model Qwen/Qwen3.5-4B
```

The plan shows optimizer/scheduler ownership, LR/betas/eps/weight decay, ZeRO stage, offload, clipping owner, batch math, warmup, and the effective generated DeepSpeed config. During supported high-level runs, the same plan is persisted as `training-plan.json`.
