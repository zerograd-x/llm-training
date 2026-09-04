# DeepSpeed backend

`llm-training` treats DeepSpeed as a training backend with explicit ownership for optimization components.

## Install

```bash
pip install -e ".[deepspeed]"
```

## ZeRO stages

```python
from llm_training import DeepSpeedConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    mixed_precision="bf16",
    deepspeed=DeepSpeedConfig(zero_stage=3),
)
```

The current backend surface supports ZeRO stages 1, 2, and 3.

By default, enabling DeepSpeed does **not** replace the optimizer or scheduler. The normal `OptimizerConfig` and `SchedulerConfig` still create the real PyTorch/Hugging Face objects, and DeepSpeed wraps/partitions their runtime state during `accelerator.prepare(...)`.

```text
OptimizerConfig + SchedulerConfig
             |
             v
      torch optimizer/scheduler
             |
             v
      accelerator.prepare(...)
             |
       +-----+------+
       |            |
   standard      DeepSpeed
                 ZeRO 1/2/3
```

## Explicit DeepSpeed-owned optimizer and scheduler

DeepSpeed ownership is opt-in. Adding an optimizer and scheduler under `DeepSpeedConfig` switches those components to DeepSpeed-managed implementations:

```python
from llm_training import (
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

config = TrainingConfig(
    output_dir="./outputs",
    optimizer=OptimizerConfig(
        learning_rate=2e-5,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
    ),
    scheduler=SchedulerConfig(
        warmup_ratio=0.03,
    ),
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

The runtime then uses:

```text
DeepSpeed optimizer config
        -> DummyOptim
        -> accelerator.prepare(...)
        -> DeepSpeed creates the real optimizer

DeepSpeed scheduler config
        -> DummyScheduler
        -> accelerator.prepare(...)
        -> DeepSpeed creates the real scheduler
```

Common hyperparameters keep one source of truth. Learning rate, weight decay, Adam betas, and epsilon come from `OptimizerConfig`; warmup length and total training steps come from `SchedulerConfig` plus the trainer's computed training length. DeepSpeed-specific `params` may add backend-specific options but may not override those shared fields.

For data-parallel runs with the current Accelerate default `split_batches=False`, each rank receives `ceil(unsharded_batches / world_size)` batches. The trainer first computes synchronized optimizer updates from that per-rank length and gradient accumulation, then expresses the scheduler horizon in the underlying scheduler-step units expected by Accelerate. `AcceleratedScheduler` advances a Torch/HF scheduler `world_size` times per synchronized optimizer update in this mode, so the horizon is `optimizer_updates * world_size`. This also handles uneven final batch groups correctly.

DeepSpeed `WarmupLR` and `WarmupDecayLR` require at least one warmup step. The runtime rejects a resolved warmup of zero before entering DeepSpeed rather than silently changing the requested schedule.

The supported ownership combinations are:

| Optimizer | Scheduler | Status |
| --- | --- | --- |
| Torch | Torch/HF | supported; default |
| Torch | DeepSpeed | supported |
| DeepSpeed | DeepSpeed | supported |
| DeepSpeed | Torch/HF | not yet supported |

The last combination requires deferred construction of a Torch/HF scheduler after DeepSpeed creates the real optimizer. It can be added later through `DummyScheduler(..., lr_scheduler_callable=...)` without changing the public configuration model.

## Offload

Optimizer offload is available with ZeRO stage 2 or 3. Parameter offload is available only with ZeRO stage 3.

```python
config = TrainingConfig(
    output_dir="./outputs",
    deepspeed=DeepSpeedConfig(
        zero_stage=3,
        offload_optimizer_device="cpu",
        offload_param_device="cpu",
    ),
)
```

NVMe offload requires the corresponding path to be configured explicitly.

## Gradient checkpointing

Gradient checkpointing is configured independently from DeepSpeed:

```python
from llm_training import GradientCheckpointingConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    gradient_checkpointing=GradientCheckpointingConfig(
        enabled=True,
        use_reentrant=False,
    ),
)
```

When enabled, the runtime disables `model.config.use_cache` when that field exists and calls the model's native `gradient_checkpointing_enable()` before backend preparation.

`use_reentrant=None` leaves the choice to the model implementation. Setting it explicitly requires a model implementation that accepts `gradient_checkpointing_kwargs`.

## Distributed lifecycle before model loading

For distributed jobs, construct `DistributedRuntime` before loading the model:

```python
from llm_training import DistributedRuntime

runtime = DistributedRuntime(config)
components = runtime.load_causal_lm(model_name_or_path)

trainer = Trainer(
    ...,
    config=config,
    runtime=runtime,
)
```

The runtime owns one Accelerator/process-group instance for the whole worker lifecycle. Local rank 0 is allowed to populate the node-local Hugging Face cache before sibling ranks enter `from_pretrained(...)`; all ranks still construct their own model object. `Trainer` reuses the same Accelerator rather than creating a second distributed runtime.

For generic S3/HDFS/shared-filesystem materialization, use `runtime.materialize_path(...)` with `scope="local"` for node-local disks or `scope="global"` for a filesystem visible to every node. See [`distributed-runtime.md`](distributed-runtime.md).

## ZeRO-3 model loading limitation

The earlier lifecycle problem—rank identity not existing until after model loading—has been removed by `DistributedRuntime`: the backend/process group can now exist before `from_pretrained()`.

However, **ZeRO-3 initialization-time parameter partitioning is still not enabled**. `build_accelerator(...)` currently constructs the `DeepSpeedPlugin` with `zero3_init_flag=False`, so the lifecycle is:

```text
DistributedRuntime
    -> Accelerator + DeepSpeedPlugin
    -> local-main-first model/cache resolution
    -> every rank from_pretrained() materializes its model object
    -> Trainer(..., runtime=runtime)
    -> accelerator.prepare(...)
    -> ZeRO-3 partitions parameters
```

Therefore a large model may still need enough host/device memory to be materialized before ZeRO-3 takes ownership. Moving the backend earlier was a prerequisite for future init-time partitioning, but it does not by itself make `from_pretrained()` ZeRO-3-sharded.

Supporting true initialization-time partitioning requires a separate loading path that enters the appropriate DeepSpeed/Transformers initialization context during model construction and is validated on real multi-GPU hardware.

## Checkpoint and artifact ownership

Training-state checkpointing is a collective/backend operation. Under ZeRO, every participating rank may own distinct model/optimizer state, so `accelerator.save_state(...)` is intentionally called by all ranks. Small logical metadata (`trainer_state.json`, training config, tokenizer metadata) is written by global rank 0 after the collective save.

Adapter export follows the same ownership split: required adapter parameters participate in any necessary gather, but only global rank 0 serializes the final adapter files.

LoRA and TrainableTokens merge are **driver-only** operations. `merge_lora()` and `merge_trainable_tokens()` reject an active/environment-declared world size greater than one. The intended lifecycle is distributed training -> rank-0 adapter serialization -> worker exit -> single-process offline merge.

## Export

Distributed full-model export obtains the state dict through `accelerator.get_state_dict(model)` before calling Hugging Face `save_pretrained(...)`. This is important for sharded backends such as ZeRO-3, where the in-memory model parameters are not necessarily present as one complete copy on each rank.

## Validation levels

DeepSpeed support is validated at multiple levels rather than treating all tests as equivalent.

### CPU unit tests

Normal CI validates configuration translation, ownership rules, `DummyOptim` / `DummyScheduler` construction, offload validation, gradient-checkpointing configuration, and ZeRO-aware export contracts without starting a DeepSpeed engine.

### Two-process generic distributed integration

A CPU integration test launches two real processes with `torch.distributed.run` and validates rank/global-local ownership, local-rank-0 materialization, all-rank model construction, DataLoader sharding, reuse of one `DistributedRuntime` inside `Trainer`, distributed scheduler horizon, and the driver-only guard. This validates generic distributed lifecycle semantics without claiming ZeRO-3 parameter sharding.

### Real DeepSpeed CPU integration

A separate CPU CI job installs the real DeepSpeed package and starts an actual DeepSpeed runtime. The integration test uses a tiny locally constructed Hugging Face GPT-2 model and validates:

- `deepspeed.initialize()` through `Accelerator.prepare(...)`;
- a real `DeepSpeedEngine` rather than a mock;
- DeepSpeed-owned AdamW through `DummyOptim`;
- DeepSpeed-owned `WarmupDecayLR` through `DummyScheduler`;
- forward, external causal-LM loss, backward, optimizer step, and scheduler step;
- a step-1 DeepSpeed checkpoint followed by an intentional process interruption;
- resume from that checkpoint in a fresh Python process;
- completion of the next optimizer step;
- LoRA/TrainableTokens adapter lifecycle tests where configured;
- Hugging Face model export after resume for the full-fine-tune path.

This test currently exercises ZeRO stage 2 with world size 1. It verifies real DeepSpeed/Accelerate API integration and checkpoint lifecycle on CPU, but it does **not** prove cross-rank parameter partitioning.

### GPU validation still required

The following remain separate validation levels:

```text
1 GPU DeepSpeed smoke
    -> CUDA + bf16 + DeepSpeed runtime behavior

2+ GPU ZeRO-3 integration
    -> real cross-rank parameter partitioning
    -> all-gather / reduce-scatter communication
    -> distributed checkpoint/resume
    -> adapter gather/save under partitioning
    -> consolidated Hugging Face export
```

A single CPU rank or single GPU rank can enter ZeRO code paths, but cannot validate the defining multi-rank partitioning behavior of ZeRO-3. Multi-GPU validation therefore remains a release/milestone test rather than a claim implied by the CPU integration suite.
