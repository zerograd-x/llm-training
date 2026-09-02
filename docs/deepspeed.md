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

## ZeRO-3 model loading limitation

ZeRO-3 parameter partitioning is active after `accelerator.prepare(...)`, so training-time model parameters are sharded across data-parallel ranks.

The current public API still loads a model before constructing `Trainer`:

```text
load_causal_lm()
    -> from_pretrained() creates the model
    -> Trainer(...)
    -> Accelerator + DeepSpeedPlugin
    -> accelerator.prepare(...)
    -> ZeRO-3 parameter partitioning
```

Therefore the runtime does **not yet claim ZeRO-3 initialization-time partitioning** (`zero3_init_flag=True`). A large model may still need enough memory to be materialized during `from_pretrained()` before ZeRO-3 takes ownership.

Supporting initialization-time partitioning requires moving backend context into model construction rather than merely changing a flag after the model has already been loaded.

## Export

Distributed export obtains the state dict through `accelerator.get_state_dict(model)` before calling Hugging Face `save_pretrained(...)`. This is important for sharded backends such as ZeRO-3, where the in-memory model parameters are not necessarily present as one complete copy on each rank.

## Validation levels

DeepSpeed support is validated at multiple levels rather than treating all tests as equivalent.

### CPU unit tests

Normal CI validates configuration translation, ownership rules, `DummyOptim` / `DummyScheduler` construction, offload validation, gradient-checkpointing configuration, and ZeRO-aware export contracts without starting a DeepSpeed engine.

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
- Hugging Face model export after resume.

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
    -> consolidated Hugging Face export
```

A single CPU rank or single GPU rank can enter ZeRO code paths, but cannot validate the defining multi-rank partitioning behavior of ZeRO-3. Multi-GPU validation therefore remains a release/milestone test rather than a claim implied by the CPU integration suite.
