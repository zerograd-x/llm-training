# DeepSpeed backend

`llm-training` treats DeepSpeed as a training backend rather than as a second source of optimizer and scheduler configuration.

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

The current backend surface supports ZeRO stages 1, 2, and 3. The same `OptimizerConfig` and `SchedulerConfig` are used whether DeepSpeed is disabled or enabled; DeepSpeed wraps and partitions the resulting runtime state during `accelerator.prepare(...)`.

This keeps optimizer and scheduler semantics backend-independent:

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

## Optimizer ownership

The current default is **client-owned optimization**:

```text
llm-training creates AdamW + scheduler
            -> DeepSpeed wraps them
            -> ZeRO partitions runtime state
```

DeepSpeed-owned optimizers and schedulers are intentionally reserved for an explicit future mode. They are useful for DeepSpeed-native choices such as CPUAdam, Muon, or other backend-specific optimizers, but enabling DeepSpeed should not silently replace the optimizer configured by `OptimizerConfig`.

The planned ownership model is:

```text
optimization backend = "torch"       # default
    -> OptimizerConfig / SchedulerConfig
    -> real torch optimizer + scheduler

optimization backend = "deepspeed"   # explicit opt-in
    -> DeepSpeed optimizer config
    -> DummyOptim placeholder
    -> DeepSpeed creates the real optimizer
    -> optional DeepSpeed scheduler config
    -> DummyScheduler placeholder when needed
```

The two modes will be mutually exclusive so learning rate, betas, weight decay, warmup, and scheduler ownership never have two competing sources of truth.
