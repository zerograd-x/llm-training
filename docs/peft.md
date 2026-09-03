# PEFT and LoRA

`llm-training` supports LoRA through the optional Hugging Face PEFT dependency.

## Install

```bash
pip install -e ".[peft]"
```

## Standard adapter-only LoRA

The minimal configuration is:

```python
from llm_training import LoRAConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    lora=LoRAConfig(
        rank=8,
        alpha=16,
        dropout=0.0,
    ),
)
```

When `lora` is enabled and `freeze=None`, PEFT freezes the base model and only LoRA parameters are trainable. This is the standard adapter-only path.

The trainer applies model adaptation before optimizer construction:

```text
base model
    ↓
optional FreezeConfig
    ↓
LoRA target discovery
    ↓
get_peft_model(...)
    ↓
restore explicitly full-train base parameters
    ↓
gradient checkpointing setup
    ↓
optimizer construction
    ↓
accelerator.prepare(...)
```

This ordering guarantees that newly created LoRA parameters are visible to the optimizer and to backends such as DeepSpeed.

## LoRA target discovery

`target_modules=None` uses a conservative allow-list of common attention and MLP projection names. Discovery is restricted to modules inside transformer blocks and produces fully-qualified module names, so embeddings, LM heads, and unrelated modules outside the decoder blocks are not accidentally adapted.

You can provide explicit module suffixes or exact names:

```python
config = TrainingConfig(
    output_dir="./outputs",
    lora=LoRAConfig(
        target_modules=(
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
        ),
    ),
)
```

If no requested target can be found, configuration fails before training rather than silently creating a model with no LoRA parameters.

To adapt only the last transformer blocks, use `target_last_n_layers`:

```python
LoRAConfig(target_last_n_layers=8)
```

`target_last_n_layers` belongs to LoRA placement. It is intentionally separate from `FreezeConfig.train_last_n_layers`.

## FreezeConfig and hybrid training

`FreezeConfig` controls which **base-model parameters remain fully trainable**. LoRA controls where **adapter parameters are inserted**. These are different decisions.

For example:

```python
from llm_training import FreezeConfig, LoRAConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    freeze=FreezeConfig(
        train_last_n_layers=2,
        train_embeddings=False,
        train_lm_head=False,
    ),
    lora=LoRAConfig(
        rank=8,
        alpha=16,
    ),
)
```

This means:

- LoRA adapters are inserted according to `LoRAConfig`.
- The last two base transformer blocks also remain fully trainable.
- Embedding and LM-head base parameters remain frozen.

`get_peft_model(...)` normally freezes base parameters. The runtime records the base parameters selected by `FreezeConfig` before PEFT wrapping and restores those exact parameter objects afterward. Parameter names may gain PEFT wrapper prefixes, so restoration is based on parameter identity rather than string rewriting.

If `freeze=None`, no base parameter is restored after PEFT freezes the model.

## Gradient checkpointing

With LoRA and gradient checkpointing enabled, the runtime calls `enable_input_require_grads()` before enabling model-native gradient checkpointing. This keeps the autograd path from frozen base activations to trainable adapter parameters.

The runtime also disables `model.config.use_cache` when the model exposes that setting.

```python
from llm_training import GradientCheckpointingConfig

config = TrainingConfig(
    output_dir="./outputs",
    lora=LoRAConfig(),
    gradient_checkpointing=GradientCheckpointingConfig(
        enabled=True,
        use_reentrant=False,
    ),
)
```

The `use_reentrant` choice remains explicit. Multi-GPU ZeRO-3 gradient-checkpointing behavior is not claimed as validated by the CPU tests.

## Optimizer ownership

When `config.lora` is enabled, `Trainer` must construct the optimizer after PEFT wrapping because LoRA parameters do not exist before that transformation. For that reason a pre-built optimizer cannot be injected together with `config.lora`.

Both the normal Torch-owned optimizer path and the DeepSpeed-owned optimizer path consume the final trainable parameter set after adaptation.

## Checkpoint and resume

Training-state checkpoints use the existing Accelerate/DeepSpeed checkpoint mechanism. A fresh process reconstructs the same LoRA structure from `TrainingConfig` before loading checkpoint state.

CPU integration covers:

- real PEFT LoRA forward/backward/update
- gradient checkpointing with a frozen base
- optimizer ownership of the final trainable parameter set
- base parameters remaining unchanged in adapter-only training
- explicit base full-training restoration through `FreezeConfig`
- real DeepSpeed ZeRO-2 initialization with LoRA
- DeepSpeed-owned optimizer and scheduler
- checkpoint at step 1
- fresh-process resume
- continued training after resume

These tests do **not** validate multi-rank ZeRO-3 parameter partitioning or NCCL behavior.

## Export boundary

Training-state checkpointing and deployment artifacts are intentionally separate concepts.

The current LoRA phase does not overload the existing full-model `Trainer.export_model()` path. Calling `export_model()` with `config.lora` enabled fails explicitly. Adapter-only PEFT export and optional merged full-model export belong to the adapter-artifact phase so their memory, ZeRO gathering, and portability contracts can be implemented and tested independently.
