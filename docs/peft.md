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

This keeps the selected base transformer blocks fully trainable in addition to the LoRA adapters, while explicitly controlling token-embedding and LM-head trainability.

`FreezeConfig` is a structural policy rather than an implicit global `requires_grad=False` pass. Parameters outside the transformer-block, input-embedding, and LM-head structures it manages keep their prior trainability unless another configured rule changes them. For example, an architecture-specific positional embedding or final normalization parameter may remain trainable. Artifact export records the actual base parameters selected by the resulting policy rather than assuming that only the last N block parameters changed.

`get_peft_model(...)` normally freezes base parameters. The runtime records the base parameters selected by `FreezeConfig` before PEFT wrapping and restores those exact parameter objects afterward. Parameter names may gain PEFT wrapper prefixes or internal names such as `base_layer`, so restoration and artifact identity are connected by parameter identity rather than string rewriting.

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
- pure and hybrid LoRA artifact export
- pure and hybrid offline merge equivalence
- real DeepSpeed ZeRO-2 initialization with LoRA
- DeepSpeed-owned optimizer and scheduler
- checkpoint at step 1
- fresh-process resume
- continued training after resume
- adapter save and offline merge after the DeepSpeed worker exits

These tests do **not** validate multi-rank ZeRO-3 parameter partitioning or NCCL behavior.

## Deployment artifact lifecycle

Training-state checkpoints, PEFT adapter artifacts, hybrid base overrides, and merged deployment models are separate artifacts.

### Pure LoRA

For pure LoRA (`freeze=None`, `lora.bias="none"`), save the small PEFT artifact from the trainer:

```python
adapter_dir = trainer.save_adapter()
```

The artifact contains the standard PEFT adapter files plus generic training metadata and the tokenizer when one was supplied to `Trainer`:

```text
adapter/
    adapter_config.json
    adapter_model.safetensors
    training_config.json
    tokenizer files ...        # when available
```

No base-model weight is serialized in this case.

### Hybrid LoRA

If `FreezeConfig` leaves base parameters fully trainable, the same `save_adapter()` API produces a hybrid artifact:

```text
adapter/
    adapter_config.json
    adapter_model.safetensors
    base_trainable_state.safetensors
    training_config.json
    tokenizer files ...        # when available
```

`base_trainable_state.safetensors` contains **only** the fully trained base parameters, not the full base model. Its keys are the original Hugging Face parameter names recorded before PEFT wrapping. This avoids leaking PEFT implementation names such as `base_model.model...` or `.base_layer...` into the deployment artifact contract.

`lora.bias` modes other than `"none"` are still rejected by artifact export because PEFT bias-training serialization semantics have not yet been validated in this runtime.

### ZeRO-aware save boundary

`save_adapter()` does not consolidate the full base model. If the configured DeepSpeed stage is ZeRO-3, the implementation enters `deepspeed.zero.GatheredParameters` for only:

```text
LoRA parameters
+
fully trained base overrides, if hybrid
```

before serialization. This is designed to avoid a whole-model ZeRO-3 gather. Real multi-rank ZeRO-3 adapter serialization remains a Phase-5 validation item; current real DeepSpeed CPU integration is ZeRO-2, world size 1.

### Offline merge

Use `merge_lora(...)` outside the distributed training workers to create a plain Hugging Face model:

```python
from llm_training import merge_lora

merged_dir = merge_lora(
    base_model_name_or_path="./exact-base-model",
    adapter_path="./outputs/adapter",
    output_dir="./outputs/merged-model",
)
```

Pure LoRA follows:

```text
exact base HF checkpoint
        +
PEFT adapter
        ↓
PeftModel.from_pretrained(...)
        ↓
merge_and_unload()
        ↓
plain HF model
```

Hybrid LoRA follows:

```text
exact base HF checkpoint
        ↓
restore base_trainable_state.safetensors
        ↓
attach PEFT adapter
        ↓
merge_and_unload()
        ↓
plain HF model
```

The order matters. If a LoRA-targeted base weight was also fully trained, the merge must fold the LoRA delta into the **trained base value**, not back into the original base value.

The exact base matters: the merge must use the same base model shape and vocabulary that the adapter was trained against. `merge_lora()` validates hybrid override names and tensor shapes before applying them. This becomes especially important once vocabulary extension / TrainableTokens are supported.

If the adapter artifact contains tokenizer files, `merge_lora()` copies their effective tokenizer configuration into the merged model. Otherwise it uses the base tokenizer when the supplied base location exposes one. Generic `training_config.json` metadata is also carried over. Storage-specific sidecars and upload logic remain outside `llm-training`.

Integration tests compare logits from the trained PEFT model against logits from the offline merged plain HF model, rather than checking only that files were created. Both pure and hybrid paths are also exercised after a real DeepSpeed CPU checkpoint and fresh-process resume.

`Trainer.export_model()` remains reserved for ordinary full fine-tuning. With LoRA enabled it directs callers to `save_adapter()` followed by offline `merge_lora(...)` instead of guessing which artifact type the caller intended.
