# TrainableTokens

`llm-training` supports partial vocabulary training through Hugging Face PEFT's
`TrainableTokensConfig`. The runtime does **not** make the full embedding matrix
trainable and does not implement manual gradient masking.

## Contract

The model passed to `Trainer` must already contain the vocabulary expansion.
For example, if a 32-token model was resized to 36 tokens and the four appended
rows should be trained:

```python
from llm_training import TrainableTokensConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    trainable_tokens=TrainableTokensConfig(
        n_tokens_added=4,
    ),
)
```

The runtime resolves the trainable token ids from the current input-embedding
shape:

```text
vocab_size = 36
n_tokens_added = 4

trainable token ids = [32, 33, 34, 35]
```

`n_tokens_added` must be positive and cannot exceed the resized vocabulary
size. Supplying a TrainableTokens config is therefore an explicit request for
partial-token training; there is no silent `n_tokens_added == 0` fallback.

The first implementation intentionally supports the common vocabulary-extension
case only: the trainable rows are the final N appended rows. PEFT itself can
train arbitrary token indices, so a future config can expose explicit token ids
without changing the training mechanism.

## Model adaptation

TrainableTokens target discovery happens on the ordinary Hugging Face model
before PEFT wrapping.

The input embedding is always targeted. For an untied model the output
embedding / LM head is targeted separately. For a tied model only the input
embedding is named explicitly; PEFT discovers the tied output weight and
creates the tied TrainableTokens wrapper itself.

```text
untied:
    input embedding  -> TrainableTokens
    LM head          -> TrainableTokens

tied:
    input embedding  -> TrainableTokens
    LM head          -> PEFT tied adapter, not an explicit second target
```

The runtime resolves targets to fully-qualified module names. This is narrower
than using a leaf-name suffix such as `embed_tokens` and avoids accidentally
adapting multiple embedding modules in a multimodal or custom architecture.

Once targets and token ids have been resolved, adaptation is delegated to PEFT:

```text
resized base model
    ↓
resolve embedding / LM-head targets
    ↓
resolve final N token ids
    ↓
PEFT TrainableTokensConfig(...)
    ↓
get_peft_model(...)
    ↓
gradient-checkpointing setup
    ↓
optimizer construction
    ↓
accelerator.prepare(...)
```

Only PEFT `trainable_tokens_delta` parameters are expected to remain trainable
in standalone TrainableTokens mode. The optimizer is constructed only after
those parameters exist.

## Gradient checkpointing

TrainableTokens is treated as adapter training for gradient-checkpointing
purposes. When gradient checkpointing is enabled, the runtime calls
`enable_input_require_grads()` before enabling model-native checkpointing and
disables `model.config.use_cache` when available.

This is important for TrainableTokens because the trainable rows live at the
bottom of an otherwise frozen network; autograd still needs a path through the
frozen transformer blocks.

`use_reentrant` remains an explicit runtime setting. Non-reentrant GC passes
on the pinned two-node tiny-model stack in the [GPU validation matrix](gpu-validation.md).
Other model/backend combinations still need their own validation before a
universal checkpointing mode can be selected.

## Adapter artifact

TrainableTokens uses the same high-level artifact boundary as LoRA:

```text
training worker
    ↓
small PEFT adapter
    ↓
worker / DeepSpeed lifecycle ends
    ↓
offline merge
    ↓
plain Hugging Face checkpoint
```

Save the adapter with:

```python
adapter_dir = trainer.save_adapter()
```

The artifact contains PEFT TrainableTokens state plus generic training metadata
and an optional tokenizer:

```text
adapter/
    adapter_config.json
    adapter_model.safetensors
    training_config.json
    tokenizer files ...        # when supplied to Trainer
```

A vocabulary resize creates an important PEFT serialization edge case. PEFT's
`save_embedding_layers="auto"` can decide to serialize the full embedding
matrix when it detects that the vocabulary size changed. That would defeat the
purpose of a compact partial-token artifact. `llm-training` therefore uses
`save_embedding_layers=False` explicitly for standalone TrainableTokens and
serializes only the TrainableTokens adapter state.

Under configured ZeRO-3, adapter saving enters
`deepspeed.zero.GatheredParameters` only for the trainable token-delta
parameters. It does not request a consolidated full-model state dict. The
2026-09-04 two-node L4 run verified this gather scope, actual token-delta
partitioning and global-rank-0 serialization, followed by offline merge/reload.
See the [GPU validation matrix](gpu-validation.md) for source and fixture scope.

## Offline merge

Use the **same vocabulary-expanded base checkpoint that was used for training**:

```python
from llm_training import merge_trainable_tokens

merged_dir = merge_trainable_tokens(
    base_model_name_or_path="./expanded-base-model",
    adapter_path="./outputs/adapter",
    output_dir="./outputs/merged-model",
)
```

The merge is deliberately outside the distributed worker:

```text
exact expanded base
        +
TrainableTokens adapter
        ↓
PeftModel.from_pretrained(...)
        ↓
merge_and_unload()
        ↓
selected rows are written into the embedding / tied output weight
        ↓
plain HF model
```

`merge_trainable_tokens()` validates that every adapter token id exists in the
supplied base vocabulary before attaching the adapter. This catches the common
mistake of trying to merge a token adapter into the original pre-resize base.

The CPU integration test verifies more than file existence:

- only `trainable_tokens_delta` parameters are optimized;
- the saved adapter is smaller than the full embedding matrix and contains no
  full-vocabulary embedding tensor;
- after merge, all pre-existing token rows remain bit-identical to the expanded
  base;
- the appended rows change after training;
- logits from the merged plain HF model match the trained PEFT model within
  numerical tolerance.

## Adaptation and artifact contracts

1. **Partial-token training has its own configuration.** `FreezeConfig` controls
   fully trained base parameters; `TrainableTokensConfig` selects a different
   parameterization in which only the selected token deltas are trainable.
2. **Combined adaptation is explicit.** Version 1 rejects both
   `LoRA + TrainableTokens` and `FreezeConfig + TrainableTokens`. Future combined
   modes require defined optimizer coverage, checkpoint and artifact contracts;
   sequential adapter wrapping must not replace an existing adapter.
3. **Invalid token counts fail early.** `n_tokens_added` must be greater than zero
   when partial-token training is requested.
4. **Targets are fully qualified.** Exact resolved module names prevent suffix
   matches against unrelated embedding modules.
5. **Tied weights are identified before partitioning.** The current runtime loads
   full weights before adaptation and backend preparation. Initialization-time
   sharding would require revisiting this discovery contract using model metadata.
6. **Adapter serialization is compact.** `save_embedding_layers=False` prevents
   vocabulary-resize handling from saving the complete embedding matrix.
7. **Artifacts use ordinary paths.** Model and adapter APIs have no dependency on
   a storage service, organization platform or deployment environment.

## Current validation boundary

Validated:

```text
PEFT TrainableTokens CPU training            ✅
appended-row selection                       ✅
tied GPT-2 embedding / LM-head path          ✅
gradient checkpointing CPU path              ✅
single-L4 BF16 / ZeRO-3 lifecycle            ✅
two-node NCCL + ZeRO-3 lifecycle             ✅
non-reentrant GC + two-node ZeRO-3           ✅
optimizer ownership after PEFT wrapping      ✅
compact adapter serialization                ✅
offline merge into expanded base             ✅
old rows unchanged / new rows updated        ✅
merged logits ~= trained PEFT logits          ✅
```

GPU results use the pinned tiny-GPT-2 stack, two hosts with one L4 each, and
shared checkpoint storage. The P1 rerun verifies resume, compact adapter save,
merge/reload and unchanged old rows with and without non-reentrant checkpointing.
The [matrix](gpu-validation.md) records historical source provenance separately
from the maintained regression driver.

Not established by these tests:

```text
same-host GPUs with distinct local indices   ❌
other architectures / GC modes / long runs   ❌
ZeRO-3 init-time partitioning                ❌
LoRA + TrainableTokens combined adapter      ❌
FreezeConfig + TrainableTokens hybrid state  ❌
```

