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

`use_reentrant` remains an explicit runtime setting. The CPU integration tests
do not establish a universal choice for multi-rank ZeRO-3. The older reference
implementation selected reentrant checkpointing for its ZeRO-3 environment;
real multi-GPU validation is required before this runtime hard-codes such a
backend-specific rule.

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
parameters. It does not request a consolidated full-model state dict. This code
path is designed for ZeRO-3, but real multi-rank ZeRO-3 TrainableTokens saving
is not yet claimed as validated.

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

## Deliberate differences from the reference implementation

The user-provided reference is the design source for using PEFT TrainableTokens,
appended-token selection, tied-weight handling, adapter-only worker saves, and
offline merge. This runtime intentionally differs in several places:

1. **TrainableTokens is not a FreezeConfig flag.** `FreezeConfig` in this repo
   means which base-model parameters remain fully trainable. Partial-token
   parameterization is a separate adaptation mechanism and therefore has its
   own `TrainableTokensConfig`.
2. **`LoRA + TrainableTokens` is rejected in v1.** The reference path could
   construct LoRA and then replace that PEFT model with a standalone
   TrainableTokens model, silently losing the LoRA adapter. Current PEFT has a
   native combined mechanism (`LoraConfig.trainable_token_indices`), but that is
   a separate feature and should be implemented and tested explicitly rather
   than obtained accidentally by sequential wrapping.
3. **`FreezeConfig + TrainableTokens` is rejected in v1.** Standalone
   TrainableTokens has a simple invariant: only selected token deltas are
   trainable. Combining arbitrary full-trained base parameters with it would
   require a hybrid artifact contract analogous to hybrid LoRA and should not
   be implicit.
4. **No silent zero-token no-op.** If `TrainableTokensConfig` is present,
   `n_tokens_added` must be greater than zero. A configuration typo fails before
   training instead of changing the training regime.
5. **Targets are fully qualified.** The reference helper returned embedding leaf
   names. This runtime uses exact resolved names to avoid suffix matches against
   unrelated embedding modules.
6. **Tied detection currently uses actual weight identity.** This is correct for
   the current lifecycle because the complete HF model is materialized before
   PEFT/Accelerate/DeepSpeed and `zero3_init_flag=False`. The reference also had
   a separate PEFT-style auto-tie detector based on model config metadata,
   which becomes important if this runtime later supports ZeRO-3 init-time
   partitioning where tensor identity may not yet be a reliable signal.
7. **TrainableTokens serialization explicitly disables full embedding saves.**
   The reference delegated directly to PEFT `save_pretrained`. This runtime
   additionally guards against PEFT's vocabulary-resize auto-save behavior by
   forcing `save_embedding_layers=False`.
8. **Storage is not part of the training API.** The reference carried
   `base_hdfs` / `ckp_hdfs`. This repo accepts ordinary model/adapter paths and
   keeps HDFS, S3, or other publishing concerns outside the training core.

## Current validation boundary

Validated:

```text
PEFT TrainableTokens CPU training            ✅
appended-row selection                       ✅
tied GPT-2 embedding / LM-head path          ✅
gradient checkpointing CPU path              ✅
optimizer ownership after PEFT wrapping      ✅
compact adapter serialization                ✅
offline merge into expanded base             ✅
old rows unchanged / new rows updated        ✅
merged logits ~= trained PEFT logits          ✅
```

Not established by these tests:

```text
multi-rank ZeRO-3                            ❌
NCCL                                         ❌
ZeRO-3 init-time partitioning                ❌
LoRA + TrainableTokens combined adapter      ❌
FreezeConfig + TrainableTokens hybrid state  ❌
```
