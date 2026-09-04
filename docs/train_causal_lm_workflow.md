# Portable causal-LM training workflow

This is a design for composing reusable components with ordinary Python calls
and filesystem paths. The training APIs exist today; the complete workflow,
vocabulary-expansion utility and evaluation interfaces described here are
planned integration work, not an existing `main()` API.

No organization-specific platform, workflow transpiler, distributed data engine,
remote scheduler, storage service or experiment-tracking service is required.
GPU workers can be launched directly with PyTorch/Accelerate. Cross-node runs
need an explicitly shared checkpoint path; single-node runs can use local disk.

## Component boundaries

| Component | Responsibility |
|---|---|
| Data preparation | Read ordinary datasets; produce explicit train/validation/test splits |
| `llm-training-data` | Render examples, tokenize, construct labels/masks, pad and optionally pack |
| Optional vocabulary preparation | Produce a matching tokenizer and resized base model before training |
| `llm-training` | Load the model, adapt/freeze parameters, optimize, checkpoint/resume and export |
| Offline artifact processing | Merge adapters after workers exit; produce a loadable HF model directory |
| Inference and task evaluation | Consume the exported model and an explicitly selected dataset |

## Data flow

```text
raw examples -> explicit dataset splits -> tokenization/collation
                                               |
base model + tokenizer -> optional vocab expansion
                                               |
                                           training
                                               |
                   +---------------------------+--------------------+
                   |                                                |
              full-model export                              adapter export
                   |                                                |
                   |                                   offline merge with exact base
                   +---------------------------+--------------------+
                                               |
                                  model + matching tokenizer
                                               |
                                optional inference -> evaluation
```

## Stage contracts

### 1. Prepare data

Accept paths or dataset objects. Keep train, validation and test inputs explicit;
never infer a test set from a positional argument named `train_data`. Validate
required columns and split availability before launching GPU workers. When a
held-out split is required, verify the split policy and sample overlap.

The existing data-package adapter handles column-batch collators. Dataset
readers and workflow split validation are integration-layer responsibilities.

### 2. Optionally expand vocabulary

Accept an explicit list of new token strings and a documented initialization
policy. Produce the resized base model and matching tokenizer together. Record
the actual number and IDs of added tokens; do not infer additions merely from
the requested list length, since a token may already exist.

Training, merge and inference must use the same resulting token-to-ID mapping.
Current `TrainableTokensConfig(n_tokens_added=N)` expects the final N vocabulary
rows and requires expansion to happen before Trainer construction. A reusable
expansion utility and its manifest remain planned work.

### 3. Train

Use `DistributedRuntime` and `Trainer` with an explicit training configuration.
For runtime-first HF loading, use `runtime.load_causal_lm()` or enter
`runtime.model_loading_context()` around a custom loader.

Validate the adaptation mode before training. Full/base-parameter training,
LoRA, hybrid LoRA and standalone TrainableTokens follow their documented
contracts. Unsupported combinations fail early. A training checkpoint includes
runtime state for resume and is distinct from a downstream inference artifact.

### 4. Export and optionally merge

Full-parameter training exports a complete HF model. Adapter training exports
its adapter state; hybrid LoRA additionally preserves its trained base overrides.
All required ranks participate in checkpointing/gathering, with global rank 0
responsible for final adapter serialization.

After distributed workers exit, merge the adapter into the exact base used for
training. Export the matching tokenizer and preserve generation-relevant
metadata, including chat-template or prompt configuration when supplied.

Choose artifact handling from the explicit adaptation/artifact type. Avoid an
implicit priority order that silently chooses one of two incompatible modes.

### 5. Optionally run inference and evaluation

Use a fresh process to load the completed model and tokenizer. Pass the dataset
to score explicitly. A supplied test dataset remains separate from the
validation dataset used during training or checkpoint selection.

Keep two evaluation layers distinct:

- Training validation: loss/perplexity and optional metrics aggregated over all
  ranks. This is planned core runtime work.
- Post-training evaluation: generated predictions and task-specific metrics.
  This belongs to an independently usable evaluation component.

Define named outputs for predictions and metrics instead of reusing positional
train/validation tuple slots. Record parsing failures and coverage rather than
silently discarding invalid predictions. Specific retrieval or business-label
schemes are outside this generic workflow.

## Proposed artifact metadata

The workflow should record artifact type, model/tokenizer paths, base revision
or fingerprint, vocabulary identity, adaptation mode, training configuration,
and optional generation configuration. Checkpoints, adapters and complete models
must be distinguishable. This unified manifest is proposed; existing APIs do
not yet expose all of these fields as one result type.

## Validation plan

1. Validate workflow inputs and branch selection on CPU before allocating GPUs.
2. Complete the existing boundary/failure, hybrid LoRA and full ZeRO-3 export
   tests listed in the GPU roadmap.
3. Add a real-tokenizer end-to-end path: raw examples -> masked batches ->
   training -> export/merge -> fresh-process generation -> task evaluation.
4. Add vocabulary-expansion variants that verify token IDs and tokenizer/model
   consistency through save, resume, merge and reload.
5. Verify that missing scoring data, incompatible adaptation modes, mismatched
   vocabulary artifacts and failed upstream stages prevent downstream execution.

The first implementation should run locally using plain Python and ordinary
paths. External infrastructure adapters can be supplied independently by users;
they are not part of this workflow's implementation requirements.
