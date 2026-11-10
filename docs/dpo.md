# Config-driven DPO training

`llm-training` supports Direct Preference Optimization through the same
`llm-train --config` entry point used for SFT and causal-LM training. Set
`data.format: dpo`; the runtime then uses TRL's `DPOTrainer` while reusing
this package's model loading, LoRA/freeze policy, DeepSpeed settings, precision,
checkpoint cadence, and artifact policy.

## Install

For full-model DPO:

```bash
pip install "llm-training[dpo]"
```

For the usual LoRA path:

```bash
pip install "llm-training[dpo,peft]"
```

Add the `deepspeed` extra when using ZeRO:

```bash
pip install "llm-training[dpo,peft,deepspeed]"
```

The DPO and GRPO extras use the same supported TRL 1.12 release. The runtime
preserves `max_prompt_length` by truncating prompts before handing the dataset
to TRL, whose newer DPO configuration no longer exposes that field.

## Preference data

The input is local JSONL. Each non-empty row must contain one prompt and two
different completions:

```json
{"prompt":"What is 2 + 2?","chosen":"4","rejected":"5"}
{"prompt":"Name a primary color.","chosen":"Blue.","rejected":"Transparent."}
```

Column names can be changed without rewriting the data:

```yaml
data:
  train_file: data/preferences.jsonl
  validation_file: data/preferences-validation.jsonl
  format: dpo
  prompt_field: question
  chosen_field: winner
  rejected_field: loser
```

This CLI accepts local JSONL in the standard string preference format.
Convert other data sources to `prompt/chosen/rejected` JSONL before calling
the CLI.

## Configuration

A complete LoRA example is in
[`examples/train_dpo.yaml`](../examples/train_dpo.yaml). The DPO-specific
section is:

```yaml
dpo:
  beta: 0.1
  loss_type: sigmoid       # sigmoid, hinge, or ipo
  label_smoothing: 0.0
  max_length: 1024
  max_prompt_length: 512
  reference_free: false
  system_prompt: You are a helpful and precise assistant.
```

- `beta` controls the strength of the preference objective relative to the
  reference policy.
- `loss_type` selects the TRL DPO objective.
- `label_smoothing` models noisy preference labels and must be below `0.5`.
- `max_length` bounds the combined prompt and completion.
- `max_prompt_length` bounds the prompt and cannot exceed `max_length`.
- `reference_free: false` uses the initial policy as the reference.
- `system_prompt`, when present, renders each prompt through the tokenizer's
  chat template as a system turn followed by a user turn.

The usual `training`, `optimizer`, `scheduler`,
`gradient_checkpointing`, `lora`, `freeze`, `evaluation`,
`checkpoint`, `deepspeed`, and `artifacts` sections are shared with
SFT.

DPO owns its preference loss, so the custom `loss` section is not accepted.
Standalone `trainable_tokens` is not supported for DPO yet. Vocabulary
extension is still supported by pointing `model.name_or_path` at the already
extended base checkpoint.

## Validate and run

Validate configuration and every JSONL row without loading a model:

```bash
llm-train --config examples/train_dpo.yaml --dry-run
```

Run on one process:

```bash
llm-train --config examples/train_dpo.yaml
```

Run on multiple GPUs through Accelerate:

```bash
accelerate launch --num_processes 8 -m llm_training.cli \
  --config examples/train_dpo.yaml
```

The dry run verifies required fields, rejects unknown configuration fields,
checks that chosen and rejected differ, and prints the effective batch,
precision, optimizer, DeepSpeed, checkpoint, and evaluation plan. It does not
load or download the model and does not use a GPU.

## Reference-model behavior

The caller does not need to configure a second model path:

- With LoRA, TRL evaluates the reference policy with the adapter disabled.
- With full or selective base-model training, TRL creates a frozen copy of the
  initial policy.
- `reference_free: true` is rejected by the supported TRL release because its\n  current DPO API no longer exposes the reference-free objective.

Full-model DPO therefore needs memory for both policy and reference weights.
LoRA is normally the practical choice for a single GPU.

## DeepSpeed and checkpoint ownership

For DPO, Hugging Face Trainer owns the optimizer and scheduler. The shared
`optimizer` and `scheduler` sections feed its arguments. DeepSpeed receives
only ZeRO/offload, precision, clipping, and automatic batch-size settings;
`deepspeed.optimizer` and `deepspeed.scheduler` are rejected to avoid two
owners defining the same state.

By default, `checkpoint.save_every_steps: null` writes only the final model or
adapter. Set a positive cadence to enable TRL/Hugging Face step checkpoints;
`checkpoint.keep_last` controls retention. Under ZeRO-3, every rank
participates in the final save while only the main process writes tokenizer
files.

Artifact behavior is unchanged:

```yaml
artifacts:
  save: adapter  # LoRA
```

Use `save: model` for full-model DPO, `save: auto` to infer this choice, or
`save: none` when no final artifact should be written.
