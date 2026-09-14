# Run training from YAML or JSON

After installing the wheel, common SFT, causal-LM, and DPO jobs can be launched
without writing Python:

```bash
pip install 'llm-training[peft]'
llm-train --config train.yaml --dry-run
llm-train --config train.yaml
```

`--dry-run` validates the configuration and local JSONL datasets, then prints
the effective training plan without loading model weights or creating output
artifacts.

## LoRA SFT example

```yaml
model:
  name_or_path: Qwen/Qwen3.5-4B
  dtype: bfloat16

data:
  train_file: train.jsonl
  validation_file: validation.jsonl
  format: sft
  prompt_field: prompt
  completion_field: completion
  max_sequence_length: 2048

training:
  output_dir: outputs/qwen35-lora
  num_epochs: 1
  per_device_batch_size: 1
  gradient_accumulation_steps: 8
  mixed_precision: bf16
  max_steps: 100
  log_every_steps: 10

lora:
  rank: 8
  alpha: 16
  dropout: 0.0
  target_last_n_layers: 8

optimizer:
  learning_rate: 0.00002
  weight_decay: 0.1

scheduler:
  name: cosine
  warmup_ratio: 0.03

gradient_checkpointing:
  enabled: true
  use_reentrant: false

checkpoint:
  save_every_steps: null
  keep_last: 1

artifacts:
  save: auto
```

SFT input is JSONL with one object per line:

```json
{"prompt":"Classify this product: cotton t-shirt. Category:","completion":" apparel"}
```

Prompt tokens receive label `-100`; loss is computed on completion tokens. The
collator dynamically pads each batch and appends the tokenizer EOS token.

With LoRA or TrainableTokens, `artifacts.save: auto` saves the adapter. Otherwise
it exports a Hugging Face model under `OUTPUT_DIR/final-model`.

## Causal-LM example

```yaml
model:
  name_or_path: Qwen/Qwen3.5-4B
  dtype: bfloat16

data:
  train_file: documents.jsonl
  format: causal_lm
  text_field: text
  max_sequence_length: 2048

training:
  output_dir: outputs/qwen35-pretrain
  num_epochs: 1
  per_device_batch_size: 1
  mixed_precision: bf16

checkpoint:
  save_every_steps: null
  keep_last: 1
```

Input:

```json
{"text":"Causal language modeling predicts the next token from earlier tokens."}
```

This first config-driven version treats every JSONL row as one document and
truncates it to `max_sequence_length`. It does not pack multiple documents into
one sequence. Use the Python API and `llm-training-data` when deterministic
packing, streaming mixtures, or custom transforms are required.

## DPO preference training

Set `data.format: dpo` and provide `prompt`, `chosen`, and `rejected`
strings in each JSONL row. DPO uses the same CLI and shared training sections,
with preference-specific controls under `dpo:`.

Install `llm-training[dpo]`, plus `peft` for LoRA. See
[`dpo.md`](dpo.md) and
[`examples/train_dpo.yaml`](../examples/train_dpo.yaml) for the complete
schema, reference-model behavior, and multi-GPU launch command.

## Partial and full fine-tuning

To train only selected base-model parameters, add `freeze`:

```yaml
freeze:
  train_last_n_layers: 2
  train_embeddings: false
  train_lm_head: false
```

Omit both `lora` and `freeze` for full fine-tuning. Config-driven LoRA rejects a
simultaneous `freeze` section because saving only an adapter would lose updated
base-model parameters; use the Python API when a custom hybrid artifact lifecycle
is intentional.

## DeepSpeed

Install the extra and launch one CLI process per rank:

```bash
pip install 'llm-training[deepspeed]'
torchrun --standalone --nproc_per_node=4 \
  $(command -v llm-train) --config train.yaml
```

```yaml
deepspeed:
  zero_stage: 3
  zero3_save_16bit_model: true
```

The same configuration file is read by every rank. The runtime owns Accelerate
and DeepSpeed initialization, checkpointing, and final export.

## Configuration behavior

- `.yaml`, `.yml`, and `.json` use the same schema.
- Unknown top-level and nested fields fail instead of being ignored.
- `model.model_kwargs` and `model.tokenizer_kwargs` expose advanced Hugging Face
  arguments without allowing them to override explicit dtype/trust settings.
- `artifacts.save` is one of `auto`, `model`, `adapter`, or `none`.
- Dataset paths are local JSONL files in this version.
- Custom loss modules remain configurable through the existing `loss` section.

The Python API remains the extension point for custom datasets, collators,
metrics, packed/streaming data, remote storage, and specialized artifact flows.
