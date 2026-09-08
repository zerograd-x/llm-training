# Qwen3.5 AWS Training Runbook

This runbook records the operational checks used for the validated single-node Qwen3.5-4B SFT path on AWS.

## Validated setup

- Instance: `g5.12xlarge`
- GPUs: 4 x NVIDIA A10G
- Model: `Qwen/Qwen3.5-4B`
- Launch: single-node `torchrun --nproc_per_node=4`
- Precision: BF16
- Distributed backend: DeepSpeed ZeRO-3
- Trainable parameters: last 2 transformer layers only
- Input embedding: frozen
- LM head: frozen
- Per-GPU batch size: 8
- Gradient accumulation: 1
- Effective global batch: 32
- Padding: dynamic padding to the longest sequence in each local batch
- `max_length=256` is a truncation ceiling, not a fixed padded sequence length

## Where training logs live

Each run should have its own run directory. Example:

```text
/home/ubuntu/qwen35-100step/
├── category-3200.jsonl
├── run-meta.txt
├── train.log
└── output/
```

The main training log is:

```bash
/home/ubuntu/qwen35-100step/train.log
```

For a longer run, replace the run directory accordingly.

## Watch optimizer steps and loss

The trainer logs every 10 optimizer steps by default.

```bash
grep -E 'step=[0-9]+' /home/ubuntu/qwen35-100step/train.log | tail -20
```

Example:

```text
step=10 loss=2.137265
step=20 loss=0.018190
step=30 loss=0.000091
...
step=100 loss=0.000005
```

Important: `step` means optimizer step, not micro-batch. With `batch_size=8`, 4 GPUs, and `grad_accum=1`, each optimizer step processes 32 examples globally.

## Tail the raw log

```bash
tail -n 100 /home/ubuntu/qwen35-100step/train.log
```

Follow it continuously:

```bash
tail -f /home/ubuntu/qwen35-100step/train.log
```

Look for:

- `step=<N> loss=<value>`
- `Traceback`
- `OutOfMemoryError`
- `ChildFailedError`
- `NaN` / `nan`
- NCCL or DeepSpeed errors

Warnings about deprecated collective APIs are not necessarily fatal; use the process exit status and traceback to determine failure.

## Check GPU utilization and memory

```bash
nvidia-smi
```

Compact view:

```bash
nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits
```

Validated dynamic-padding/batch-8 training used roughly 9.9-10.1 GiB per A10G during steady-state training.

To see which process owns GPU memory:

```bash
nvidia-smi \
  --query-compute-apps=pid,used_memory \
  --format=csv,noheader,nounits
```

If GPU memory unexpectedly doubles, check for duplicate `torchrun` jobs before debugging ZeRO-3.

## Check for duplicate training jobs

```bash
ps -eo pid,ppid,etime,pcpu,pmem,cmd \
  | grep -E 'torchrun|train_qwen35_category_sft.py' \
  | grep -v grep
```

There should be one `torchrun` parent plus four worker processes for a single 4-GPU job.

## Check checkpoints

Example checkpoint directory:

```text
/home/ubuntu/qwen35-100step/output/checkpoint-00000100/
```

Inspect it with:

```bash
find /home/ubuntu/qwen35-100step/output/checkpoint-00000100 \
  -maxdepth 2 -type f | sort
```

A ZeRO-3 checkpoint should contain per-rank model and optimizer shards under `pytorch_model/`, plus trainer state, scheduler state, random states, tokenizer files, and `zero_to_fp32.py`.

A training checkpoint is a resumable training artifact. It is not the same thing as the final Hugging Face inference model.

## Final-model layout

A completed export should produce:

```text
output/final-model/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── chat_template.jinja
├── pytorch_model-00001-of-00002.bin
├── pytorch_model-00002-of-00002.bin
└── pytorch_model.bin.index.json
```

The exact number of weight shards can vary.

## Fresh-process reload validation

Do not validate the exported model inside the same active DeepSpeed distributed process. Exit `torchrun`, then start a new single Python process.

Example:

```bash
python - <<'PY'
import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM

path = '/home/ubuntu/qwen35-100step/output/final-model'
tok = AutoTokenizer.from_pretrained(path)
model = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
inputs = tok('Classify this product: wireless mouse', return_tensors='pt')
with torch.no_grad():
    out = model(**inputs)
print(tuple(out.logits.shape), torch.isfinite(out.logits).all().item())
PY
```

Validated result:

```text
shape (1, 13, 248320) finite True
PASS fresh-process final-model reload
```

The important checks are that loading succeeds, logits have the expected vocabulary dimension, and all logits are finite.

## Verify S3 artifacts

Validated prefix example:

```text
s3://llm-training-gpu-513586111913-20260904-r2/
  qwen35-4b-category-sft/20260908-100step/
```

List it with:

```bash
aws s3 ls \
  s3://llm-training-gpu-513586111913-20260904-r2/qwen35-4b-category-sft/20260908-100step/ \
  --recursive
```

Verify both:

```text
checkpoint-00000100/
final-model/
```

## S3 permissions required by `aws s3 sync`

The EC2 role needs object write permission for the artifact prefix and bucket-list permission for the prefix because `aws s3 sync` calls `ListObjectsV2` before copying.

At minimum:

```text
s3:PutObject on bucket/qwen35-4b-category-sft/*
s3:ListBucket on bucket with prefix qwen35-4b-category-sft/*
```

The training example is defensive: an artifact-sync failure should warn and preserve local training/checkpoint/export progress rather than terminate the training job.

## ZeRO-3 memory interpretation

Do not expect `nvidia-smi` to show only the persistent model shard size.

For a 4-GPU ZeRO-3 run, persistent model parameters are partitioned, but total GPU memory also includes:

- optimizer-state shards
- gradient shards
- activations
- temporary parameter all-gather buffers
- CUDA context
- allocator/cache

Also note that this project currently uses `zero3_init_flag=False`, so each rank may temporarily load the full model before `accelerator.prepare()` partitions it.

## Fast troubleshooting checklist

```bash
# 1. Is the process alive?
ps -eo pid,etime,cmd | grep -E 'torchrun|train_qwen35' | grep -v grep

# 2. Is training progressing?
grep -E 'step=[0-9]+' RUN_DIR/train.log | tail -20

# 3. Are GPUs active and memory sane?
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits

# 4. Any fatal errors?
grep -Ei 'traceback|outofmemory|childfailederror|nan|nccl.*error' RUN_DIR/train.log | tail -50

# 5. Did a checkpoint appear?
find RUN_DIR/output -maxdepth 1 -type d -name 'checkpoint-*' -print

# 6. Did final-model appear?
find RUN_DIR/output/final-model -maxdepth 1 -type f -print

# 7. Did S3 receive both checkpoint and final-model?
aws s3 ls S3_PREFIX --recursive
```

## Validated 100-step lifecycle

The validated short run used 3,200 synthetic examples:

```text
3200 examples / (8 examples per GPU x 4 GPUs) = 100 optimizer steps
```

It successfully demonstrated:

```text
4-GPU ZeRO-3 training
→ step 100
→ local checkpoint-00000100
→ ZeRO-3 shard reconstruction
→ standard final-model
→ fresh-process reload and forward
→ S3 persistence
```
