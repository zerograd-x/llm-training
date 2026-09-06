# Qwen3.5 4B text-only SFT

This path intentionally uses `Qwen3_5ForCausalLM` and `AutoTokenizer`; it does not load a vision processor or require image inputs.

## Synthetic category data

```bash
python examples/generate_category_sft.py /tmp/category-50k.jsonl --rows 50000 --seed 42
```

Each row contains `system`, `user`, and `assistant`. The training script masks all prompt tokens with `-100`, so loss is computed only on the category answer tokens.

## Single-node 4-GPU smoke

Use this first on `g5.12xlarge` or `g6e.12xlarge`:

```bash
torchrun --standalone --nproc_per_node=4 \
  tests/gpu/qwen35_4b_single_node.py \
  --model Qwen/Qwen3.5-4B \
  --output /tmp/qwen35-4b-gpu-smoke
```

The smoke test requires exactly four CUDA ranks, BF16 support, DeepSpeed ZeRO-3, and gradient checkpointing. It performs a real optimizer update, writes a checkpoint, and prints peak allocated GPU memory from rank 0.

## 50K category SFT

After the smoke passes:

```bash
torchrun --standalone --nproc_per_node=4 \
  examples/train_qwen35_category_sft.py \
  /tmp/category-50k.jsonl \
  --model Qwen/Qwen3.5-4B \
  --output /tmp/qwen35-4b-category-sft \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 256 \
  --epochs 1
```

The default training path is full-parameter BF16 SFT with gradient checkpointing and DeepSpeed ZeRO-3, on one four-GPU node only. No cross-node configuration is required.

## Validation sequence

1. Load the official `Qwen/Qwen3.5-4B` checkpoint through the text-only `Qwen3_5ForCausalLM` path.
2. Confirm four ranks and CUDA/BF16.
3. Run one real ZeRO-3 optimizer update with GC enabled.
4. Confirm checkpoint creation and record peak VRAM.
5. Run the 50K-row SFT job.
6. Exercise checkpoint resume before considering the GPU path complete.

## CPU validation already completed

A tiny real `Qwen3_5ForCausalLM` built from `Qwen3_5TextConfig` passed forward + LM loss + backward on CPU. Qwen3.5 layer/embedding/LM-head introspection and gradient checkpointing also passed. CPU tests in this PR cover loader routing for both `qwen3_5` and `qwen3_5_text`, plus answer-only label masking.
