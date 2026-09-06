# Qwen3.5 text-only SFT

This path intentionally uses `Qwen3_5ForCausalLM` and `AutoTokenizer`; it does not load a vision processor or require image inputs.

## Synthetic category smoke data

```bash
python examples/generate_category_sft.py /tmp/category-50k.jsonl --rows 50000 --seed 42
```

Each row contains `system`, `user`, and `assistant`. The example training script masks all prompt tokens with `-100`, so loss is computed only on the category answer tokens.

## Single-node GPU training

```bash
torchrun --standalone --nproc_per_node=4 examples/train_qwen35_category_sft.py \
  /tmp/category-50k.jsonl \
  --model Qwen/Qwen3.5-9B \
  --output /tmp/qwen35-category-sft \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 256 \
  --epochs 1
```

The example uses BF16, gradient checkpointing, and DeepSpeed ZeRO-3. It is intended for a single four-GPU node such as `g5.12xlarge` or `g6e.12xlarge`.

## CPU validation

CPU CI instantiates a tiny real `Qwen3_5ForCausalLM` from `Qwen3_5TextConfig`, runs forward + LM loss + backward, validates loader routing for both `qwen3_5` and `qwen3_5_text`, and verifies answer-only label masking.
