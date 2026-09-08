# Qwen3.5 4B text-only SFT

This path intentionally uses `Qwen3_5ForCausalLM` and `AutoTokenizer`; it does not load a vision processor or require image inputs.

## Reproducible GPU bootstrap

On a fresh Ubuntu/DLAMI GPU host, run from any working directory:

```bash
bash scripts/bootstrap_gpu.sh
```

The script resolves the repository root from its own location, installs missing Ubuntu prerequisites such as `python3-venv`, `git`, `curl`, and `awscli`, creates `.venv`, installs the project with DeepSpeed/dev extras, applies the tested GPU dependency constraints from `constraints/gpu-tested.txt`, and validates CUDA, BF16, GPU count, and the Qwen3.5 model import.

The current tested stack is intentionally separated from the supported-version ranges in `pyproject.toml`: `pyproject.toml` is the compatibility contract, while `constraints/gpu-tested.txt` is the reproducible AWS GPU environment verified by this project.

## Synthetic category data

```bash
.venv/bin/python examples/generate_category_sft.py /tmp/category-50k.jsonl --rows 50000 --seed 42
```

Each row contains `system`, `user`, and `assistant`. The training script masks all prompt tokens with `-100`, so loss is computed only on the category answer tokens.

## Artifact lifecycle

Training writes resumable DeepSpeed checkpoints to the local `--output` directory. After training, `trainer.export_model()` writes a standard Hugging Face model under `final-model/`.

For EC2 runs, pass `--artifact-s3` so local artifacts survive instance termination. After every checkpoint save, rank 0 mirrors the output directory to S3; after `final-model/` export, it syncs once more.

Example durable layout:

```text
s3://BUCKET/qwen35-4b/category-sft/RUN_ID/
  checkpoint-00000500/
  checkpoint-00001000/
  ...
  final-model/
    config.json
    *.safetensors
    tokenizer.json
    ...
```

The EC2 filesystem is still used for active training speed. S3 is the durable artifact store.

## Trainable-parameter policy

The Qwen3.5-4B category SFT path freezes the base model except for the final two transformer/hybrid layers and the LM head:

```text
input embedding       requested frozen
all early layers      frozen
last 2 layers         trainable
LM head               trainable
```

Qwen3.5 ties the input embedding and LM-head weight tensor. Therefore, when `train_lm_head=True`, the shared embedding/LM-head tensor is effectively trainable even though `train_embeddings=False` is requested. The runtime emits a warning for this tied-weight behavior.

## Single-node 4-GPU smoke

Use this first on `g5.12xlarge` or `g6e.12xlarge`:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  tests/gpu/qwen35_4b_single_node.py \
  --model Qwen/Qwen3.5-4B \
  --output /tmp/qwen35-4b-gpu-smoke
```

The smoke test requires exactly four CUDA ranks, BF16 support, DeepSpeed ZeRO-3, and gradient checkpointing. It verifies the selective-freeze policy, performs a real optimizer update, writes a checkpoint, exports `final-model/`, frees the distributed training model, reloads the exported model, and runs a finite-logit forward pass.

## 50K category SFT

After the smoke passes:

```bash
.venv/bin/torchrun --standalone --nproc_per_node=4 \
  examples/train_qwen35_category_sft.py \
  /tmp/category-50k.jsonl \
  --model Qwen/Qwen3.5-4B \
  --output /mnt/training/qwen35-4b-category-sft \
  --artifact-s3 s3://BUCKET/qwen35-4b/category-sft/RUN_ID/ \
  --batch-size 1 \
  --grad-accum 8 \
  --max-length 256 \
  --epochs 1
```

The default training path is selective BF16 SFT of the last two layers plus the tied LM-head/embedding tensor, with gradient checkpointing and DeepSpeed ZeRO-3, on one four-GPU node only. No cross-node configuration is required.

## Validation sequence

1. Bootstrap the tested GPU environment with `scripts/bootstrap_gpu.sh`.
2. Load the official `Qwen/Qwen3.5-4B` checkpoint through the text-only `Qwen3_5ForCausalLM` path.
3. Confirm four ranks and CUDA/BF16.
4. Confirm the selective trainable-parameter policy.
5. Run one real ZeRO-3 optimizer update with GC enabled.
6. Confirm checkpoint creation and record peak VRAM.
7. Export a standard Hugging Face `final-model/`.
8. Reload `final-model/` in a fresh model object and run forward.
9. Verify checkpoints and `final-model/` exist in S3 before EC2 termination.
10. Run the 50K-row SFT job and exercise checkpoint resume before considering the GPU path complete.

## CPU validation already completed

A tiny real `Qwen3_5ForCausalLM` built from `Qwen3_5TextConfig` passed forward + LM loss + backward on CPU. Qwen3.5 layer/embedding/LM-head introspection and gradient checkpointing also passed. CPU tests in this PR cover loader routing for both `qwen3_5` and `qwen3_5_text`, plus answer-only label masking.
