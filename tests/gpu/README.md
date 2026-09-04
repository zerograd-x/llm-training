# GPU validation workers

These scripts were **reconstructed**, not recovered byte-for-byte from the
temporary September 4 workers. Their reference is main commit
`3ba90dd119c4dbf6e67319444dcd46f1f1dbf7a1`, its existing CPU integration tests,
and the retained SSM logs indexed in [the matrix](../../docs/gpu-validation-matrix.md).

The original scripts lived only at `/tmp/{fresh_case,tt_probe,gpu_validation}.py`
on the temporary L4 and in a subsequently deleted S3 staging bucket. The
accessible conversation and local session records do not contain their full
source. Do not treat this reconstruction as evidence that the historical
assertions ran or passed.

## Files

- `gpu_validation.py`: standard-library driver, isolated subprocesses, phase
  logs, incremental JSON matrix, nonzero exit for failed/missing results.
- `fresh_case.py`: GPU-only phases for training, unfinished checkpoint/resume,
  optimizer/LR tracing, adapter-only saves and offline merge/reload.
- `tt_probe.py`: effective embedding-row checks through the adapter forward.
- `test_validation_driver.py`: CPU-only checks against false-positive results.

## Run

Use one NVIDIA L4 or equivalent BF16 GPU and Python 3.10. Match PyTorch's CUDA
build to the installed toolkit; the original environment used torch 2.7.1+cu128,
CUDA toolkit 12.8, DeepSpeed 0.19.6, Accelerate 1.14.0, Transformers 5.16.1 and
PEFT 0.20.0. Install the exact source being tested with `pip install -e .` and
record its commit. The worker records the actual installed Trainer file hash.

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
unset DS_SKIP_CUDA_CHECK
python tests/gpu/gpu_validation.py /tmp/llm-gpu-run-unique
```

Output must be a new/empty directory. `--cases lora_bf16 lora_zero3` narrows the
suite. No cloud resource is created by these scripts. The driver exposes only
GPU 0 and rejects a multi-rank launch; do not use it for 2-GPU ZeRO-3.

Each case has prepare, uninterrupted baseline, interrupt after step 1, fresh
resume to step 2, and fresh offline verification processes. Four microbatches
with accumulation 2 exercise real optimizer boundaries. Exact restored and
final parameter equality, RNG restore, LR sequence, batch order, trainable
coverage, and frozen parameter stability are asserted. Export/merge logits use
explicit BF16 tolerances (atol 0.005, rtol 0.05). These are tiny-model integration
checks, not accuracy/performance validation for a production model.

The default TrainableTokens ZeRO-3 case starts from a CPU-loaded model to expose
the public initialization path; device placement is handled by Trainer,
without a CUDA-preload workaround in the test. For clean offload, the driver uses a new extension cache, rejects an
inherited CUDA-check bypass, records nvcc/CPUAdam module information and checks
the real optimizer class. A clean extension build does not by itself pass the
remaining training/resume/export phases.

```bash
python tests/gpu/test_validation_driver.py
```

Preserve the scripts, source commit, full output directory and `matrix.json`
before removing temporary GPU resources. A successful process is accepted only
if its JSON result also says `ok: true`.

## Current validation status

The September 4 rerun passes all eight cases / 40 fresh-process phases on one
NVIDIA L4 with the pinned stack above. The fixed implementation also passes
70 CPU regressions. Original-main failures and exact tested source are archived
and detailed in the matrix. The CPU regressions include later-epoch resume with
and without dropout and scheduler ownership after optimizer state restoration.
A documented optimizer-step wrapper warning remains for CPUAdam; actual update
ordering, LR and resume equivalence are asserted instead of hiding the warning.

## Two-node, two-GPU validation

`multinode_validation.py` coordinates two drivers over a **new shared output
directory**. `multirank_case.py` executes CUDA/NCCL phases and validates the
rank, sharding, resume and artifact contracts. Each host must have exactly one
visible BF16 GPU, the same source/environment, and access to the shared path.
Allow the two hosts to communicate privately for torchrun/NCCL and shared
storage. The scripts do not provision cloud infrastructure.

Run these concurrently, one command on each host (replace `NODE0_PRIVATE_IP`):

```bash
# Node 0
python tests/gpu/multinode_validation.py /shared/unique-run \
  --node-rank 0 --master NODE0_PRIVATE_IP
# Node 1
python tests/gpu/multinode_validation.py /shared/unique-run \
  --node-rank 1 --master NODE0_PRIVATE_IP
```

Both commands must use identical `--cases` and shared directory arguments.
The cases are `runtime_ddp`, `lora_zero3`, `lora_zero3_gc`, `tt_zero3` and
`tt_zero3_gc`. Each has prepare, baseline, interrupt, fresh-process resume and
offline verify phases. Baseline/interrupt/resume each launch one worker per
host; prepare/verify run on node 0 after the relevant workers exit.

Assertions cover real parameter partitioning, disjoint training batches,
world-size-aware LR, actual optimizer updates, exact checkpoint/RNG restoration,
exact interrupted/uninterrupted parameter and output equivalence, frozen/base
row stability, agreement between ranks, adapter-only gathering, rank-0-only
serialization and offline adapter/merge reload. The GC cases explicitly use
`use_reentrant=False`. A timed-out process group is terminated; both exit
status and result JSON must succeed. Keep the entire shared output, including
failed attempts, source and environment, before deleting workers.

This topology tests global ranks 0/1 with **local rank 0 on each host**. It does
not test local CUDA indices 0/1 on one machine or node-local-only checkpoints.
See the [P0/P1 matrix](../../docs/gpu-p0-p1-validation.md) for the actual run.
