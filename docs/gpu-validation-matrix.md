# Single-GPU validation — 2026-09-04 UTC

**Fixed source: 8 cases / 40 fresh-process phases passed on one NVIDIA L4.**
The CPU regression suite passes **70 tests**. Original main passed six complete
GPU cases and failed two. The [earlier historical audit](gpu-validation-history-20260904.md)
is retained separately; its outstanding single-card gates are resolved here.

## Source and environment

Base main: `3ba90dd119c4dbf6e67319444dcd46f1f1dbf7a1`.
Fixes and permanent tests: [PR #1](https://github.com/zerograd-x/llm-training/pull/1).
One NVIDIA L4 (`g6.xlarge`, us-west-2), world size 1, Python 3.10,
torch 2.7.1+cu128, CUDA toolkit 12.8, DeepSpeed 0.19.6, Accelerate 1.14.0,
Transformers 5.16.1, PEFT 0.20.0, llm-training 0.1.0.

The workers assert actual device, CUDA/BF16 support, versions, world size and
DeepSpeed stage. Tested core files are archived and committed byte-for-byte:

- Trainer SHA256: `99fb7667a485c10dcbc09a622853416ecb67981dd5945cef30df1bb1234ba630`
- Scheduler SHA256: `298a93066831626a2a27cd2593902ba60333053d1f644fd009cd589971c7bb1a`

## GPU matrix

Each complete pass means prepare → uninterrupted baseline → interrupt at step 1
→ new-process resume to step 2 → new-process artifact verification.

| Case | Original main | Fixed source | Reload max absolute logit error |
| --- | --- | --- | ---: |
| LoRA + BF16 | Pass | Pass | 0.0009765625 |
| LoRA + ZeRO-3 | Pass | Pass | 0.0010986328125 |
| LoRA + gradient checkpointing | Pass | Pass | 0.0009765625 |
| LoRA + ZeRO-3 + gradient checkpointing | Pass | Pass | 0.0010986328125 |
| TrainableTokens + BF16 | Pass | Pass | 0 |
| TrainableTokens + ZeRO-3, CPU-loaded public path | CPU/NCCL initialization fails | Pass | 0.0009765625 |
| Torch AdamW + HF scheduler + ZeRO-3 | Pass | Pass | 0.0009765625 |
| Clean CPUAdam offload + ZeRO-3 | Build/train pass; resumed actual LR fails | Pass | 0.0009765625 |

Four microbatches with accumulation 2 exercise actual optimizer boundaries.
Hooks verify that real optimizer updates precede scheduler advancement and that
LR follows 0.01 → 0.005 → 0. Checkpoint tensors and RNG are checked upon restore;
final parameters and training logits match the uninterrupted control exactly,
and observed batches equal interrupt + resume batches.

Adapter cases check pre-prepare optimizer coverage, trainable names, parameter
mutation, frozen parameters and adapter-only file contents. Adapter reload and
merged-model reload use explicit BF16 tolerances: `atol=0.005, rtol=0.05`.
TrainableTokens checks effective embedding rows: unselected rows are bit-identical,
selected rows change, and old rows also remain identical after merge.
Gradient checkpointing is non-reentrant with model caching disabled.

Clean offload builds CPUAdam in a fresh extension directory, records compiler
commands/module paths, rejects inherited `DS_SKIP_CUDA_CHECK`, and verifies both
nvcc and torch CUDA are 12.8. The real optimizer is `DeepSpeedCPUAdam`.
Resume, full-model export and fresh-process reload all pass.

## Four defects repaired

1. **TrainableTokens initialization device:** PEFT broadcasts selected token
   weights before Accelerate prepares the model. Trainer now places the model
   on the rank's CUDA device before TrainableTokens adaptation under ZeRO-3.
   The test itself still supplies an ordinary CPU-loaded model.
2. **Extra RNG consumption on resume:** DataLoader iterator creation consumed
   the global torch RNG. A dedicated loader generator keeps an extra resumed
   iterator from changing CPU dropout's random sequence.
3. **Resumed epoch sampling:** New prepared/skip loaders start at iteration zero.
   Trainer explicitly sets the loader epoch after constructing it, preserving
   sample order during both the resumed and subsequent epochs.
4. **Offload scheduler after restore:** AdamW and replacement CPUAdam initially
   share parameter-group dictionaries. Checkpoint loading replaces CPUAdam's
   groups, leaving the HF scheduler attached to obsolete groups. Trainer now
   binds the scheduler to the actual prepared optimizer before restoration.

CPU resume regressions interrupt during the first or second epoch, with or
without dropout. All four variants failed on original main and pass after the
repair. Another CPU regression checks scheduler updates after optimizer state
loading replaces parameter-group dictionaries.

## Scheduler warning diagnosis

Without offload, no ordering warning was captured. With offload, original main
warned about scheduler-before-optimizer even though the trace proves CPUAdam
updates happened first and uninterrupted LR was correct. PyTorch tracked the
original AdamW step while DeepSpeed called CPUAdam: the warning was a wrapper
artifact. The resumed LR disconnection above was a separate functional defect.

After rebinding, PyTorch emits an optimizer-step-overridden warning because
CPUAdam was substituted after scheduler construction. This warning is retained;
there is no blanket suppression or fabricated step flag. Actual step ordering,
resumed LR, final parameters and exported logits pass. Relevant upstream code:
[Accelerate preparation](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/accelerator.py),
[DeepSpeed wrappers](https://github.com/huggingface/accelerate/blob/v1.14.0/src/accelerate/utils/deepspeed.py),
[PyTorch scheduler](https://github.com/pytorch/pytorch/blob/v2.7.1/torch/optim/lr_scheduler.py),
[PEFT token initialization](https://github.com/huggingface/peft/blob/v0.20.0/src/peft/tuners/trainable_tokens/layer.py).

## Evidence

Run instructions: [tests/gpu](../tests/gpu/README.md).
Instance: `i-013a896804f5260da`; temporary resource prefix:
`llm-training-gpu-20260904-r2`.

| Evidence | Identifier / SHA256 |
| --- | --- |
| Main GPU run | `71d98f76-fbb6-41ae-af38-2673c2b6be05` |
| Fixed GPU + 70 CPU tests | `a1f369e3-51ab-4105-b11b-e620a56e1524` |
| Main CPU resume failures | `5aaeb817-7f19-44f9-965a-d47be1ebfb3c` |
| Fixed isolated CPU resume | `c554179e-531d-4f8a-b460-0c281caf45f5` |
| Main source archive | `2d318d3248c8e1d2627c8c177080624ba9f34776830ebfa789892984a194fce4` |
| Fixed tested source archive | `04010994abaaee860aabe46025ca5f9ef991ed8cf7e1d470a1e4d710b2ce14ab` |
| Main results archive | `89ed63fa8b6c0c568a3d5213510d3c5308c73b21a09a9e4fcadfa5fbdbee615f` |
| Fixed results archive | `69d14d7437f37808e286bfab0b742ae072124f8922b7ed7c21aadee486acc076` |

Archives include per-phase JSON/logs, checkpoints, adapters, merged models,
environment versions and exact tested source. All were downloaded and checksum
verified before terminating the GPU. Temporary S3 paths are not durable download
links; retain the delivered local evidence bundle.

## Remaining scope

These tiny GPT-2 fixtures validate correctness on the pinned single-GPU stack.
They do not establish model quality, throughput, maximum size, or compatibility
across every permitted dependency version. TrainableTokens CUDA initialization
places the unpartitioned model on GPU; its full initial footprint must fit.
ZeRO-3 initialization-time partitioning remains unimplemented.

This single-GPU round was followed by the [P0/P1 two-node run](gpu-p0-p1-validation.md),
which validates NCCL, disjoint data sharding, world-size-aware scheduling and
LoRA/TrainableTokens ZeRO-3 checkpoint and artifact collectives on two L4 hosts.
Later coverage: realistic models, tied/untied embeddings, uneven accumulation
and epoch boundaries, other optimizer ownership combinations and performance.
