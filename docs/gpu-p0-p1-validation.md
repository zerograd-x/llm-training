# P0/P1 GPU validation — 2026-09-04

P0 passes **8/8 single-GPU cases (40 fresh-process phases)** on main
`40d7fba1676a61017cc44819772987f60e0908da`. P1 passes **5/5 two-node cases
(25 coordinated phases, 40 worker processes)** after fixing the runtime's
model-loading context. Both P1 drivers exit zero. This is tiny-model correctness
validation, not a production-model quality or performance claim.

The previous audit branch was merged through PR #1, but its final report and
scheduler regression were committed after that merge. PR #2 merged those final
changes into the main commit above after five CI jobs passed. A retained branch
name does not imply its commits remain unmerged.

## Hardware and environment

- Two `g6.xlarge` EC2 hosts in `us-west-2`, each with one NVIDIA L4; 8 vCPUs total.
- P0 cases split across independent single-GPU processes on the two hosts.
- P1 uses NCCL, global ranks 0/1, local rank **0 on both hosts**, and private NFS
  shared checkpoint storage. Only same-security-group inbound traffic is allowed.
- Python 3.10; torch 2.7.1+cu128; CUDA toolkit 12.8; DeepSpeed 0.19.6;
  Accelerate 1.14.0; Transformers 5.16.1; PEFT 0.20.0; llm-training 0.1.0.
- All cases use BF16, tiny locally generated GPT-2 fixtures and synthetic rows.

## P0 — single GPU, current-main regression

Every row passes prepare, baseline, interruption after step 1, a new process
resuming to step 2, and a new offline verification process.

| Case | Result | Lifecycle/artifact |
|---|---|---|
| LoRA + BF16 | PASS | adapter save, reload, merge, HF reload |
| LoRA + ZeRO-3 | PASS | adapter save, reload, merge, HF reload |
| LoRA + gradient checkpointing | PASS | adapter save, reload, merge, HF reload |
| LoRA + ZeRO-3 + gradient checkpointing | PASS | adapter save, reload, merge, HF reload |
| TrainableTokens + BF16 | PASS | selected rows change; old rows stay exact; merge/reload |
| TrainableTokens + ZeRO-3 | PASS | selected rows change; old rows stay exact; merge/reload |
| Torch optimizer + HF scheduler + ZeRO-3 | PASS | actual updates/LR and resumed full-model export/reload |
| Clean CPUAdam offload | PASS | fresh native build; no DS_SKIP_CUDA_CHECK; resume/export/reload |

Checkpoint tensors, RNG, scheduler state, batch continuation and final parameters
are compared exactly against the uninterrupted run. Optimizer coverage and
frozen-weight stability are checked. CPUAdam's optimizer-step wrapper warning
remains documented: actual optimizer-before-scheduler ordering and LR correctness
pass. This round does not suppress it or claim the warning itself is a new bug.

## P1 — two nodes, two real GPUs

| Case | Baseline | Interrupt/resume | Adapter/full export, merge/reload |
|---|---|---|---|
| DDP distributed runtime | PASS | PASS | PASS, full-model export/reload |
| LoRA + ZeRO-3 | PASS | PASS | PASS |
| LoRA + ZeRO-3 + non-reentrant GC | PASS | PASS | PASS |
| TrainableTokens + ZeRO-3 | PASS | PASS | PASS |
| TrainableTokens + ZeRO-3 + non-reentrant GC | PASS | PASS | PASS |

Assertions include:

- Distinct hosts, correct global/local ranks, CUDA binding, NCCL collectives and
  reuse of the runtime's single Accelerator by Trainer.
- Global-main/local-main materialization ownership and rejection of offline
  merge inside distributed workers.
- Eight examples split into disjoint sets of four per rank, with complete union.
  Accumulation 2 gives two actual optimizer updates per rank. The underlying
  scheduler advances four times; LR is `0.01 -> 0.005 -> 0`.
- Actual ZeRO-3 storage partitioning, including adapter parameters. For example,
  the TrainableTokens delta has 32 elements globally and 16 stored on each rank.
- Exact optimizer coverage; trainable parameters change; frozen base parameters
  stay unchanged; final full-parameter digests agree across ranks.
- Both ranks restore checkpoint tensors, scheduler and RNG at logical position
  `(global_step=1, epoch=0, batch_in_epoch=2)`, then reach step 2. Final parameters
  and logits equal each rank's uninterrupted baseline exactly.
- Both ranks participate in adapter-only gathering, without frozen base weights;
  only global rank 0 calls adapter serialization. Artifacts contain adapter keys.
- Offline adapter reload, merge and plain-HF reload after all workers exit.
  BF16 comparisons use `atol=0.005, rtol=0.05`; merged reload maximum absolute
  logit error is 0 for DDP and 0.0009765625 for each PEFT case. TrainableTokens
  preserves old effective rows exactly, including after merge.

## Defect found and fixed

The first P1 run on unmodified main passed DDP but failed all four ZeRO-3
baselines before training. LoRA reported no discoverable `c_attn` targets;
TrainableTokens reported an embedding without a 2D weight. Later phases for
those cases were not run and are not counted as passing.

With Accelerate 1.14.0, `set_deepspeed_weakref()` installs a Transformers
ZeRO-3 configuration even when `zero3_init_flag=False`. Creating runtime before
`from_pretrained` therefore unexpectedly partitions weights before adapter
injection. The implementation's full-weight discovery cannot operate on those
partition placeholders. This was a real runtime-first loading compatibility bug.

`DistributedRuntime.model_loading_context()` temporarily suspends that hook and
restores it, including on exceptions. `runtime.load_causal_lm()` applies the
context automatically; custom model loaders must explicitly enter it. PEFT
collectives still see the restored DeepSpeed configuration before Trainer
prepares the model. Initial full model loading remains required; this change
does not implement initialization-time partitioning.

The rerun repeats all five P1 cases, including DDP. Six focused CPU checks pass,
including real HF loading plus nested-context/exception restoration regressions.
P0 predates this narrow loading-context patch; its model-before-Trainer path is
unchanged. The patched runtime is directly exercised by every P1 training phase.

## Reproducibility and evidence

Workers and launch instructions: [tests/gpu](../tests/gpu/README.md).
Machine-readable phase matrix: [results JSON](results/gpu-p0-p1-20260904.json).
Full logs/results include the failed first attempt and successful rerun.

S3 results object:
`s3://llm-training-gpu-513586111913-20260904-r2/p1/results/gpu-p0-p1-results-20260904.zip`

Archive SHA256:
`6e74c078b42afec41c3c18fd0a94c7ee3d8ea8fae763c36842d5136acbcdb432`

All 631 archived JSON/log/text files were downloaded and checksum-verified.
The S3 archive excludes source and binary test weights. Source is committed in
the repository and included in the local deliverable; fixtures are regenerated
by the workers. P0 uses the main commit above; P1 applies only the loading-context
runtime fix plus the new validation workers/regressions. Exact tested hashes:

| File | SHA256 |
|---|---|
| `src/llm_training/distributed.py` | `aebcf88c66d24f66c4567877c6bebe1fafcd01312400720782e647a5abaf2f50` |
| `src/llm_training/trainer.py` | `99fb7667a485c10dcbc09a622853416ecb67981dd5945cef30df1bb1234ba630` |
| `tests/gpu/multirank_case.py` | `ea194551a7721e05510e9e30bd330ae14ea1eeeda7921b508e456212fcd8a3d0` |
| `tests/gpu/multinode_validation.py` | `10115bf37550cf8b58e45c5f5ce98e350b0265615c9cd38becdb54a63c4bee19` |

Both nodes' runtime/worker hashes agree, and all archived Python source hashes
match the local source prepared for commit. Cloud cleanup status is delivered
separately; the existing results bucket is retained.

## Remaining test coverage

1. Same-host GPUs with local ranks 0/1 and different CUDA indices.
2. Hybrid LoRA plus full-trained base overrides under multi-GPU ZeRO-3.
3. Uneven data/accumulation boundaries and multi-epoch distributed resume.
4. Larger representative models, tied/untied embeddings, reentrant GC and more
   supported dependency combinations.
5. Multi-GPU CPU offload and additional optimizer/scheduler ownership combinations.
6. Node failure/restart and node-local checkpoint storage.
7. Long-run stability, peak memory and throughput.
8. Initialization-time ZeRO-3 partitioning requires implementation before validation.
