# GPU validation

This matrix consolidates GPU runs completed on 2026-09-04. Results apply to the
recorded fixtures and sources; adding or restoring a driver does not mean its
current version has been rerun on GPU.

## Tested environment and source

NVIDIA L4 on `g6.xlarge`; PyTorch 2.7.1+cu128; CUDA toolkit 12.8;
NCCL 2.26.2; DeepSpeed 0.19.6; Accelerate 1.14.0; Transformers 5.16.1;
PEFT 0.20.0. `DS_SKIP_CUDA_CHECK` was absent.

| Evidence | Source and scope |
|---|---|
| P0 | Main `40d7fba1676a61017cc44819772987f60e0908da`; eight single-L4 cases, 40 fresh-process phases |
| P1 rerun | Same base plus the runtime model-loading-context fix; five two-node cases, 25 coordinated phases |
| Hardening/workflow | Phase123 changes on base `eaecf279f51b9cf2d5813334fd4dd9492d9ab7f7`; exact source retained with the evidence |
| Final checkpoint | Source `b92a43276dd9c2a9ea9780649926e1067f4e1e75`, merged into main as `7472aec8fdd9973b4bdd47c590f166c95986ad16`; NCCL race point test only |

All two-node runs used one L4 per host, global ranks 0/1, local rank 0 on both
hosts and shared checkpoint storage. They do not cover two local CUDA indices
on one host. Exact P1 source hashes and archive references are in the
[machine-readable summary](results/gpu-validation-20260904.json).

## Result matrix

| Scope | Case | Result | Evidence |
|---|---|---|---|
| One L4 | LoRA + BF16 | Pass | P0 |
| One L4 | LoRA + ZeRO-3 | Pass | P0 |
| One L4 | LoRA + gradient checkpointing | Pass | P0 |
| One L4 | LoRA + ZeRO-3 + gradient checkpointing | Pass | P0 |
| One L4 | LoRA checkpoint, fresh-process resume, adapter save, merge and reload | Pass | P0 |
| One L4 | TrainableTokens + BF16 | Pass | P0 |
| One L4 | TrainableTokens + ZeRO-3 through the public CPU-loaded model path | Pass | P0 |
| One L4 | Torch AdamW + HF scheduler + ZeRO-3 | Pass | P0 |
| One L4 | Clean CPUAdam offload + ZeRO-3, resume, export and reload | Pass | P0 |
| One L4 | Full/LoRA/TrainableTokens portable workflow | Pass | Hardening/workflow |
| One L4 | SmolLM2-135M-Instruct LoRA portable workflow | Pass | Hardening/workflow |
| Two nodes | DDP runtime checkpoint, resume, export and reload | Pass | P1 rerun |
| Two nodes | LoRA + ZeRO-3 lifecycle | Pass | P1 rerun |
| Two nodes | LoRA + ZeRO-3 + non-reentrant gradient checkpointing | Pass | P1 rerun |
| Two nodes | TrainableTokens + ZeRO-3 lifecycle | Pass | P1 rerun |
| Two nodes | TrainableTokens + ZeRO-3 + non-reentrant gradient checkpointing | Pass | P1 rerun |
| Two nodes | Uneven DDP resume in the middle of an epoch | Pass | Hardening/workflow |
| Two nodes | Uneven DDP resume at an epoch boundary | Pass | Hardening/workflow |
| Two nodes | Full-parameter ZeRO-3 checkpoint, export and reload | Pass | Hardening/workflow |
| Two nodes | Hybrid LoRA + fully trained layer norm under ZeRO-3 | Pass | Hardening/workflow |
| Two nodes | Rank, materialization, damaged-checkpoint and masked-loss failure propagation | Pass | Hardening/workflow |
| Two nodes | Final-checkpoint decision broadcast and collective save | Pass | Final checkpoint |

The P0/P1 lifecycle cases use fresh prepare, baseline, interrupt, resume and
verify processes. P1 asserts actual ZeRO-3 partitioning, exact optimizer
coverage, RNG/data-position restoration and final parameters matching each
rank's uninterrupted control. Adapter parameters are gathered collectively and
serialized only by global rank 0; offline reload/merge follows worker exit.
TrainableTokens preserves old effective rows exactly. The GC cases set
`use_reentrant=False` on tiny GPT-2.

The initial P1 attempt passed DDP but failed all four ZeRO-3 baselines before
training because implicit HF loading partitioned parameters before adaptation.
Only the successful rerun after the model-loading-context fix is marked Pass.
The maintained hardening driver restores these four cases on a newer, uneven
multi-epoch fixture; historical P1 results do not validate that new driver run.

The final checkpoint point test delays rank 1 if it inspects metadata. The
fixed implementation records `find_latest_checkpoint` calls `[1, 0]`: rank 0
makes and broadcasts the save decision, then both ranks enter the collective.
Both ranks completed two updates and wrote a complete manifest.

## Scheduler warning

The actual optimizer update precedes the scheduler; the observed LR sequence
is `0.01 -> 0.005 -> 0`. DeepSpeed CPU offload substitutes CPUAdam behind
Accelerate's wrapper while PyTorch's scheduler observes the original AdamW
object. That can produce an optimizer-step warning despite correct execution.

A separate resume defect was real: loading CPUAdam replaced parameter-group
objects and disconnected the HF scheduler. Trainer now rebinds the scheduler
to the prepared optimizer before state restoration. Uninterrupted and resumed
LR, parameters and exported logits match.

## Maintained GPU drivers

Three entry points and two single-GPU support modules remain. Use fresh output
directories; GPU drivers reject CPU fallback.

```bash
# Run on both nodes; change --node-rank to 1 on the second node.
python tests/gpu/hardening_validation.py /shared/new-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0

# Focus only on restored LoRA/TrainableTokens ZeRO-3 cases, on both nodes.
python tests/gpu/hardening_validation.py /shared/new-peft-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0 \
  --cases lora_zero3 lora_zero3_gc tt_zero3 tt_zero3_gc

# One L4: lifecycle, scheduler and clean CPUAdam offload matrix.
python tests/gpu/gpu_validation.py /local/new-matrix-run

# One L4: portable workflow, local tiny or downloaded SmolLM2 fixture.
python tests/gpu/portable_validation.py /local/new-tiny-run
python tests/gpu/portable_validation.py /local/new-pretrained-run --pretrained
```

The manual GPU smoke workflow runs `gpu_validation.py --cases lora_bf16`
after requiring CUDA and BF16. Full matrices need explicitly provisioned GPU
hosts. CPUAdam validation requires a CUDA toolkit matching the PyTorch build.
See [test commands](tests.md) for CPU CI and distributed contracts.

## Evidence and limits

The [JSON summary](results/gpu-validation-20260904.json) associates every row
with its dated source. P0/P1 archive SHA256:
`6e74c078b42afec41c3c18fd0a94c7ee3d8ea8fae763c36842d5136acbcdb432`.
The phase123 archive retains logs, artifacts, versions and exact tested source.

The fixtures establish lifecycle correctness. Same-host multi-GPU operation,
distributed CPU offload, broader architectures/dependencies/GC modes, model
quality thresholds, peak memory, throughput and long-run stability remain
untested. Initialization-time ZeRO-3 partitioning, Muon, combined LoRA +
TrainableTokens and automatic restart/recovery require implementation.
