# GPU validation

This is the single source of truth for the GPU validation completed on
2026-09-04. Historical audit notes and one-off launch workers were consolidated
after the final run.

## Tested environment

- NVIDIA L4 on `g6.xlarge`
- PyTorch 2.7.1+cu128 and CUDA toolkit 12.8
- NCCL 2.26.2
- DeepSpeed 0.19.6
- Accelerate 1.14.0
- Transformers 5.16.1
- PEFT 0.20.0
- `DS_SKIP_CUDA_CHECK` absent

The final source is main merge commit
`7472aec8fdd9973b4bdd47c590f166c95986ad16`. The final-checkpoint NCCL point
test used source tree `b92a43276dd9c2a9ea9780649926e1067f4e1e75`, which is
the tree merged by that commit.

## Result matrix

| Scope | Case | Result |
|---|---|---|
| One L4 | LoRA + BF16 | Pass |
| One L4 | LoRA + ZeRO-3 | Pass |
| One L4 | LoRA + gradient checkpointing | Pass |
| One L4 | LoRA + ZeRO-3 + gradient checkpointing | Pass |
| One L4 | LoRA checkpoint, fresh-process resume, adapter save, merge and reload | Pass |
| One L4 | TrainableTokens + BF16 | Pass |
| One L4 | TrainableTokens + ZeRO-3 through the public CPU-loaded model path | Pass |
| One L4 | Torch AdamW + HF scheduler + ZeRO-3 | Pass |
| One L4 | Clean CPUAdam offload + ZeRO-3, resume, export and reload | Pass |
| One L4 | Full/LoRA/TrainableTokens portable workflow | Pass |
| One L4 | SmolLM2-135M-Instruct LoRA portable workflow | Pass |
| Two nodes, one L4 each | Uneven DDP resume in the middle of an epoch | Pass |
| Two nodes, one L4 each | Uneven DDP resume at an epoch boundary | Pass |
| Two nodes, one L4 each | Full-parameter ZeRO-3 checkpoint, export and reload | Pass |
| Two nodes, one L4 each | Hybrid LoRA + fully trained layer norm under ZeRO-3 | Pass |
| Two nodes, one L4 each | Rank, materialization, damaged-checkpoint and masked-loss failure propagation | Pass |
| Two nodes, one L4 each | Final-checkpoint decision broadcast and collective save | Pass |

The single-GPU lifecycle cases used fresh processes for interruption, resume,
adapter or model export, merge and reload. They verify trainable-parameter
ownership, optimizer and scheduler position, RNG and data position, exact final
parameters against uninterrupted controls, and artifact reload tolerances.

The two-node ZeRO-3 cases assert actual cross-rank partitioning. The final
checkpoint point test amplifies the old race by delaying rank 1 if it inspects
metadata. The fixed implementation records `find_latest_checkpoint` calls
`[1, 0]`: rank 0 decides whether a final save is needed, broadcasts that
decision, and both ranks enter the save collective. Both ranks completed two
updates and wrote a complete manifest.

## Scheduler warning

The optimizer/scheduler order is correct: the actual optimizer update happens
before the scheduler advances, and the observed LR sequence is
`0.01 -> 0.005 -> 0`. DeepSpeed CPU offload substitutes CPUAdam behind
Accelerate's wrapper, while PyTorch's scheduler still observes the original
AdamW object. That can produce an optimizer-step warning even though execution
order is correct.

A separate resume defect was real: loading CPUAdam replaced its parameter-group
objects and disconnected the HF scheduler. Trainer now rebinds the scheduler to
the prepared optimizer before state restoration. Uninterrupted and resumed LR,
parameters and exported logits now match.

## Retained GPU drivers

Three manual entry points remain:

```bash
# Run on both nodes; use --node-rank 1 on the second node.
python tests/gpu/hardening_validation.py /shared/new-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0

# One L4, focused lifecycle/scheduler/offload matrix.
python tests/gpu/gpu_validation.py /local/new-matrix-run

# One L4, local tiny model.
python tests/gpu/portable_validation.py /local/new-tiny-run

# One L4, downloaded SmolLM2 fixture.
python tests/gpu/portable_validation.py /local/new-pretrained-run --pretrained
```

`gpu_validation.py` covers the focused LoRA, TrainableTokens, scheduler and
clean CPUAdam lifecycles. `hardening_validation.py` covers distributed resume,
ZeRO-3 artifacts and asymmetric failures. `portable_validation.py` covers full,
LoRA and TrainableTokens workflows with BF16, ZeRO-3, validation, manifests,
merge or export, reload and generation. The focused driver has two support
modules in the same directory; they are not separate entry points.

Use a fresh output directory. GPU drivers reject CPU fallback. CPUAdam validation
requires a CUDA toolkit matching the PyTorch CUDA build.

## Evidence and limits

The machine-readable summary is
[`results/gpu-validation-20260904.json`](results/gpu-validation-20260904.json).
The retained external evidence archive contains phase logs, checkpoints,
adapters, merged models, environment versions and exact tested source.

These runs validate lifecycle correctness on the recorded fixtures. They do not
establish model quality, maximum model size, peak throughput or long-run
stability. ZeRO-3 initialization-time partitioning, Muon, combined LoRA +
TrainableTokens, performance profiling and restart recovery remain roadmap work.
