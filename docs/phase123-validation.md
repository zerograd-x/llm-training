# Training hardening, evaluation and portable workflow validation

Implementation starts from main `eaecf279f51b9cf2d5813334fd4dd9492d9ab7f7`.
Companion data source: `7589e4f6ae716451292f1714b09de7082a20103c`.

Two runtime fixes were required: incomplete accumulation normalization and single-process
model loading attempting a DeepSpeed barrier before its process group existed.
The workflow also launches its DeepSpeed training stage through a one-rank PyTorch
launcher, avoiding an implicit MPI discovery dependency.

The partial-accumulation numerical test reproduced an underweighted final window
before the fix (maximum parameter difference 0.01945 in its SGD fixture). Training
now divides by the actual number of microbatches in that window. The training
objective remains the mean of microbatch means; evaluation is global token NLL.

| Area | Current evidence |
|---|---|
| CPU unit/workflow | 94 tests passed, including checkpoint-damage and single-process runtime guards |
| Linux two-rank Gloo hardening | Six cases passed; additional global custom-metric/error-propagation test passed |
| Two-node GPU DDP / full ZeRO-3 / hybrid LoRA | All four lifecycle cases passed; two ZeRO-3 cases asserted actual cross-rank partitions |
| GPU asymmetric failures | Rank, materialization, checkpoint and masked-loss cases passed; rank failure propagated in 14.845 s |
| Single-GPU portable workflow | Full, LoRA and TrainableTokens BF16 + ZeRO-3 passed through generation |
| Pretrained tokenizer/model workflow | SmolLM2-135M-Instruct LoRA BF16 + ZeRO-3 passed through merge/reload/generation |

GPU runs use temporary g6.xlarge instances with NVIDIA L4, PyTorch 2.7.1+cu128,
CUDA toolkit 12.8, DeepSpeed 0.19.6, Accelerate 1.14.0, Transformers 5.16.1 and PEFT
0.20.0. Two instances fit the existing 8-vCPU quota. Shared checkpoint storage is
explicit. No test uses `DS_SKIP_CUDA_CHECK`.

The machine-readable matrix is [phase123-gpu-20260904.json](results/phase123-gpu-20260904.json).
All results are scoped to the recorded fixtures; exact-match generation had no
quality threshold. See
[tests](tests.md) for assertions and runnable drivers. Step 4 capabilities remain
[roadmap-only](roadmap.md): Muon, combined LoRA/TrainableTokens, performance,
initialization-time partitioning, MTP and tensor parallelism.

## Final GPU matrix

- Uneven DDP mid-epoch and epoch-boundary resume: passed.
- Full-parameter two-node ZeRO-3 checkpoint/export/reload: passed.
- Hybrid LoRA plus fully trained layer norm under two-node ZeRO-3, adapter/base
  override save, offline merge and reload: passed.
- Materialization, incomplete checkpoint and one-rank masked/non-finite loss:
  both ranks reported the same failure without timeout.
- Rank failure: the surviving rank's Gloo monitored barrier named rank 1 after
  10 seconds; both launchers exited nonzero in 14.845 seconds.
- Single-L4 full/LoRA/TrainableTokens BF16 + ZeRO-3 workflows: passed through
  validation, checkpoint, export/adapter, merge, manifest check and generation.
- SmolLM2-135M-Instruct LoRA BF16 + ZeRO-3: passed through the same fresh-process
  adapter/merge/generation lifecycle. Its four validation losses were recorded;
  the synthetic prompt's generation exact match was 0.0 and was not a quality gate.

The initial probes incorrectly observed PEFT's pre-wrap model and treated an
asynchronous NCCL operation as complete. Those harness failures were corrected
and rerun in fresh directories; they did not alter training implementation.
