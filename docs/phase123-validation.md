# Training hardening, evaluation and portable workflow validation

Implementation starts from main `eaecf279f51b9cf2d5813334fd4dd9492d9ab7f7`.
Companion data source: `7589e4f6ae716451292f1714b09de7082a20103c`.

The partial-accumulation numerical test reproduced an underweighted final window
before the fix (maximum parameter difference 0.01945 in its SGD fixture). Training
now divides by the actual number of microbatches in that window. The training
objective remains the mean of microbatch means; evaluation is global token NLL.

| Area | Current evidence |
|---|---|
| CPU unit/workflow | 90 tests passed; 3 additional checkpoint-damage variants passed in a 9-test focused rerun |
| Linux two-rank Gloo hardening | Validation in progress |
| Two-node GPU DDP / full ZeRO-3 / hybrid LoRA | Validation in progress |
| GPU asymmetric failures | Validation in progress |
| Single-GPU portable workflow | Validation in progress |
| Pretrained tokenizer/model workflow | Validation in progress |

GPU runs use temporary g6.xlarge instances with NVIDIA L4, PyTorch 2.7.1+cu128,
CUDA toolkit 12.8, DeepSpeed 0.19.6, Accelerate 1.14.0, Transformers 5.16.1 and PEFT
0.20.0. Two instances fit the existing 8-vCPU quota. Shared checkpoint storage is
explicit. No test uses `DS_SKIP_CUDA_CHECK`.

This page will record completed GPU results before the change is merged. See
[tests](tests.md) for assertions and runnable drivers. Step 4 capabilities remain
[roadmap-only](roadmap.md): Muon, combined LoRA/TrainableTokens, performance,
initialization-time partitioning, MTP and tensor parallelism.
