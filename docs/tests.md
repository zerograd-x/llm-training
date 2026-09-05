# Test contracts and commands

## CPU

```bash
pip install -e '.[dev,peft]'
pip install -e ../llm-training-data
ACCELERATE_USE_CPU=true pytest -m 'not integration and not gpu'
ACCELERATE_USE_CPU=true pytest -m 'integration and not gpu' tests/integration
```

The companion package is required for portable-workflow/data-bridge tests.
Those tests may skip in a minimal local installation. The dedicated
`portable-workflow` CI job installs a pinned `llm-training-data` revision plus
PEFT and checks dependency imports before running the workflow and bridge suites;
missing dependencies fail that job. Other CPU jobs retain their separate HF,
PEFT, DeepSpeed and unit coverage.

Vocabulary/manifest tests use a real HF fast tokenizer and a tiny locally
initialized GPT-2 model. The portable suite runs actual data collators and
separate training, merge and inference processes without downloading weights.

| Suite | Contract |
|---|---|
| `test_training_boundaries.py` | Partial accumulation matches a manual gradient reference; uneven three-epoch resume at four positions |
| `test_evaluation.py` | Global token NLL/perplexity, padding exclusion, custom statistics, RNG/mode/update/LR preservation |
| `test_checkpoint.py` | Model/optimizer/scheduler/RNG restore; missing, truncated and unfinished checkpoints rejected before load |
| `test_vocabulary_manifest.py` | Appended IDs, duplicate tokens, mean/random initialization, tied/untied old-row preservation, wrong-base/tamper rejection |
| `test_portable_workflow.py` | Full/LoRA/TrainableTokens raw-example workflow, real collator, separate train/merge/inference, explicit split validation |
| `integration/test_training_data_bridge.py` | Companion-package dataset/collator integration and labels |
| `integration/test_distributed_evaluation.py` | Globally reduced custom accuracy; mismatched widths and rank-local callback exceptions propagate without hanging |
| `integration/test_hardening.py` | Two-rank Gloo boundaries, fresh-process resume, global evaluation and asymmetric failures |
| `integration/test_distributed_runtime.py` | Rank ownership, sharding, scheduler horizon and deterministic final-checkpoint metadata race |

Distributed hardening uses Linux. The pinned Accelerate/PyTorch pair selects an
unsupported MPS barrier on macOS, where that test explicitly skips. Use a CPU
environment without DeepSpeed for Gloo tests: a CUDA DeepSpeed/Triton install
with hidden GPUs can fail at import. The DeepSpeed CPU job has its own setup.

## GPU

Three maintained entry points cover single-GPU and two-node contracts. The
hardening driver uses two nodes with one GPU each and a shared filesystem:

```bash
# Run on both nodes, changing --node-rank to 1 on the second node.
python tests/gpu/hardening_validation.py /shared/new-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0

# Focused LoRA/TrainableTokens ZeRO-3 coverage, with and without non-reentrant GC.
python tests/gpu/hardening_validation.py /shared/new-peft-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0 \
  --cases lora_zero3 lora_zero3_gc tt_zero3 tt_zero3_gc
```

Positive cases include uneven DDP resume, full/hybrid ZeRO-3 and the four
LoRA/TrainableTokens cases above. Each uses prepare/baseline/interrupt/resume/
verify processes. Nine examples, batch size two per rank and accumulation two
produce three microbatches and two updates per rank per epoch. Training repeats
three examples across the job per epoch; evaluation counts exactly nine
examples and 23 tokens. Three epochs produce six updates. Baseline/resumed data,
weights, logits, RNG, scheduler and evaluation histories are compared; artifacts
are exported, merged when applicable and reloaded.

Negative cases terminate rank 1, fail owner materialization, remove a checkpoint
runtime file or inject fully masked labels on one rank. A pass requires both
ranks to record the expected failure and the launcher to fail within its
timeout. A timeout is a failure. These cases test prompt failure propagation;
automatic restart/recovery remains unimplemented.

Single-GPU entry points:

```bash
# LoRA, TrainableTokens, scheduler and clean-offload lifecycle matrix.
python tests/gpu/gpu_validation.py /local/new-matrix-run

# Portable train/merge/generate workflow with a local or pretrained fixture.
python tests/gpu/portable_validation.py /local/new-tiny-run
python tests/gpu/portable_validation.py /local/new-pretrained-run --pretrained
```

The portable driver's default fixture covers full/LoRA/TrainableTokens with
BF16 and ZeRO-3; `--pretrained` uses SmolLM2-135M-Instruct with LoRA. Both cover
tokenizer/label masking, vocabulary expansion, manifests, separate merge and
fresh-process generation. The short pretrained run records generation exact
match without a quality acceptance threshold. The focused driver has two
support modules, not additional runnable suites.

Use fresh output paths. The manual `gpu-smoke.yml` workflow requires CUDA/BF16
and runs the focused `lora_bf16` case, failing on a missing GPU or failed phase.
Push/PR CI is CPU-only. Configure matching CUDA/PyTorch for JIT extensions and
never set `DS_SKIP_CUDA_CHECK` for clean CPU offload validation.

See the [GPU matrix](gpu-validation.md) for historical passes, tested sources
and limits. Restored two-node cases are executable regression coverage; their
historical P1 passes are not a new GPU run of the maintained driver.
