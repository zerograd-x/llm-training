# Test contracts and commands

## CPU

```bash
pip install -e '.[dev,peft]'
pip install -e ../llm-training-data
pytest -m 'not integration and not gpu'
pytest -m 'integration and not gpu' tests/integration
```

The optional companion package is needed for portable-workflow/data-bridge tests;
without it those tests skip explicitly. Vocabulary/manifest tests use a real HF
fast tokenizer and a tiny locally initialized GPT-2 model, without downloading
weights. A successful run establishes lifecycle correctness, not model quality.

New focused suites:

| Suite | Contract |
|---|---|
| `test_training_boundaries.py` | Partial accumulation matches a manual gradient reference; uneven three-epoch resume at four positions |
| `test_evaluation.py` | Global token NLL/perplexity, padding exclusion, custom statistics, RNG/mode/update/LR preservation |
| `test_checkpoint.py` | Model/optimizer/scheduler/RNG restore; missing, truncated and unfinished checkpoints rejected before load |
| `test_vocabulary_manifest.py` | Actual appended IDs, duplicate tokens, mean/random initialization, tied/untied old-row preservation, wrong-base/tamper rejection |
| `test_portable_workflow.py` | Full/LoRA/TrainableTokens raw-example workflow, actual collator, distinct training/merge/inference processes, explicit split validation |
| `integration/test_distributed_evaluation.py` | Globally reduced custom accuracy; mismatched statistic widths and rank-local callback exceptions propagate without hanging |
| `integration/test_hardening.py` | Two-rank Gloo boundaries, fresh-process resume, global evaluation and asymmetric failures |
| `integration/test_distributed_runtime.py` | Rank ownership, data sharding, scheduler horizon and a deterministic final-checkpoint metadata race |

The distributed hardening test uses Linux. The pinned Accelerate/PyTorch pair
selects an unsupported MPS barrier on macOS; that test explicitly skips there.
Use a CPU environment without DeepSpeed for Gloo tests. A CUDA DeepSpeed/Triton
installation with all CUDA devices hidden can fail during import before tests
reach the runtime. The separate DeepSpeed CPU suite has its own dependency setup.

## GPU

Two maintained GPU drivers cover the current single-GPU and two-node contracts.
The hardening driver uses two nodes with one GPU each and a shared filesystem:

```bash
# Run on both nodes, changing --node-rank to 1 on the second node.
python tests/gpu/hardening_validation.py /shared/new-run \
  --node-rank 0 --master PRIVATE_ADDRESS_OF_NODE_0
```

Positive cases are uneven DDP mid-epoch/epoch-boundary resume, full ZeRO-3 and
hybrid LoRA ZeRO-3. Each uses prepare/baseline/interrupt/resume/verify processes.
Nine examples, batch size two per rank and accumulation two produce three
microbatches and two updates per rank per epoch. Training pads three examples
across the job per epoch; evaluation counts exactly nine examples and 23 tokens.
Three epochs produce six optimizer updates. Baseline/resumed data sequence,
weights, logits, RNG restore, scheduler and evaluation histories are checked.
Full model or hybrid adapter/overrides are exported, merged when applicable,
reloaded and compared with the trained model.

Negative cases deliberately terminate rank 1, fail owner materialization,
remove a checkpoint runtime file, or inject fully masked labels on one rank.
A negative case passes only when both ranks record the expected failure and the
launcher exits unsuccessfully within its timeout. A timeout is always a failure.
These tests establish prompt failure propagation, not elastic restart/recovery.

Single-GPU portable workflow:

```bash
# Focused LoRA, TrainableTokens, scheduler and clean-offload lifecycle matrix.
python tests/gpu/gpu_validation.py /local/new-matrix-run

# Portable train/merge/generate workflow.
python tests/gpu/portable_validation.py /local/new-tiny-run
python tests/gpu/portable_validation.py /local/new-pretrained-run --pretrained
```

The first covers full/LoRA/TrainableTokens with BF16 and ZeRO-3; the second uses
SmolLM2-135M-Instruct with LoRA. Both use actual tokenizer/label masking, vocabulary
expansion, manifest checks, separate merge and fresh-process generation. The
pretrained fixture is a short integration run; generated exact match is recorded
without a quality acceptance threshold.

Use fresh output paths. GPU drivers explicitly assert CUDA/BF16 availability;
CPU fallback cannot count as a GPU pass. Configure a matching CUDA toolkit and
PyTorch build for JIT extensions. Never set `DS_SKIP_CUDA_CHECK` for clean offload
validation. Normal GitHub CI is CPU-only.

See the [consolidated GPU validation matrix](gpu-validation.md) for results,
versions, evidence and current limits.
