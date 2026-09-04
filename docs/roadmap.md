# Core training roadmap

This roadmap tracks reusable capabilities in `llm-training`. Application-specific
data systems, orchestration frameworks, serving engines, retrieval logic, and
storage transports stay outside this repository.

The [portable workflow design](train_causal_lm_workflow.md) defines a plain
Python/local-filesystem path through data preparation, optional vocabulary
expansion, training, export and optional inference/evaluation. No organization
platform, remote scheduler or storage service is required by that design.

## Current status

```text
Foundation                                      ✅
Phase 1  PEFT adaptation foundation             ✅
Phase 2  LoRA                                   ✅
Phase 3  TrainableTokens                        ✅ CPU / single + two-node GPU
Phase 4  adapter + offline merge lifecycle      ✅ CPU / single + two-node GPU
Phase 5  PEFT + distributed hardening           🟡 boundary/failure suite implemented; validation tracked below
Phase 6  Muon / richer optimizers               ❌
Phase 7  evaluation framework                   ✅ first implementation
Phase 8  advanced distributed / observability   ❌
```

Validation is scoped to the documented fixtures and dependency stack. See the
[P0/P1 GPU matrix](gpu-p0-p1-validation.md) for single-L4 and two-node evidence;
this is not blanket validation of every model, topology or configuration.

---

## Foundation — complete

The runtime currently provides:

- Hugging Face causal-LM loading and ordinary full-model export
- transformer-layer, embedding, LM-head, and tied-weight introspection
- selective base-parameter freezing through `FreezeConfig`
- configurable causal-LM loss with external next-token shifting
- FP32 default cross-entropy reduction for lower-precision logits
- AdamW and Hugging Face schedulers
- gradient accumulation and clipping
- deterministic epoch-seeded sampling
- training-state checkpoint/resume including optimizer, scheduler, RNG, and
  logical trainer position
- optional model-native gradient checkpointing
- Accelerate + optional DeepSpeed ZeRO-1/2/3 configuration
- explicit Torch-owned vs DeepSpeed-owned optimizer/scheduler contracts
- CPU, Hugging Face, PEFT, and real DeepSpeed CPU integration tests

ZeRO-3 currently partitions after `accelerator.prepare(...)`; model loading does
not yet happen inside a ZeRO-3 initialization context.

---

## Phase 1 — PEFT adaptation foundation — complete

PEFT transformations run before optimizer construction so newly created adapter
parameters belong to the optimizer/backend that owns training.

The central distinction is:

```text
FreezeConfig
    -> which base parameters are fully trained

PEFT method config
    -> which adapter parameterization is applied
```

For hybrid LoRA, the runtime records the intended full-trained base parameters
before PEFT freezes the base, then restores those exact parameter objects after
wrapping.

The current lifecycle is:

```text
base HF model
    ↓
optional base full-train policy
    ↓
PEFT adaptation
    ↓
restore explicit hybrid base trainables when applicable
    ↓
adapter-aware gradient checkpointing
    ↓
optimizer / scheduler construction
    ↓
accelerator.prepare(...)
```

---

## Phase 2 — LoRA — complete for current CPU contract

Implemented:

- `LoRAConfig`
- rank / alpha / dropout / bias configuration
- conservative transformer-block target discovery
- fully-qualified target module names
- optional `target_last_n_layers`
- adapter-only LoRA (`freeze=None`)
- hybrid LoRA (`FreezeConfig + LoRA`)
- adapter-aware gradient checkpointing
- Torch and real DeepSpeed CPU training/checkpoint/resume
- pure and hybrid adapter artifact export
- offline merge into a plain Hugging Face model
- numerical equivalence tests between trained PEFT and merged model

### Independent adaptation and freezing policies

LoRA placement is intentionally **not** derived from
`FreezeConfig.train_last_n_layers`. `FreezeConfig` controls full-trained base
parameters; `LoRAConfig.target_last_n_layers` controls adapter placement. The
same model can therefore, for example, place LoRA on all blocks while fully
training only selected base parameters.

Default LoRA discovery is restricted to transformer-block projections. Embedding
or LM-head adaptation is not implicitly inferred from freeze settings.

---

## Phase 3 — TrainableTokens — complete for current CPU contract

TrainableTokens is implemented with PEFT's native `TrainableTokensConfig`; the
runtime does not make the full embedding matrix trainable and does not implement
manual gradient masking.

Current public configuration:

```python
TrainableTokensConfig(n_tokens_added=N)
```

The model must already contain the vocabulary expansion. The final N rows of the
current input-embedding vocabulary are selected as trainable token ids.

Implemented:

- final-N appended-token selection
- target discovery before PEFT wrapping
- tied embedding / LM-head handling
- untied input/output embedding handling
- PEFT `trainable_tokens_delta` ownership
- gradient-checkpointing input-grad setup
- compact adapter serialization
- explicit `save_embedding_layers=False` so a vocab resize does not cause PEFT
  to serialize the full embedding matrix
- offline merge into the exact vocabulary-expanded base
- old-row bit-identity check after merge
- new-row update check
- merged-logit numerical equivalence
- real DeepSpeed CPU checkpoint/resume/save/offline-merge lifecycle

Standalone TrainableTokens v1 deliberately rejects:

```text
LoRA + TrainableTokens
FreezeConfig + TrainableTokens
```

Combined LoRA + partial-token training requires an explicit adaptation and
artifact contract. Sequentially wrapping standalone adapters can replace an
existing adapter, so unsupported combinations fail before training. A combined
implementation can use PEFT's native `trainable_token_indices` mechanism once
its optimizer, checkpoint and merge behavior is independently tested.

Presence of `TrainableTokensConfig` explicitly requests partial-token training;
`n_tokens_added <= 0` is a configuration error.

See `docs/trainable_tokens.md` for the complete adaptation contract.

---

## Phase 4 — adapter-aware artifact lifecycle — complete for current CPU contract

Training-state checkpoint, adapter artifact, and deployment model are separate
concepts.

```text
training-state checkpoint
    -> resume optimizer/scheduler/RNG/trainer state

PEFT adapter artifact
    -> small portable training result

merged model
    -> ordinary Hugging Face checkpoint for deployment
```

### LoRA

Pure LoRA saves standard PEFT adapter state. Hybrid LoRA additionally writes:

```text
base_trainable_state.safetensors
```

with original Hugging Face parameter names for base tensors intentionally
full-trained through `FreezeConfig`.

Offline merge order for hybrid LoRA is:

```text
load exact base
    ↓
restore base_trainable_state
    ↓
attach LoRA adapter
    ↓
merge_and_unload()
    ↓
plain HF model
```

### TrainableTokens

TrainableTokens saves only PEFT partial-token state and merges offline into the
same vocabulary-expanded base used for training.

### Distributed-memory contract

Adapter save paths do not request a consolidated full-model state dict. For
configured ZeRO-3 they enter `deepspeed.zero.GatheredParameters` only around
adapter parameters and, for hybrid LoRA, the additional full-trained base
overrides.

Adapter-only LoRA and TrainableTokens gather/save are validated on two L4
nodes with real ZeRO-3 partitions and shared checkpoint storage. Hybrid LoRA
base-override gather/save remains pending on real multi-GPU hardware.

These APIs accept ordinary model and adapter paths. Optional artifact transfer
is supplied by the caller and is independent of training.

---

## Phase 5 — PEFT + DeepSpeed hardening — partial

Already validated:

```text
real DeepSpeed CPU engine                         ✅
DeepSpeed-owned optimizer + scheduler             ✅
LoRA checkpoint / fresh-process resume            ✅
hybrid LoRA checkpoint / fresh-process resume     ✅
TrainableTokens checkpoint / fresh-process resume ✅
adapter save after resume                         ✅
offline merged-model equivalence                  ✅
```

The September 4 P1 run additionally validates two-node NCCL, actual ZeRO-3
partitions, adapter-only gathering and rank-0 serialization, LoRA/TrainableTokens
with non-reentrant gradient checkpointing, distributed checkpoint/resume and
offline merge/reload. Both ranks restore RNG/data position and match their
uninterrupted baselines exactly. See [P0/P1 evidence](gpu-p0-p1-validation.md).

Remaining distributed coverage:

- two different local CUDA indices on one multi-GPU host
- hybrid LoRA with full-trained base overrides under ZeRO-3
- uneven batches/accumulation and multi-epoch distributed resume
- larger representative models, tied/untied embeddings and alternative GC modes
- distributed offload and additional optimizer/scheduler ownership combinations
- node failure, restart and node-local checkpoint storage
- performance, peak memory and longer-run numerical stability

Non-reentrant GC passes the pinned tiny-GPT-2 two-node tests; this does not
resolve compatibility for every architecture or dependency combination.

A second future item is ZeRO-3 initialization-time partitioning. The current
lifecycle materializes the complete HF model before the backend is prepared.
This also means current tied-weight discovery can safely use actual parameter
identity. If init-time partitioning is added, tied-module discovery should be
revisited using PEFT/Transformers model metadata rather than assuming tensor
identity is already materialized.

---

## Phase 6 — Muon and richer optimizers — deferred

Muon support is not just accepting `type="Muon"`.

The intended policy is approximately:

```text
eligible hidden 2D trainable matrices -> Muon
embedding / LM head                    -> companion AdamW
norm / bias / vectors / scalars        -> companion AdamW
other ineligible trainables            -> companion AdamW
```

Requirements:

- explicit parameter classification before optimizer initialization
- every trainable parameter belongs to exactly one optimizer policy
- DeepSpeed-native metadata/tagging when using DeepSpeed Muon
- version requirements documented and validated
- checkpoint/resume tests
- PEFT/TrainableTokens behavior defined rather than inferred accidentally from
  tensor rank
- Torch-owned Muon only when a stable implementation/API is selected

Generic AdamW must remain independent of Muon-specific tensor-shape rules.

---

## Phase 7 — evaluation framework — initial implementation

Implemented optional validation data, epoch/step schedules, global token-weighted
NLL/perplexity and an additive custom-metric interface. Explicit masks exclude
samples repeated only to balance evaluation ranks. Evaluation preserves training
RNG/mode/optimizer/scheduler state and records its position in checkpoints.

See [evaluation](evaluation.md) for definitions, extension points and limitations.
Task-specific classification/ranking/judge metrics remain application plugins.

---

## Phase 8 — advanced distributed and observability

Later work includes:

- ZeRO-3 initialization-time partitioning during model construction
- stage-3 bucket/prefetch/persistence tuning when justified by workloads
- deferred Torch/HF schedulers with DeepSpeed-owned optimizers if a stable
  ownership contract is implemented
- PyTorch profiler integration
- step time, tokens/sec, memory, and optional FLOP reporting
- backend-neutral logging hooks
- tensor parallelism after PEFT/checkpoint/evaluation contracts are stable

---

## Single-GPU validation — 2026-09-04

All eight single-L4 cases pass, including 40 fresh-process phases and 70 CPU
regressions. See the [GPU matrix](gpu-validation-matrix.md) for exact versions,
assertions, evidence and limits. LoRA BF16/ZeRO-3/checkpointing, TrainableTokens
BF16/ZeRO-3, advancing resume, adapter save/merge/reload, scheduler tracing and
clean CPUAdam offload/resume/export are validated on tiny local models.

This validation repaired TrainableTokens CPU/NCCL initialization, resumed CPU
RNG consumption, resumed/subsequent epoch sampling and offload scheduler binding
after optimizer restoration. One documented PyTorch optimizer-wrapper warning
remains, with actual step ordering and LR verified. Init-time partitioning and
real multi-rank GPU behavior were not validated in that single-card round.
The subsequent [P0/P1 round](gpu-p0-p1-validation.md) reran all eight single-GPU
cases on main and passed five two-node GPU cases after fixing the runtime
model-loading context. The 8-vCPU quota permitted two g6.xlarge nodes, each
with one L4; initialization-time partitioning remains unsupported.

## Current delivery and deferred work

The approved implementation order is:

1. Training hardening: correct partial accumulation windows; three-epoch uneven
   dataset resume; rank/materialization/checkpoint/non-finite-loss failures;
   hybrid LoRA and full-model ZeRO-3 artifact checks.
2. Evaluation: scheduled global NLL/perplexity, additive metrics and no training
   state/RNG changes.
3. Portable workflow: real data-package collator, vocabulary expansion with
   actual IDs, artifact manifests, separate train/merge/generation processes.

Implementation and test evidence are tracked in [phase 1–3 validation](phase123-validation.md).
A test being implemented does not by itself establish GPU support.

The following **step 4 items are roadmap only**, with no implementation added in
this delivery:

- **Muon / richer optimizers:** explicit matrix/embedding/norm parameter policies,
  ownership and checkpoint contracts before integrating a selected backend.
- **Combined LoRA + TrainableTokens:** one native combined adaptation contract,
  exact trainable ownership and adapter/resume/merge tests. Current combinations
  continue to fail early.
- **Performance / stability:** tokens/sec, step time, peak memory, longer runs,
  representative architectures, offload/topology comparisons and profiling.
- **ZeRO-3 initialization-time partitioning:** load large models without a full
  replica per rank; revisit tied-weight discovery and pre-adaptation metadata.
- **Advanced model/distribution features:** MTP and tensor parallelism after
  checkpoint/evaluation/artifact semantics are stable.

Remaining validation beyond the current fixtures also includes two different
local CUDA indices on one host, node restart/recovery and node-local checkpoints,
alternative gradient-checkpointing modes, and broader dependency versions.
Multi-GPU jobs require explicitly provisioned hardware; push/PR CI stays CPU-only.

---

## Design constraints

These rules should remain true as the project grows:

- Core APIs stay general and backend-independent where practical.
- Data engines, orchestration, serving, retrieval, and storage transports stay
  outside the runtime.
- Torch ownership is the default; backend ownership is explicit.
- A hyperparameter has one source of truth.
- `FreezeConfig` means full-training policy for base parameters; PEFT placement
  has its own configuration.
- Trainable parameters must exist before optimizer/backend ownership is
  established.
- Unsupported adaptation combinations fail early rather than silently changing
  training semantics.
- Training checkpoints, adapter artifacts, and merged deployment models are
  distinct artifacts.
- Adapter-only save must not gather a full model merely for convenience.
- Optimizer-specific grouping rules are named, explicit, and tested.
- Distributed metrics must represent all participating ranks.
- Features are not marked multi-GPU/ZeRO-3 supported until real multi-rank tests
  cover training, checkpoint/resume, and export.

