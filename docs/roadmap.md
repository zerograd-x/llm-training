# Core training roadmap

This roadmap tracks reusable capabilities in `llm-training`. Application-specific
data systems, orchestration frameworks, serving engines, retrieval logic, and
storage transports stay outside this repository.

## Current status

```text
Foundation                                      ✅
Phase 1  PEFT adaptation foundation             ✅
Phase 2  LoRA                                   ✅
Phase 3  TrainableTokens                        ✅ CPU / DeepSpeed CPU
Phase 4  adapter + offline merge lifecycle      ✅ CPU / DeepSpeed CPU
Phase 5  PEFT + distributed hardening           🟡 multi-rank ZeRO-3 pending
Phase 6  Muon / richer optimizers               ❌
Phase 7  evaluation framework                   ❌
Phase 8  advanced distributed / observability   ❌
```

“Done” above means the feature's implemented CPU paths have integration tests.
It does not imply that a multi-rank GPU/ZeRO-3 path has been validated.

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

### Important design difference from the older reference

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

The older reference had a dangerous LoRA+TrainableTokens path where sequential
PEFT wrapping could replace the previously created LoRA model. This runtime
fails early instead. Current PEFT has a native LoRA
`trainable_token_indices` mechanism; combined LoRA + partial-token training can
be added later as an explicit, independently tested feature.

The reference also allowed “enabled + zero added tokens” to warn and fall back.
Here, presence of `TrainableTokensConfig` means the user explicitly requested
that training regime, so `n_tokens_added <= 0` is a configuration error.

See `docs/trainable_tokens.md` for the detailed contract and reference
comparison.

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

This is the intended multi-rank design, but real multi-rank ZeRO-3 artifact
saving remains Phase 5 validation work.

Storage-specific concepts such as HDFS/S3 upload paths do not appear in these
APIs. Publishing remains the responsibility of the artifact-transfer layer.

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

Still required before claiming distributed ZeRO-3 support:

- 2+ GPU multi-rank ZeRO-3 training
- actual adapter parameter partitioning behavior
- NCCL reduce-scatter/all-gather lifecycle
- adapter-only gather/save from sharded parameters
- hybrid base-override gather/save
- TrainableTokens under real ZeRO-3
- gradient-checkpoint recomputation behavior with partitioned adapter/base
  tensors
- distributed checkpoint/resume and offline merge

The older reference forced reentrant gradient checkpointing for its ZeRO-3
environment because non-reentrant recomputation checks observed partitioned
parameter shape changes. This runtime keeps `use_reentrant` explicit until that
behavior is reproduced and validated on the supported multi-rank stack.

A second future item is ZeRO-3 initialization-time partitioning. The current
lifecycle materializes the complete HF model before the backend is prepared.
This also means current tied-weight discovery can safely use actual parameter
identity. If init-time partitioning is added, tied-module discovery should be
revisited using PEFT/Transformers model metadata rather than assuming tensor
identity is already materialized.

---

## Phase 6 — Muon and richer optimizers — next major feature

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

## Phase 7 — evaluation framework

Planned:

- optional validation dataset/dataloader
- epoch and optional step-based evaluation
- validation loss / perplexity where meaningful
- pluggable metrics
- `accelerator.gather_for_metrics` or equivalent distributed aggregation
- no rank-local metric published under a global metric name
- evaluation state separated from optimization state

Application-specific classification/ranking/judge metrics remain plugins rather
than core causal-LM assumptions.

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
real multi-rank GPU behavior remain unvalidated. Quota is now 8 vCPUs; no two-GPU
run was performed in this single-card round.

## Priority from here

```text
1. real multi-rank PEFT / ZeRO-3 validation when GPU infrastructure is used
2. Muon and richer optimizer policies
3. evaluation framework
4. ZeRO-3 init-time loading / profiling / advanced distributed features
5. optional combined LoRA + trainable-token adapter support
```

Multi-GPU validation should be run explicitly on dedicated hardware; normal
push/PR CI remains CPU-only.

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

