# Core training roadmap

This roadmap tracks capabilities that belong in the reusable `llm-training` runtime itself. The goal is a general causal-LM training stack with explicit contracts for model adaptation, optimization, distributed execution, checkpointing, and evaluation.

Application-specific data systems, serving stacks, retrieval logic, domain wrappers, and orchestration frameworks remain outside this repository.

## Current foundation

The current runtime already provides a strong base:

- Hugging Face causal-LM loading and export
- model-structure introspection for transformer layers, embeddings, LM heads, and tied weights
- selective freezing through `FreezeConfig`
- configurable causal-LM loss with runtime injection and dotted-path loading
- causal next-token alignment outside the loss implementation
- FP32 cross-entropy reduction for BF16 logits
- finite-loss validation before backward
- AdamW and Hugging Face schedulers
- checkpoint/resume with model, optimizer, scheduler, RNG state, and logical trainer position
- deterministic epoch-seeded sampling for resumable training
- gradient accumulation and gradient clipping
- model-native gradient checkpointing
- optional DeepSpeed ZeRO-1/2/3 backend
- CPU/NVMe optimizer and parameter offload configuration where supported by ZeRO stage
- explicit DeepSpeed-owned optimizer and scheduler through `DummyOptim` / `DummyScheduler`
- Torch-owned optimizer + Torch/Hugging Face scheduler as the default path
- Torch-owned optimizer + DeepSpeed-owned scheduler support
- DeepSpeed-owned optimizer + DeepSpeed-owned scheduler support
- ZeRO-3-aware full-model export through `accelerator.get_state_dict(...)`
- CPU integration tests, Hugging Face integration tests, resume tests, and an opt-in GPU smoke path

The largest remaining gaps are no longer in the basic training loop. They are concentrated in model adaptation, optimizer specialization, evaluation, and more advanced distributed lifecycle handling.

---

## Phase 1: PEFT foundation

PEFT is the highest-priority next step because it changes which parameters exist, which parameters are trainable, and how training state and exported artifacts must be interpreted.

### Scope

- optional `peft` dependency
- general PEFT configuration owned by `llm-training`
- a model-adaptation layer that runs after base freeze policy is established and before optimizer construction
- explicit interaction between `FreezeConfig` and adapter configuration
- helpers that operate on the existing model-introspection APIs rather than hard-coded model-family branches
- clear accounting of trainable versus frozen parameters after adaptation
- early validation of unsupported or contradictory combinations

### Design contract

`FreezeConfig` remains the source of truth for **what should be trainable**. PEFT decides **how selected parameters are trained**.

The intended ordering is:

```text
load base model
    ↓
apply FreezeConfig
    ↓
apply PEFT transformation
    ↓
restore any intentionally full-train parameters hidden by PEFT wrapping
    ↓
configure gradient checkpointing
    ↓
build optimizer
    ↓
accelerator.prepare(...)
```

### Exit criteria

- PEFT can be enabled without changing the default full-fine-tuning path.
- Trainable parameter ownership is deterministic and inspectable.
- Optimizer construction only sees parameters that are intentionally trainable.
- The feature is covered by unit tests before LoRA-specific behavior is layered on top.

---

## Phase 2: LoRA

LoRA is the first concrete PEFT method to support.

### Scope

- general `LoRAConfig`
- configurable rank, alpha, dropout, bias behavior, and adapter name where useful
- target-module discovery based on model structure
- support for limiting LoRA to the subset of transformer layers selected by `FreezeConfig.train_last_n_layers`
- fully-qualified target-module names when necessary to avoid accidentally adapting every layer with the same leaf module name
- predictable handling of attention and MLP projection modules
- correct handling of embeddings and LM heads
- tied-weight-safe target selection so the same underlying tensor is not wrapped twice
- support for intentionally full-training selected non-LoRA parameters such as an LM head or embedding table
- re-application of those full-train decisions after `get_peft_model(...)` if PEFT freezes the base model
- trainable-parameter summary after wrapping

### Gradient checkpointing requirements

Adapter training changes the gradient path. When the base model is frozen, gradient checkpointing may require input tensors to participate in autograd so gradients reach adapter modules.

The LoRA path must therefore handle:

- `enable_input_require_grads()` where required by the model/PEFT combination
- `model.config.use_cache = False`
- model-native `gradient_checkpointing_enable(...)`
- compatibility with ZeRO-3 checkpoint recomputation behavior

### Exit criteria

- LoRA trains on at least one decoder-only Hugging Face model in CPU integration tests where practical.
- A real GPU smoke test demonstrates parameter updates only in the expected trainable set.
- LoRA + gradient checkpointing works.
- LoRA + ZeRO-3 is tested on the distributed path when multi-GPU integration infrastructure is available.
- Full-fine-tuning behavior remains unchanged when LoRA is disabled.

---

## Phase 3: TrainableTokens / vocabulary-extension training

Supporting newly added vocabulary rows is a separate general capability from LoRA and should be treated explicitly.

### Scope

- optional training of only newly added token rows
- explicit `n_tokens_added` or equivalent runtime metadata
- validation that the tokenizer/model vocabulary expansion is consistent
- support for tied and untied embedding/LM-head configurations
- target discovery from the unwrapped base model before PEFT wrapping
- prevention of double-wrapping tied embedding and LM-head tensors
- correct interaction with `FreezeConfig`
- adapter-compatible gradient checkpointing
- clear behavior when the feature is enabled but no tokens were added

### Exit criteria

- only the intended new token rows change after training
- existing vocabulary rows remain unchanged within numerical tolerance
- tied embedding/LM-head models are covered by tests
- checkpoint/resume preserves the trainable-token state

---

## Phase 4: adapter-aware checkpoint and export lifecycle

Full-model export and adapter export have different memory and portability requirements, especially under ZeRO-3.

### Scope

- adapter-only checkpoints for LoRA and TrainableTokens
- avoid full-model ZeRO-3 consolidation when only adapter state is needed
- gather only trainable adapter parameters when necessary
- save standard PEFT artifacts where applicable
- preserve tokenizer and training metadata alongside adapter artifacts
- optional offline merge into a full Hugging Face checkpoint
- merged-model export that produces a normal model consumable without PEFT runtime dependencies
- explicit distinction between:
  - training-state checkpoints used for resume
  - adapter artifacts used for deployment or sharing
  - merged full-model artifacts
- configurable `max_shard_size` for full-model export
- release large consolidated state dictionaries as early as practical during export to reduce host-memory pressure

### Exit criteria

- LoRA checkpoint/resume works without forcing a full-model export.
- Adapter-only export works under the normal backend and ZeRO-3.
- Optional merge produces numerically equivalent outputs to the unmerged base+adapter model within tolerance.

---

## Phase 5: PEFT + DeepSpeed integration hardening

After LoRA and TrainableTokens exist independently, their interaction with distributed ownership must be validated explicitly.

### Scope

- LoRA + ZeRO-1/2/3
- TrainableTokens + ZeRO-3
- adapter-only checkpointing under parameter partitioning
- resume from distributed training-state checkpoints
- gradient checkpointing under frozen-base adapter training
- optimizer ownership checks for both Torch-owned and DeepSpeed-owned optimizers
- no trainable adapter parameter may exist outside the backend's optimization/checkpoint ownership

### Exit criteria

A multi-GPU integration matrix exists for the supported combinations, and unsupported combinations fail before expensive training starts.

---

## Phase 6: richer optimizers, starting with Muon

The optimizer layer should support more than AdamW while preserving explicit ownership and parameter-group semantics.

### General optimizer work

- optimizer selection in the Torch-owned path
- DeepSpeed-native optimizer selection in the DeepSpeed-owned path
- optimizer-specific configuration without duplicating common hyperparameters across multiple config trees
- optimizer state checkpoint/resume tests
- explicit parameter grouping APIs where an optimizer applies different update rules to different parameter classes

### Muon

Muon requires more than accepting `type="Muon"` in configuration. Common Muon implementations apply the Muon update only to selected matrix parameters and use a companion optimizer such as AdamW for embeddings, LM heads, norms, biases, and other parameters.

Planned Muon support includes:

- explicit parameter classification/tagging before optimizer initialization
- documented rules for which trainable parameters receive Muon updates
- embeddings and LM heads excluded from Muon unless an implementation explicitly supports them
- norms, biases, scalar/vector parameters, and other non-matrix parameters handled by the companion optimizer path
- DeepSpeed Muon integration using the parameter metadata required by the installed DeepSpeed implementation
- Torch-owned Muon support when a stable implementation is available
- compatibility with LoRA/PEFT trainable parameter sets
- tests that verify every trainable parameter belongs to exactly one intended optimizer group

The generic AdamW path must not regain an implicit `ndim`-based weight-decay policy. Any dimension-based rule used by Muon is optimizer-specific, named, documented, and tested.

---

## Phase 7: evaluation framework

Evaluation should be a first-class trainer capability rather than application-specific callback code.

### Scope

- optional validation dataset/dataloader
- evaluation at epoch boundaries
- optional evaluation every N optimizer steps
- validation loss
- perplexity for causal-LM evaluation where meaningful
- pluggable metric functions
- distributed-safe metric aggregation through Accelerate primitives such as `gather_for_metrics`
- no rank-local metric reported under a global metric name
- evaluation state isolated from training state
- no optimizer or scheduler mutation during evaluation
- clean interaction with gradient accumulation and checkpoint cadence

### Future metric layers

Task-specific classification, ranking, generation, or judge metrics should be implemented as pluggable metric components rather than embedded into the core causal-LM trainer.

### Exit criteria

- single-process and multi-process evaluation produce equivalent aggregate metrics
- evaluation cadence does not change optimizer-step semantics
- validation can be disabled without affecting the training path

---

## Phase 8: backend and observability sophistication

These features matter for larger-scale training, but they are lower priority than PEFT, Muon, and evaluation because the current basic runtime is already functional.

### ZeRO-3 initialization-time partitioning

Current ZeRO-3 support partitions parameters after `accelerator.prepare(...)`. A future model-loading path should allow parameters to be partitioned during `from_pretrained()` so a full model does not need to materialize first on every rank.

Planned work:

- move backend context creation early enough to participate in model construction
- support ZeRO-3 initialization-time partitioning without breaking the simple non-DeepSpeed load API
- test large-model loading semantics separately from training-time ZeRO-3 partitioning

### Advanced DeepSpeed configuration

Potential additions, only when justified by real workloads:

- bucket-size tuning
- stage-3 prefetch and parameter-persistence controls
- more explicit offload tuning
- DeepSpeed-owned optimizer with a deferred Torch/Hugging Face scheduler through `DummyScheduler(lr_scheduler_callable=...)`
- distributed profiler integration

### Profiling and observability

- optional PyTorch profiler integration
- step-time accounting
- throughput metrics such as tokens/sec
- memory reporting
- optional FLOP estimation where reliable
- backend-neutral logging hooks rather than hard-coding a tracking vendor

### Tensor parallelism

Tensor parallelism is intentionally later than ZeRO because it changes model execution semantics rather than only data-parallel state ownership.

Before adding TP, the project should have stable:

- PEFT behavior
- checkpoint/export contracts
- distributed evaluation
- optimizer ownership
- packed-sequence backend validation

---

## Data and objective roadmap outside the immediate trainer core

The companion `llm-training-data` package already covers supervised causal-LM rendering, chat formatting, tokenization, label masking, dynamic padding, and packed SFT.

Future cross-repository work can include:

- preference-data preparation with `prompt` / `chosen` / `rejected` contracts
- DPO-compatible dataset adapters
- stronger training/inference formatting parity
- checkpoint-carried prompt/tokenization metadata where it improves reproducibility
- more explicit packed-sequence backend capability checks

These should remain separated from the trainer runtime so the training package does not become coupled to a specific dataset engine or application schema.

---

## Priority order

The current priority order is:

```text
1. PEFT foundation
2. LoRA
3. TrainableTokens
4. adapter-aware checkpoint / merge lifecycle
5. PEFT + gradient-checkpointing + ZeRO integration hardening
6. Muon and richer optimizers
7. evaluation framework
8. profiler / advanced DeepSpeed / ZeRO-3 init-time loading / tensor parallelism
```

The first five phases are the most important feature-completeness gap in the current runtime.

---

## Design constraints

The following rules should remain true as the project grows:

- Core APIs remain general and backend-independent where possible.
- Application-specific orchestration, data systems, serving engines, and domain wrappers stay outside the core runtime.
- Torch ownership remains the default; backend-specific ownership is explicit.
- A hyperparameter has one source of truth.
- `FreezeConfig` defines what should be trainable; PEFT/optimizer layers must not silently redefine that policy.
- Trainable parameters must be visible to the backend that owns optimization and checkpointing.
- Adapter artifacts, training-state checkpoints, and merged deployment models are distinct concepts.
- Distributed metrics must represent all participating ranks, not a single local shard.
- Optimizer-specific parameter grouping must be explicit and documented.
- No hidden generic policy should be inferred from tensor rank, parameter name, or backend unless that rule is part of a named feature contract.
- Features are not marked supported until their relevant training, checkpoint/resume, and export paths are tested.
- GPU/distributed tests remain separate from ordinary CPU CI unless the repository gains reliable dedicated hardware runners.
