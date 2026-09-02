# Core training roadmap

This roadmap tracks capabilities that belong in the reusable `llm-training` runtime itself. Application-specific data systems, serving stacks, and domain wrappers remain outside this repository.

## Current foundation

- Hugging Face causal-LM loading and export
- selective freezing
- configurable causal-LM loss
- AdamW and Hugging Face schedulers
- checkpoint/resume with optimizer, scheduler, RNG, and logical trainer position
- gradient accumulation and gradient clipping
- model-native gradient checkpointing
- optional DeepSpeed ZeRO-1/2/3 backend
- explicit DeepSpeed-owned optimizer and scheduler through `DummyOptim` / `DummyScheduler`
- ZeRO-3-aware distributed export

## Next: PEFT / LoRA

PEFT is a core training concern because it changes which parameters exist, which parameters are trainable, and how checkpoints must be exported.

Planned scope:

- optional `peft` dependency
- general LoRA configuration
- target-module discovery based on model structure rather than model-specific hard-coded names
- predictable interaction with `FreezeConfig`
- correct handling of embeddings and LM heads when they remain fully trainable outside LoRA
- gradient-checkpointing compatibility for adapter training
- adapter checkpoint export
- optional merged Hugging Face model export
- checkpoint/resume tests for LoRA training
- compatibility tests with the DeepSpeed backend, including ZeRO-3 where practical

The implementation should not depend on any company-specific model wrapper or orchestration system.

## Next: richer optimizers

The optimizer layer should support more than AdamW while preserving one explicit ownership model.

Planned scope:

- optimizer selection in the normal Torch-owned path
- DeepSpeed-native optimizer selection in the DeepSpeed-owned path
- Muon support where the installed Torch/DeepSpeed version provides a suitable implementation
- explicit parameter grouping/tagging rules for optimizers such as Muon that should apply only to selected matrix parameters
- no hidden `ndim`-based policy without a named, documented optimizer-specific rule
- optimizer state checkpoint/resume tests

Muon needs special care because common implementations do not apply the Muon update to every parameter. Embeddings, LM heads, norms, biases, and other non-matrix parameters typically require a companion optimizer path. That grouping policy will be explicit and tested rather than inferred silently from the generic AdamW path.

## Later backend work

- ZeRO-3 initialization-time partitioning so parameters can be sharded during `from_pretrained()` rather than only after `accelerator.prepare(...)`
- DeepSpeed-owned optimizer with a deferred Torch/Hugging Face scheduler through `DummyScheduler(lr_scheduler_callable=...)` if a concrete use case requires that mixed ownership mode
- multi-GPU DeepSpeed integration tests for training, checkpoint/resume, and export
- distributed evaluation/metric aggregation when evaluation becomes part of the core trainer

## Design constraints

- Core APIs remain general and backend-independent where possible.
- Torch ownership remains the default; backend-specific ownership is explicit.
- A hyperparameter has one source of truth.
- Trainable parameters must be visible to the backend that owns optimization and checkpointing.
- Features are not marked supported until their relevant training and checkpoint paths are tested.
