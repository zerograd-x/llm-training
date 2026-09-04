# Single-GPU validation audit — 2026-09-04 UTC

## Scope and provenance

Repository main inspected: `3ba90dd119c4dbf6e67319444dcd46f1f1dbf7a1`.
Existing runs used an installed `llm-training 0.1.0` wheel reported in the preceding session as built from that commit; this audit has not independently reverified the installed source hash. No new GPU training was launched during this audit.

Reported environment: NVIDIA L4, torch 2.7.1+cu128, CUDA toolkit 12.8,
DeepSpeed 0.19.6, Accelerate 1.14.0, Transformers 5.16.1, PEFT 0.20.0.
All results concern one GPU / world size 1. Two-GPU ZeRO-3 remains deferred until quota is increased and execution is authorized.

Evidence below is from retrieved SSM stdout/stderr, not just the command status.
Several workers emit `ok:false` while returning zero, and shell loops end with `exit 0`.
A successful SSM command is therefore not a suite pass.

## Matrix

| Case | Observed result | Remaining gate |
| --- | --- | --- |
| Torch optimizer + HF scheduler, ZeRO-3 | Two steps; LR 0.001 → 0; no order warning captured (A) | Full LR trace, actual optimizer update trace, warning capture scope and old/new wrapper comparison |
| Clean CPU optimizer offload | Fresh-process case reaches step 2 (B); environment alignment reported | Independently verify CPUAdam identity/build log, absence of DS_SKIP_CUDA_CHECK, and offload-specific resume/export |
| LoRA + BF16 | Two steps and adapter files; trainable names contain LoRA A/B (A) | Isolated CUDA/BF16 assertions and explicit parameter-change audit |
| LoRA + ZeRO-3 | DeepSpeedEngine, two steps, adapter saved (B) | Numerical artifact equivalence |
| LoRA + ZeRO-3 + gradient checkpointing | Two steps and adapter saved (B) | Confirm checkpointing mode/config from worker |
| LoRA checkpoint → fresh resume | Checkpoint loaded, but before=2 and after=2 (B) | Save at step 1; fresh process must perform step 2; compare optimizer/scheduler/RNG/data position with uninterrupted control |
| LoRA adapter merge → reload | Merge produced finite logits, shape [1,4,32] (B); separate adapter reload also finite (D) | Compare logits against trained adapter; connect artifacts to a resume that actually advances |
| TrainableTokens GPU | Training and adapter saved; later adapter-state probe confirms selected-token delta changed and merged model has finite logits (F) | Correct effective-row assertion; early base-weight probe reported changed_new=false (C) |
| TrainableTokens + ZeRO-3, default CPU-loaded model | Fails during PEFT token initialization: CPU tensor broadcast under NCCL (B) | Resolve initialization/device ordering and rerun unmodified public path |
| TrainableTokens + ZeRO-3, CUDA-preloaded model | Two steps; changed adapter state and finite merged logits (C,F) | Workaround only; old-row stability probe failed, requiring dtype-aligned effective-weight comparison |
| TrainableTokens fresh resume | Step 1 checkpoint → new process reports before=1, after=2 (D) | Verify backend/config and full state equivalence |
| Full-model export reload | In-process reload failed with weight must be 2-D (E); clean-process reload succeeds with embedding shape [32,16] (G) | Treat process isolation as required; do not infer corrupt export from in-process failure |
| Optimizer parameter coverage | Post-ZeRO wrapper reports zero original params (E,G) | Audit before prepare and map ZeRO-owned parameters; current probe does not establish missing ownership |
| Two-GPU ZeRO-3 | Not run | Quota increase; true partition/collective and distributed artifact tests |

## Findings and required test changes

1. Run each independent backend configuration in a separate subprocess. The initial
   eight-case worker failed on its second DeepSpeed Accelerator because process-global
   Accelerate state was already initialized.
2. Aggregate semantic results and return nonzero on any failed assertion or missing
   case. Preserve individual exit codes, stdout, stderr, versions and source SHA.
3. Resume from an unfinished run with the same configured horizon. A completed-run
   restore (2 → 2) validates loading only.
4. Inspect TrainableTokens adapter/effective weights rather than assuming frozen base
   embedding rows contain the trained token values. Compare old rows in the same
   dtype and with ZeRO parameters gathered; evaluate actual selected token ids.
5. Run offline merge/reload after training workers exit. Compare trained-adapter and
   merged logits on identical inputs, with stated BF16 tolerances.
6. Record scheduler LR before and after every real optimizer update, including
   accumulation boundaries. Current Trainer source calls optimizer.step() before
   scheduler.step(). The warning was not reproduced here; its classification as an
   Accelerate/DeepSpeed wrapper artifact remains unproven.
7. Clean offload validation must assert DS_SKIP_CUDA_CHECK is absent, record nvcc and
   torch CUDA versions, prove CPUAdam compiled/loaded without a bypass, then exercise
   checkpoint → fresh resume → export → fresh reload.

## Evidence index

SSM instance: `i-001c21699131e7aad`, region `us-west-2`.
Worker staging bucket: `llm-training-caveat-cleanup-513586111913-20260904`.

- A: `86376e11-798b-4b08-9036-7a790a5d0cfe` — original eight-case worker.
- B: `cc87d424-6a2d-4626-90db-555f678bdf0a` — isolated fresh cases.
- C: `9c5a5818-7709-46a0-9595-80c80f2537d9` — token row probes.
- D: `4485307d-8581-466d-afa7-2dc7e0a8ec32` — extra cases and token resume.
- E: `7d188f31-3fa7-4176-a3c7-4194f92fd003` — regression cases.
- F: `fc8dac81-1a36-48c7-9f51-61eca1b989b1` — saved token state and merge probe.
- G: `d8a3d7b1-5cf9-4e68-975a-1c81d4712e7d` — clean-process full-model reload.

Worker-source inspection through a new SSM command was blocked by automatic
approval review. Worker assertions and unresolved behavior must be inspected
before promoting these observations to complete validation.
