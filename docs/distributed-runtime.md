# Distributed runtime lifecycle

`llm-training` separates **distributed worker responsibilities** from
**single-process driver responsibilities**. The goal is to make rank ownership,
barriers, model materialization, training, checkpointing, adapter export, and
offline merge explicit rather than scattering ad-hoc `if rank == 0` checks
through the trainer.

## Runtime creation happens before model loading

The recommended distributed entry point is:

```python
from llm_training import DistributedRuntime, Trainer, TrainingConfig

config = TrainingConfig(output_dir="./outputs")
runtime = DistributedRuntime(config)
components = runtime.load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=train_dataset,
    collate_fn=collate_fn,
    config=config,
    runtime=runtime,
)
```

`DistributedRuntime` constructs the Accelerate/backend runtime once. `Trainer`
reuses that exact Accelerator; it does not create a second process group or a
second backend instance.

This changes the lifecycle from:

```text
model load
    ↓
Trainer
    ↓
Accelerator / process group
```

to:

```text
DistributedRuntime
    ↓
Accelerator / process group / rank identity
    ↓
model materialization + model load
    ↓
Trainer reuses the same runtime
```

A plain CPU `torchrun` launch needs special care: Accelerate only selects its
`MULTI_CPU` distributed backend when `Accelerator(cpu=True)` is used. The
runtime detects `WORLD_SIZE > 1` with no CUDA device and enables that mode so
CPU integration workers do not silently become independent single-process
runtimes. DeepSpeed keeps its own accelerator-selection path.

## Global rank versus local rank

The runtime exposes both global and node-local ownership:

```text
process_index          global rank
local_process_index    rank within this node
is_main_process        global rank 0
is_local_main_process  local rank 0 on each node
```

For example, with two four-GPU nodes:

```text
node A: global ranks 0 1 2 3   local ranks 0 1 2 3
node B: global ranks 4 5 6 7   local ranks 0 1 2 3
```

Global rank 0 is appropriate for job-global metadata or a shared filesystem.
Local rank 0 is appropriate for node-local caches or disks.

## Model/cache materialization

`runtime.load_causal_lm(...)` lets local rank 0 enter Hugging Face
`from_pretrained(...)` first. On Hub-backed models this lets local rank 0
populate the node-local Hugging Face cache before sibling ranks perform their
own model construction.

All ranks still construct a model object:

```text
local rank 0
    ↓
resolve/download into node-local cache
    ↓
barrier
    ↓
all local ranks construct their own model object from the completed cache
```

This is intentional. Data-parallel/DeepSpeed workers each need a model object;
rank 0 should own remote/cache materialization, not the only model instance.

For ZeRO-3, `load_causal_lm` also temporarily suspends Transformers' implicit
initialization-time sharding hook, restoring it after loading. The complete
model must be available for adapter injection before `accelerator.prepare`.
Custom `from_pretrained` calls made after runtime creation must use
`with runtime.model_loading_context():`; see the
[loading contract](deepspeed.md#zero-3-model-loading-limitation).

For non-Hugging-Face storage, use `materialize_path(...)`:

```python
local_path = runtime.materialize_path(
    "/local_nvme/base-model",
    lambda destination: download_from_storage(source, destination),
    scope="local",
)
```

`scope="local"` means local rank 0 on every node performs the callback.
`scope="global"` means only global rank 0 performs it and therefore requires a
destination visible to every node.

The callback must materialize the deterministic destination path. Followers do
not execute the callback; they wait for the owner and then verify that the path
is visible in their own filesystem namespace.

## Data ownership

Model-source ownership and training-data ownership are different concepts.
Normal map-style training uses one DataLoader per process. Accelerate shards the
batch sampler during `accelerator.prepare(...)`; rank 0 does not centrally read
and broadcast every training batch.

With `split_batches=False` (the current default):

```text
rank 0 -> batch 0, batch 2, ...
rank 1 -> batch 1, batch 3, ...
```

The number of synchronized optimizer updates per epoch is:

```text
per_rank_batches
    = ceil(unsharded_batches / world_size)

optimizer_updates
    = ceil(per_rank_batches / grad_accumulation)
```

Scheduler accounting has one additional Accelerate-specific detail. With
`split_batches=False`, `AcceleratedScheduler` advances the underlying scheduler
`world_size` times for every synchronized optimizer update. The runtime therefore
builds the scheduler in underlying-step units:

```text
scheduler_steps_per_epoch
    = optimizer_updates * world_size
```

This preserves the intended LR horizon even when the final batch group is
uneven and Accelerate repeats data to keep ranks synchronized.

## Rank ownership by operation

```text
remote/node-local model materialization   local rank 0 per node
shared-filesystem materialization         global rank 0
model construction                        every rank
training DataLoader iteration             every rank, sharded by Accelerate
forward/backward                          every rank
DDP/ZeRO collectives                      every rank
training-state checkpoint                 every participating rank
trainer/config/tokenizer metadata write   global rank 0
adapter parameter gather                  participating ranks
adapter file serialization                global rank 0
logs via accelerator.print                global rank 0
LoRA / TrainableTokens offline merge      single-process driver only
optional artifact publication             driver / publishing layer
```

A training-state checkpoint cannot be reduced to `if rank == 0: save()`: under
ZeRO each rank may own a distinct optimizer/model-state shard, so every rank must
participate in the backend checkpoint operation. Only small logical metadata is
rank-0-only.

## Worker versus driver boundary

Adapter training ends inside the distributed worker group:

```text
all ranks train
    ↓
all required ranks participate in adapter gather
    ↓
global rank 0 serializes adapter artifact
    ↓
barrier
    ↓
distributed workers exit
```

Merge starts after that worker group has ended:

```text
single-process driver
    ↓
load exact base
    ↓
load adapter / hybrid base overrides
    ↓
merge_and_unload()
    ↓
plain Hugging Face model
    ↓
optional publish
```

`merge_lora()` and `merge_trainable_tokens()` enforce this boundary. They raise
when called while an active/environment-declared `WORLD_SIZE` is greater than
one instead of allowing every worker to independently materialize and write a
full merged model.

## Current validation boundary

The CPU integration suite launches two real processes with
`torch.distributed.run` and validates:

```text
2-process rank/global-local identity                ✅
local-rank-0-only materialization                   ✅
all-rank model construction after materialization  ✅
rank-sharded DataLoader batches                     ✅
Trainer reuse of the same DistributedRuntime        ✅
distributed scheduler horizon                       ✅
global/local-main-only side effects                 ✅
driver-only operation guard                         ✅
```

The [GPU validation](gpu-validation.md) additionally validates two L4
hosts using NCCL and real ZeRO-3 partitions, sharded adapter export and
distributed checkpoint/resume. Global ranks are 0/1 and local rank is 0 on
both hosts. Same-host local ranks 0/1 and broader model/configuration coverage
remain pending.
