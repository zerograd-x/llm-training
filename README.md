# llm-training

A small, opinionated training runtime for causal language models.

The project is intentionally split from [`llm-training-data`](https://github.com/zerograd-x/llm-training-data):

- **llm-training-data** owns rendering, tokenization, label construction, padding, and sequence packing.
- **llm-training** owns model loading, trainable-parameter policy, optimization, loss computation, checkpoint/resume, and the training loop.
- **model-artifact-transfer** can publish completed model artifacts after training.

Dependency direction is one-way: `llm-training` may consume `llm-training-data`; `llm-training-data` must not depend on this package.

## v0 scope

The first version focuses on one golden path:

- Hugging Face decoder-only causal LMs
- PyTorch + Hugging Face Accelerate
- full fine-tuning or selective layer freezing
- optional PEFT LoRA, including explicit hybrid LoRA + fully-trained base parameters
- optional PEFT TrainableTokens for newly appended vocabulary rows
- adapter-only training artifacts with offline merge back into plain Hugging Face checkpoints
- configurable causal-LM loss with a default masked cross-entropy implementation
- AdamW + Hugging Face schedulers
- optional model-native gradient checkpointing
- optional DeepSpeed ZeRO-1/2/3 backend
- optional DeepSpeed-owned optimizer and scheduler through explicit configuration
- deterministic epoch sampling
- checkpoint/save/resume of model, optimizer, scheduler, RNG, and trainer position
- a thin adapter for `llm-training-data` column-batch collators
- scheduled global token-weighted validation loss/perplexity and additive metrics
- a portable workflow with vocabulary preparation, artifact manifests and fresh-process generation

Serving, data-processing systems, domain-specific model wrappers, and other application-specific extensions stay outside the core runtime.

Muon and additional optimizer policies remain on the core training roadmap. See [`docs/roadmap.md`](docs/roadmap.md), [`docs/peft.md`](docs/peft.md), and [`docs/trainable_tokens.md`](docs/trainable_tokens.md).

## Install

```bash
pip install -e .
```

Install PEFT support for LoRA and TrainableTokens with:

```bash
pip install -e ".[peft]"
```

Install the optional DeepSpeed backend with:

```bash
pip install -e ".[deepspeed]"
```

For the SFT example and the cross-repository integration test, install the data package alongside this repository:

```bash
pip install -e ../llm-training-data
```

## Architecture

```text
raw examples
    |
    v
llm-training-data
(render/tokenize/mask/pad/pack)
    |
    v
llm-training
(model/freeze/PEFT/loss/optimizer/scheduler/train/checkpoint)
    |
    v
training state / adapter / merged model
    |
    v
model-artifact-transfer   # optional publishing layer
```

The [portable workflow](docs/train_causal_lm_workflow.md) composes these components
using plain Python and local paths, including separate train/merge/inference
processes. See [evaluation](docs/evaluation.md) for validation metrics and schedules.

## Minimal usage

```python
from llm_training import (
    FreezeConfig,
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
    apply_freeze_policy,
    load_causal_lm,
)
from llm_training_data import build_sft_collator

components = load_causal_lm("HuggingFaceTB/SmolLM2-135M-Instruct")
apply_freeze_policy(components.model, FreezeConfig())

data_collator = build_sft_collator(
    components.tokenizer,
    max_sequence_length=512,
)

trainer = Trainer(
    model=components.model,
    tokenizer=components.tokenizer,
    train_dataset=train_dataset,
    collate_fn=adapt_column_batch_collator(data_collator),
    config=TrainingConfig(output_dir="./outputs"),
)
trainer.train()
```

## Custom losses

The trainer owns causal next-token alignment. A loss implementation therefore receives position-aligned `logits` and `labels`; labels equal to `-100` are unsupervised and must not contribute to the loss or its reduction denominator.

The default implementation is `DefaultCrossEntropyLoss`, which computes cross-entropy in float32 even when model logits use a lower-precision dtype.

For Python-native customization, subclass `BaseLoss` and inject an instance directly:

```python
import torch
from torch.nn import functional as F

from llm_training import BaseLoss, Trainer


class FocalLoss(BaseLoss):
    def __init__(self, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        flat_logits = logits.float().reshape(-1, logits.size(-1))
        flat_labels = labels.reshape(-1)
        keep = flat_labels != -100
        flat_logits = flat_logits[keep]
        flat_labels = flat_labels[keep]
        logp = F.log_softmax(flat_logits, dim=-1)
        logp = logp.gather(1, flat_labels[:, None]).squeeze(1)
        p = logp.exp()
        return -((1 - p) ** self.gamma * logp).mean()


trainer = Trainer(
    ...,
    loss_fn=FocalLoss(gamma=2.0),
)
```

A custom loss can also be selected from configuration with a dotted import path:

```python
from llm_training import LossConfig, TrainingConfig

config = TrainingConfig(
    output_dir="./outputs",
    loss=LossConfig(
        module_path="my_project.losses.FocalLoss",
        kwargs={"gamma": 2.0},
    ),
)
```

Use either `Trainer(loss_fn=...)` or `TrainingConfig.loss`, not both. Loss modules must not contain trainable parameters; trainable objective components belong on the model so distributed/sharded training backends can manage them with the rest of the model state.

## DeepSpeed and gradient checkpointing

DeepSpeed is an optional backend. Enabling ZeRO alone keeps the normal Torch/Hugging Face optimizer and scheduler; DeepSpeed-owned optimizer/scheduler behavior is a separate explicit opt-in.

```python
from llm_training import (
    DeepSpeedConfig,
    GradientCheckpointingConfig,
    TrainingConfig,
)

config = TrainingConfig(
    output_dir="./outputs",
    mixed_precision="bf16",
    gradient_checkpointing=GradientCheckpointingConfig(
        enabled=True,
        use_reentrant=False,
    ),
    deepspeed=DeepSpeedConfig(zero_stage=3),
)
```

To let DeepSpeed create the optimizer and scheduler:

```python
from llm_training import (
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    TrainingConfig,
)

config = TrainingConfig(
    output_dir="./outputs",
    deepspeed=DeepSpeedConfig(
        zero_stage=3,
        optimizer=DeepSpeedOptimizerConfig(type="AdamW"),
        scheduler=DeepSpeedSchedulerConfig(type="WarmupDecayLR"),
    ),
)
```

This path uses Accelerate `DummyOptim` and `DummyScheduler` placeholders until `accelerator.prepare(...)`, where DeepSpeed creates the real implementations. A Torch optimizer with a DeepSpeed-owned scheduler is also supported. A DeepSpeed-owned optimizer with a Torch/Hugging Face scheduler is intentionally deferred because it requires constructing that scheduler only after the real DeepSpeed optimizer exists.

ZeRO-3 partitions parameters after `accelerator.prepare(...)`. The current API does not yet partition parameters during `from_pretrained()` itself, and each rank must initially hold a complete model. Adapter save gathers only required adapter/hybrid tensors. See the GPU matrices for the exact validated models, modes and topologies.

See [`docs/deepspeed.md`](docs/deepspeed.md) for ownership combinations, ZeRO stages, offload settings, initialization-time limitations, and export behavior.

## Checkpoint semantics

A checkpoint is a **training checkpoint**, not merely exported model weights. Accelerate owns the runtime state (model, optimizer, scheduler, mixed-precision scaler when present, and RNG state), while `llm-training` stores the logical trainer position and configuration next to it.

```text
checkpoint-00001000/
├── ... Accelerate runtime state ...
├── trainer_state.json
├── training_config.json
└── tokenizer/
```

`TrainerState` records the next epoch/batch position and evaluation position. Checkpoint completion markers and file inventories reject unfinished or missing runtime files before restore. Saving/evaluation requires a completed accumulation boundary. Partial final accumulation windows use their actual microbatch count. See [test contracts](docs/tests.md) for resume and failure coverage.

## Tests

Run unit tests with:

```bash
pytest -m "not integration and not gpu"
```

Run real Hugging Face CPU integration tests with:

```bash
pytest -m "integration and not gpu" tests/integration
```

The checkpoint-resume integration tests deliberately interrupt training after the first optimizer step, create a fresh model and `Trainer`, resume from `latest`, restore the saved model state and trainer position, skip the already-completed batch, and finish the second optimizer step. PEFT integration additionally verifies adapter serialization and offline merged-model numerical equivalence.

The `llm-training-data` bridge test uses the real `build_sft_collator` from the sibling repository. It is skipped when `llm_training_data` is not installed. To run that cross-repository test locally:

```bash
pip install -e ../llm-training-data
pytest -m integration tests/integration/test_training_data_bridge.py
```

GPU tests are intentionally excluded from normal push/PR CI. See [the GPU validation matrix and retained drivers](docs/gpu-validation.md).


See [the test guide](docs/tests.md) for boundary, evaluation, workflow and GPU validation commands.
