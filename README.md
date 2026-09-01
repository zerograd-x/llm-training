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
- configurable causal-LM loss with a default masked cross-entropy implementation
- AdamW + Hugging Face schedulers
- deterministic epoch sampling
- checkpoint/save/resume of model, optimizer, scheduler, RNG, and trainer position
- a thin adapter for `llm-training-data` column-batch collators

Serving, data-processing systems, domain-specific model wrappers, and other application-specific extensions stay outside the core runtime.

## Install

```bash
pip install -e .
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
(model/freeze/loss/optimizer/scheduler/train/checkpoint)
    |
    v
trained checkpoint
    |
    v
model-artifact-transfer   # optional publishing layer
```

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

## Checkpoint semantics

A checkpoint is a **training checkpoint**, not merely exported model weights. Accelerate owns the runtime state (model, optimizer, scheduler, mixed-precision scaler when present, and RNG state), while `llm-training` stores the logical trainer position and configuration next to it.

```text
checkpoint-00001000/
├── ... Accelerate runtime state ...
├── trainer_state.json
├── training_config.json
└── tokenizer/
```

`TrainerState` records the next epoch/batch position, so a resumed single-process run can continue at the next batch rather than restarting the epoch.

## Tests

Run unit tests with:

```bash
pytest -m "not integration and not gpu"
```

Run real Hugging Face CPU integration tests with:

```bash
pytest -m "integration and not gpu" tests/integration
```

The checkpoint-resume integration test deliberately interrupts training after the first optimizer step, creates a fresh model and `Trainer`, resumes from `latest`, restores the saved model state and trainer position, skips the already-completed batch, and finishes the second optimizer step.

The `llm-training-data` bridge test uses the real `build_sft_collator` from the sibling repository. It is skipped when `llm_training_data` is not installed. To run that cross-repository test locally:

```bash
pip install -e ../llm-training-data
pytest -m integration tests/integration/test_training_data_bridge.py
```

GPU tests are intentionally excluded from normal push/PR CI. For the manual AWS single-GPU BF16 validation path, including EC2 quota, DLAMI, SSM, GPU checks, troubleshooting, and cleanup, see [`docs/aws-gpu-smoke.md`](docs/aws-gpu-smoke.md).
