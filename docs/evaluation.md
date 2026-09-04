# Evaluation

`Trainer` accepts `eval_dataset`, optional `eval_collate_fn`, and a tuple of
`EvaluationMetric` instances. The default evaluation collator is the training
collator. Datasets must support `len()` and indexed access.

```python
from llm_training import EvaluationConfig, Trainer, TrainingConfig

config = TrainingConfig(
    output_dir="run",
    evaluation=EvaluationConfig(batch_size=4, every_steps=100, every_epochs=1),
)
trainer = Trainer(
    model=model, train_dataset=train, eval_dataset=validation,
    collate_fn=collator, config=config,
)
trainer.train()
metrics = trainer.evaluate()  # all participating ranks must call this
```

With an evaluation dataset, epoch evaluation defaults to every epoch. Set
`every_epochs=None` to disable it; `every_steps=None` disables step evaluation.
When both schedules coincide, one evaluation is recorded. `last_eval_step` and
`last_eval_epoch` are checkpointed. In-memory `evaluation_history` contains only
evaluations executed by the current process, so callers can persist/concatenate
histories across runs. Evaluation is allowed only at a completed accumulation
boundary.

## Metric definition

- `eval_loss`: globally summed unsmoothed causal negative log likelihood divided
  by the number of supervised tokens, with labels `-100` excluded.
- `eval_perplexity`: `exp(eval_loss)`; overflow is reported as infinity.
- `eval_tokens`, `eval_examples`: real supervised-token and example counts.

The default metric is independent of a custom training loss or label smoothing.
It is also different from the training reduction: training averages each
microbatch's loss and gives each microbatch equal weight within an accumulation
window. The last window uses its actual microbatch count.

Every rank forwards the same number of equal-sized batches, including ZeRO-3
ranks. Explicit example masks remove duplicates introduced solely for evaluation
padding. Reductions use additive global statistics; rank means are never averaged.
Training sampling may still repeat examples to balance ranks on uneven datasets;
that is Accelerate's training-loader contract, not evaluation sample weighting.

Evaluation preserves Python, NumPy, Torch and initialized accelerator RNG states,
module training flags, and training-loader accumulation bookkeeping. It does not
step the optimizer or scheduler. Custom collators/metrics should avoid external
side effects; arbitrary application state cannot be restored automatically.

## Custom metrics

```python
import torch
from llm_training import EvaluationMetric

class TokenAccuracy(EvaluationMetric):
    def statistics(self, logits, labels):
        # Labels are already aligned for next-token prediction.
        keep = labels != -100
        correct = (logits.argmax(-1) == labels) & keep
        return torch.stack((correct.sum(-1), keep.sum(-1)), dim=-1)

    def compute(self, totals):
        return {"token_accuracy": (totals[0] / totals[1]).item()}
```

`statistics` returns `[batch, K]` additive values. The evaluator excludes padded
examples and sums values across batches and ranks. `compute` receives the global
vector on CPU and returns finite, uniquely named scalars. All ranks must use the
same dataset length, batch size, metric count and statistic widths. Metric
exceptions propagate to peers at validation boundaries. Model/backend crashes
still require launcher timeouts and process-group supervision.

Use a separate held-out test set for post-training generation metrics. Exact
match in the portable workflow is a simple integration metric, not a claim of
model quality or a general semantic evaluator.
