"""Global custom metric statistics and asymmetric callback failures on Gloo."""

from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
from torch import distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training_test_support import ToyLM, collate, rows
from test_evaluation import TokenAccuracy
from llm_training import EvaluationConfig, Trainer, TrainingConfig
from llm_training.loss import shift_causal_labels


def main(directory):
    dist.init_process_group("gloo", timeout=timedelta(seconds=30))
    rank = dist.get_rank()
    trainer = Trainer(
        model=ToyLM(),
        train_dataset=rows(),
        eval_dataset=rows(),
        collate_fn=collate,
        eval_metrics=(TokenAccuracy(),),
        config=TrainingConfig(
            output_dir=str(directory), evaluation=EvaluationConfig(batch_size=2)
        ),
    )
    model = trainer.accelerator.unwrap_model(trainer.model)
    batch = collate(rows())
    model.eval()
    with torch.no_grad():
        labels = shift_causal_labels(batch["labels"])
        valid = labels != -100
        expected = (
            ((model(batch["input_ids"]).logits.argmax(-1) == labels) & valid).sum()
            / valid.sum()
        ).item()
    model.train()
    state = torch.get_rng_state()
    result = trainer.evaluate()
    assert result["eval_examples"] == 5 and result["eval_tokens"] == 7
    assert abs(result["token_accuracy"] - expected) < 1e-7
    assert torch.equal(state, torch.get_rng_state()) and model.training

    class UnequalWidths(TokenAccuracy):
        def statistics(self, logits, labels):
            return torch.ones((labels.shape[0], rank + 1))

    class BrokenCompute(TokenAccuracy):
        def compute(self, totals):
            if rank == 1:
                raise ValueError("intentional metric callback failure")
            return {"first_metric": 1.0}

    errors = []
    for metrics, needle in [
        ((UnequalWidths(),), "widths must agree"),
        ((BrokenCompute(), TokenAccuracy()), "intentional metric callback failure"),
    ]:
        trainer.eval_metrics = metrics
        try:
            trainer.evaluate()
        except (ValueError, RuntimeError) as error:
            assert needle in str(error)
            errors.append(str(error))
        else:
            raise AssertionError("Invalid distributed metric did not fail")
        assert model.training and torch.equal(state, torch.get_rng_state())
    (directory / f"rank-{rank}.json").write_text(
        json.dumps(dict(ok=True, metrics=result, errors=errors))
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main(Path(sys.argv[1]))

