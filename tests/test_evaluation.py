from copy import deepcopy
import math
import random

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from llm_training import EvaluationConfig, EvaluationMetric, Trainer, TrainingConfig
from llm_training.loss import shift_causal_labels
from training_test_support import ToyLM, rows, collate


class TokenAccuracy(EvaluationMetric):
    def statistics(self, logits, labels):
        valid = labels != -100
        return torch.stack(
            [((logits.argmax(-1) == labels) & valid).sum(-1), valid.sum(-1)], -1
        )

    def compute(self, totals):
        return {"token_accuracy": (totals[0] / totals[1]).item()}


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_evaluation_token_weighted_and_no_padding_duplicates(tmp_path, batch_size):
    model = ToyLM()
    trainer = Trainer(
        model=model,
        train_dataset=rows(),
        eval_dataset=rows(),
        collate_fn=collate,
        eval_metrics=(TokenAccuracy(),),
        config=TrainingConfig(
            output_dir=str(tmp_path), evaluation=EvaluationConfig(batch_size=batch_size)
        ),
    )
    batch = collate(rows())
    with torch.no_grad():
        logits = model(batch["input_ids"]).logits
        labels = shift_causal_labels(batch["labels"])
        expected = F.cross_entropy(
            logits.float().flatten(0, 1), labels.flatten(), ignore_index=-100
        ).item()
    result = trainer.evaluate()
    assert result["eval_examples"] == 5 and result["eval_tokens"] == 7
    assert result["eval_loss"] == pytest.approx(expected, abs=1e-6)
    assert result["eval_perplexity"] == pytest.approx(math.exp(expected), rel=1e-6)
    assert 0 <= result["token_accuracy"] <= 1
    assert model.training


def test_evaluation_preserves_rng_and_training_updates(tmp_path):
    torch.manual_seed(71)
    initial = ToyLM(0.4)

    def make(name, do_eval):
        return Trainer(
            model=deepcopy(initial),
            train_dataset=rows(),
            collate_fn=collate,
            eval_dataset=rows() if do_eval else None,
            config=TrainingConfig(
                output_dir=str(tmp_path / name),
                num_epochs=3,
                gradient_accumulation_steps=2,
                evaluation=EvaluationConfig(every_steps=1, batch_size=2),
            ),
        )

    baseline = make("baseline", False)
    baseline.train()
    evaluated = make("evaluated", True)
    before = (torch.get_rng_state(), random.getstate(), np.random.get_state())
    evaluated.evaluate()
    assert (
        torch.equal(torch.get_rng_state(), before[0]) and random.getstate() == before[1]
    )
    np.testing.assert_array_equal(np.random.get_state()[1], before[2][1])
    evaluated.train()
    assert baseline.state.global_step == evaluated.state.global_step == 9
    assert baseline.scheduler.state_dict() == evaluated.scheduler.state_dict()
    for a, b in zip(baseline.model.parameters(), evaluated.model.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_evaluation_error_restores_mode_and_rng(tmp_path):
    bad = [{"input_ids": torch.tensor([1, 2]), "labels": torch.tensor([-100, -100])}]
    trainer = Trainer(
        model=ToyLM(),
        train_dataset=rows(),
        collate_fn=collate,
        config=TrainingConfig(output_dir=str(tmp_path)),
    )
    state = torch.get_rng_state()
    with pytest.raises(ValueError, match="no supervised tokens"):
        trainer.evaluate(bad)
    assert trainer.model.training and torch.equal(state, torch.get_rng_state())

