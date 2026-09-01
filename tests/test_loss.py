import pytest
import torch
from torch import nn

from llm_training import (
    BaseLoss,
    DefaultCrossEntropyLoss,
    load_loss,
    shift_causal_labels,
)
from llm_training.loss import validate_loss


def test_shift_labels_matches_sliced_logits_loss_and_gradients() -> None:
    torch.manual_seed(7)
    labels = torch.tensor(
        [
            [1, 2, 3, 4],
            [-100, 5, 6, -100],
        ]
    )
    logits_a = torch.randn(2, 4, 8, requires_grad=True)
    logits_b = logits_a.detach().clone().requires_grad_(True)
    loss_fn = DefaultCrossEntropyLoss()

    sliced_loss = loss_fn(
        logits_a[..., :-1, :].contiguous(),
        labels[..., 1:].contiguous(),
    )
    shifted_label_loss = loss_fn(logits_b, shift_causal_labels(labels))

    torch.testing.assert_close(shifted_label_loss, sliced_loss)

    sliced_loss.backward()
    shifted_label_loss.backward()
    torch.testing.assert_close(logits_b.grad[..., :-1, :], logits_a.grad[..., :-1, :])
    torch.testing.assert_close(logits_b.grad[..., -1, :], torch.zeros_like(logits_b.grad[..., -1, :]))


def test_ignored_positions_do_not_change_mean_denominator() -> None:
    torch.manual_seed(11)
    loss_fn = DefaultCrossEntropyLoss()
    logits = torch.randn(1, 2, 5)
    labels = torch.tensor([[1, 2]])

    base = loss_fn(logits, labels)
    padded = loss_fn(
        torch.cat([logits, torch.randn(1, 3, 5)], dim=1),
        torch.tensor([[1, 2, -100, -100, -100]]),
    )

    torch.testing.assert_close(padded, base)


def test_default_loss_promotes_bf16_logits_to_float32() -> None:
    logits = torch.randn(2, 3, 7, dtype=torch.bfloat16)
    labels = torch.tensor([[1, 2, 3], [4, 5, -100]])

    loss = DefaultCrossEntropyLoss()(logits, labels)

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)


def test_load_loss_supports_dotted_path_and_kwargs() -> None:
    loss_fn = load_loss(
        "llm_training.loss.DefaultCrossEntropyLoss",
        {"label_smoothing": 0.1},
    )

    assert isinstance(loss_fn, DefaultCrossEntropyLoss)
    assert loss_fn.label_smoothing == 0.1


def test_load_loss_rejects_non_loss_class() -> None:
    with pytest.raises(TypeError, match="BaseLoss subclass"):
        load_loss("llm_training.config.TrainingConfig")


class TrainableLoss(BaseLoss):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def forward(self, logits, labels):
        return logits.sum() * self.scale


def test_loss_rejects_trainable_parameters() -> None:
    with pytest.raises(ValueError, match="Trainable parameters"):
        validate_loss(TrainableLoss())
