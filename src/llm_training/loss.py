from __future__ import annotations

from abc import ABC, abstractmethod
import importlib
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


IGNORE_INDEX = -100


class BaseLoss(nn.Module, ABC):
    """Base contract for causal-LM losses.

    The trainer performs the causal next-token alignment before calling the
    loss. Implementations therefore receive position-aligned ``logits`` and
    ``labels`` and must treat labels equal to ``IGNORE_INDEX`` as unsupervised.

    Loss modules may contain non-trainable buffers, but trainable parameters are
    intentionally unsupported so model/optimizer sharding remains owned by the
    training backend rather than by the loss implementation.
    """

    @abstractmethod
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return a scalar loss for aligned logits and labels."""


class DefaultCrossEntropyLoss(BaseLoss):
    """Standard masked cross-entropy for causal language modeling."""

    def __init__(self, *, label_smoothing: float = 0.0) -> None:
        super().__init__()
        if not 0.0 <= label_smoothing <= 1.0:
            raise ValueError("label_smoothing must be in [0, 1]")
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
            label_smoothing=self.label_smoothing,
        )


def shift_causal_labels(labels: torch.Tensor) -> torch.Tensor:
    """Align labels with causal-LM logits without copying the logits tensor.

    For labels ``[t0, t1, ..., tn]`` this returns
    ``[t1, t2, ..., tn, IGNORE_INDEX]`` so ``logits[..., i, :]`` is supervised
    by the next token while the final logit position contributes no loss.
    """

    return F.pad(labels, (0, 1), value=IGNORE_INDEX)[..., 1:].contiguous()


def validate_loss(loss_fn: BaseLoss) -> BaseLoss:
    """Validate the runtime contract expected by the trainer."""

    if not isinstance(loss_fn, BaseLoss):
        raise TypeError("loss_fn must be an instance of BaseLoss")
    if any(parameter.requires_grad for parameter in loss_fn.parameters()):
        raise ValueError(
            "Trainable parameters inside loss_fn are not supported; "
            "attach trainable objective components to the model instead"
        )
    return loss_fn


def load_loss(
    module_path: str | None = None,
    kwargs: Mapping[str, Any] | None = None,
) -> BaseLoss:
    """Construct the default loss or load a custom BaseLoss by dotted path."""

    loss_kwargs = dict(kwargs or {})
    if module_path is None:
        return validate_loss(DefaultCrossEntropyLoss(**loss_kwargs))

    module_name, class_name = module_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if not isinstance(cls, type) or not issubclass(cls, BaseLoss):
        raise TypeError(f"{module_path} must point to a BaseLoss subclass")

    return validate_loss(cls(**loss_kwargs))
