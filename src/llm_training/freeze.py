from __future__ import annotations

from dataclasses import dataclass
import warnings

from torch import nn

from .config import FreezeConfig
from .model_introspection import find_embedding, find_lm_head, find_transformer_layers, weights_are_tied


@dataclass(frozen=True)
class FreezeSummary:
    trainable_parameters: int
    total_parameters: int

    @property
    def trainable_fraction(self) -> float:
        if self.total_parameters == 0:
            return 0.0
        return self.trainable_parameters / self.total_parameters


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = value


def _unfreeze_prefixes(model: nn.Module, prefixes: tuple[str, ...]) -> None:
    if not prefixes:
        return
    for name, parameter in model.named_parameters():
        if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes):
            parameter.requires_grad = True


def summarize_trainable_parameters(model: nn.Module) -> FreezeSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return FreezeSummary(trainable_parameters=trainable, total_parameters=total)


def apply_freeze_policy(model: nn.Module, config: FreezeConfig) -> FreezeSummary:
    """Apply a deterministic base-model trainability policy in place.

    Order matters:
      1. optionally freeze all transformer blocks and unfreeze the last N,
      2. set embedding / LM-head trainability,
      3. unfreeze explicitly requested parameter prefixes.
    """

    if config.train_last_n_layers is not None:
        if config.train_last_n_layers < 0:
            raise ValueError("train_last_n_layers must be >= 0 or None")
        _, layers = find_transformer_layers(model)
        if config.train_last_n_layers > len(layers):
            raise ValueError(
                f"train_last_n_layers={config.train_last_n_layers} exceeds "
                f"the model's {len(layers)} transformer layers"
            )
        for layer in layers:
            _set_requires_grad(layer, False)
        if config.train_last_n_layers:
            for layer in layers[-config.train_last_n_layers :]:
                _set_requires_grad(layer, True)

    _, embedding = find_embedding(model)
    lm_head_info = find_lm_head(model)

    if lm_head_info is not None and weights_are_tied(model):
        effective = config.train_embeddings or config.train_lm_head
        if config.train_embeddings != config.train_lm_head:
            warnings.warn(
                "Input embeddings and LM head share the same weight tensor; "
                "their effective trainability is the logical OR of "
                "train_embeddings and train_lm_head.",
                stacklevel=2,
            )
        _set_requires_grad(embedding, effective)
        _set_requires_grad(lm_head_info[1], effective)
    else:
        _set_requires_grad(embedding, config.train_embeddings)
        if lm_head_info is not None:
            _set_requires_grad(lm_head_info[1], config.train_lm_head)

    _unfreeze_prefixes(model, config.trainable_prefixes)
    return summarize_trainable_parameters(model)
