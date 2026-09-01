import pytest
import torch
from torch import nn

from llm_training import (
    FreezeConfig,
    apply_freeze_policy,
    find_embedding,
    find_lm_head,
    find_transformer_layers,
    weights_are_tied,
)


class Backbone(nn.Module):
    def __init__(self, hidden_size: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)
        self.layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )


class TinyModel(nn.Module):
    def __init__(self, *, tied: bool = False) -> None:
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(8, 32, bias=False)
        if tied:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_model_introspection() -> None:
    model = TinyModel(tied=True)
    layer_path, layers = find_transformer_layers(model)
    embed_path, embedding = find_embedding(model)
    head = find_lm_head(model)

    assert layer_path == "model.layers"
    assert len(layers) == 3
    assert embed_path == "model.embed_tokens"
    assert embedding is model.model.embed_tokens
    assert head is not None and head[0] == "lm_head"
    assert weights_are_tied(model)


def test_freeze_last_layer_and_endpoints() -> None:
    model = TinyModel()
    summary = apply_freeze_policy(
        model,
        FreezeConfig(
            train_last_n_layers=1,
            train_embeddings=False,
            train_lm_head=False,
        ),
    )

    assert all(not p.requires_grad for p in model.model.layers[0].parameters())
    assert all(not p.requires_grad for p in model.model.layers[1].parameters())
    assert all(p.requires_grad for p in model.model.layers[2].parameters())
    assert all(not p.requires_grad for p in model.model.embed_tokens.parameters())
    assert all(not p.requires_grad for p in model.lm_head.parameters())
    assert 0 < summary.trainable_parameters < summary.total_parameters


def test_tied_embedding_and_head_use_logical_or() -> None:
    model = TinyModel(tied=True)
    with pytest.warns(UserWarning, match="share the same weight tensor"):
        apply_freeze_policy(
            model,
            FreezeConfig(train_embeddings=False, train_lm_head=True),
        )

    assert model.model.embed_tokens.weight.requires_grad
    assert model.lm_head.weight.requires_grad


def test_custom_prefix_is_final_unfreeze_override() -> None:
    model = TinyModel()
    apply_freeze_policy(
        model,
        FreezeConfig(
            train_last_n_layers=0,
            train_embeddings=False,
            train_lm_head=False,
            trainable_prefixes=("model.layers.0",),
        ),
    )

    assert all(p.requires_grad for p in model.model.layers[0].parameters())
    assert all(not p.requires_grad for p in model.model.layers[1].parameters())
