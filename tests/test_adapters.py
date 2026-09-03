from __future__ import annotations

import pytest
from torch import nn

from llm_training import (
    FreezeConfig,
    LoRAConfig,
    TrainingConfig,
    adapt_model_for_training,
    discover_lora_target_modules,
)


class Attention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class MLP(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 2, bias=False)
        self.down_proj = nn.Linear(hidden_size * 2, hidden_size, bias=False)


class Block(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = Attention(hidden_size)
        self.mlp = MLP(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)


class Backbone(nn.Module):
    def __init__(self, hidden_size: int = 8, layers: int = 3) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_size)
        self.layers = nn.ModuleList([Block(hidden_size) for _ in range(layers)])


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = Backbone()
        self.lm_head = nn.Linear(8, 32, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def test_default_lora_discovery_is_scoped_to_transformer_blocks() -> None:
    model = TinyModel()
    targets = discover_lora_target_modules(model, LoRAConfig())

    assert "model.layers.0.self_attn.q_proj" in targets
    assert "model.layers.2.mlp.down_proj" in targets
    assert all(target.startswith("model.layers.") for target in targets)
    assert all("lm_head" not in target for target in targets)
    assert all("embed_tokens" not in target for target in targets)


def test_lora_discovery_can_target_only_last_layers() -> None:
    model = TinyModel()
    targets = discover_lora_target_modules(
        model,
        LoRAConfig(target_last_n_layers=1),
    )

    assert targets
    assert all(target.startswith("model.layers.2.") for target in targets)


def test_explicit_lora_targets_are_resolved_to_fully_qualified_names() -> None:
    model = TinyModel()
    targets = discover_lora_target_modules(
        model,
        LoRAConfig(target_modules=("self_attn.q_proj", "mlp.down_proj")),
    )

    assert targets == (
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.mlp.down_proj",
        "model.layers.1.self_attn.q_proj",
        "model.layers.1.mlp.down_proj",
        "model.layers.2.self_attn.q_proj",
        "model.layers.2.mlp.down_proj",
    )


def test_lora_discovery_fails_when_no_target_matches() -> None:
    with pytest.raises(ValueError, match="No LoRA target modules"):
        discover_lora_target_modules(
            TinyModel(),
            LoRAConfig(target_modules=("does_not_exist",)),
        )


def test_freeze_without_lora_is_applied_inside_model_adaptation() -> None:
    model = TinyModel()
    adapted, summary = adapt_model_for_training(
        model,
        freeze=FreezeConfig(
            train_last_n_layers=1,
            train_embeddings=False,
            train_lm_head=False,
        ),
        lora=None,
    )

    assert adapted is model
    assert summary.method is None
    assert all(not p.requires_grad for p in model.model.layers[0].parameters())
    assert all(not p.requires_grad for p in model.model.layers[1].parameters())
    assert all(p.requires_grad for p in model.model.layers[2].parameters())
    assert all(not p.requires_grad for p in model.model.embed_tokens.parameters())
    assert all(not p.requires_grad for p in model.lm_head.parameters())


@pytest.mark.parametrize(
    ("lora", "message"),
    [
        (LoRAConfig(rank=0), "lora.rank"),
        (LoRAConfig(alpha=0), "lora.alpha"),
        (LoRAConfig(dropout=1.0), "lora.dropout"),
        (LoRAConfig(target_modules=()), "lora.target_modules"),
        (LoRAConfig(target_last_n_layers=0), "lora.target_last_n_layers"),
        (LoRAConfig(adapter_name=""), "lora.adapter_name"),
    ],
)
def test_lora_config_validation(tmp_path, lora: LoRAConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingConfig(output_dir=str(tmp_path), lora=lora).validate()
