from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

import llm_training.model as model_module


def tiny_qwen35_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        max_position_embeddings=64,
        layer_types=["full_attention"],
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [2, 1, 1],
            "mrope_interleaved": True,
        },
    )


def test_qwen35_tiny_cpu_forward_backward() -> None:
    model = Qwen3_5ForCausalLM(tiny_qwen35_config())
    input_ids = torch.tensor(
        [[1, 11, 12, 13, 21, 22, 2], [1, 31, 32, 33, 41, 42, 2]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :4] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    assert outputs.logits.shape == (2, 7, 128)
    assert torch.isfinite(outputs.loss)

    outputs.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_qwen35_loader_routes_multimodal_checkpoint_to_text_class(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        model_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3_5"),
    )
    monkeypatch.setattr(
        Qwen3_5ForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: sentinel,
    )

    loaded = model_module._load_causal_lm_model(
        "Qwen/Qwen3.5-9B",
        {"dtype": torch.bfloat16},
    )
    assert loaded is sentinel


def test_qwen35_loader_routes_exported_text_checkpoint_to_text_class(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(
        model_module.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(model_type="qwen3_5_text"),
    )
    monkeypatch.setattr(
        Qwen3_5ForCausalLM,
        "from_pretrained",
        lambda *_args, **_kwargs: sentinel,
    )

    loaded = model_module._load_causal_lm_model("./exported-qwen35", {})
    assert loaded is sentinel
