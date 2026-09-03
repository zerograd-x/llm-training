from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from llm_training import (
    CheckpointConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    adapt_model_for_training,
    merge_lora,
)


pytest.importorskip("peft")


class TensorRows(Dataset):
    def __init__(self) -> None:
        self.rows = [
            torch.tensor([1, 4, 5, 2], dtype=torch.long),
            torch.tensor([1, 6, 7, 2], dtype=torch.long),
            torch.tensor([1, 8, 9, 2], dtype=torch.long),
            torch.tensor([1, 10, 11, 2], dtype=torch.long),
        ]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids = self.rows[index]
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids.clone(),
        }


def stack_collator(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([row[key] for row in rows])
        for key in rows[0]
    }


def build_model() -> GPT2LMHeadModel:
    return GPT2LMHeadModel(
        GPT2Config(
            vocab_size=32,
            n_positions=8,
            n_ctx=8,
            n_embd=16,
            n_layer=2,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
        )
    )


def snapshot(model, *, trainable: bool) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad is trainable
    }


def hybrid_freeze_config() -> FreezeConfig:
    return FreezeConfig(
        train_last_n_layers=1,
        train_embeddings=False,
        train_lm_head=False,
    )


def lora_training_config(tmp_path, *, freeze=None) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        freeze=freeze,
        lora=LoRAConfig(rank=4, alpha=8, dropout=0.0),
        optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2),
    )


@pytest.mark.integration
def test_real_peft_lora_train_save_adapter_and_offline_merge(tmp_path) -> None:
    base_dir = tmp_path / "base-model"
    build_model().save_pretrained(base_dir, safe_serialization=True)

    trainer = Trainer(
        model=GPT2LMHeadModel.from_pretrained(base_dir),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=lora_training_config(tmp_path),
    )

    assert trainer.adaptation_summary.method == "lora"
    assert trainer.adaptation_summary.target_modules
    assert any(
        target.endswith("transformer.h.0.attn.c_attn")
        for target in trainer.adaptation_summary.target_modules
    )

    model = trainer.accelerator.unwrap_model(trainer.model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("lora_" in name for name in trainable_names)
    assert model.config.use_cache is False

    optimizer_parameter_ids = {
        id(parameter)
        for group in trainer.optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert optimizer_parameter_ids == trainable_parameter_ids

    trainable_before = snapshot(model, trainable=True)
    frozen_before = snapshot(model, trainable=False)

    state = trainer.train()

    assert state.global_step == 2
    assert state.epoch == 1
    assert (tmp_path / "checkpoint-00000002" / "trainer_state.json").is_file()

    trainable_after = snapshot(model, trainable=True)
    frozen_after = snapshot(model, trainable=False)
    assert any(
        not torch.equal(trainable_before[name], trainable_after[name])
        for name in trainable_before
    )
    for name, expected in frozen_before.items():
        torch.testing.assert_close(frozen_after[name], expected, rtol=0, atol=0)

    fixed_input = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        reference_logits = model(input_ids=fixed_input).logits.detach().cpu()

    adapter_dir = trainer.save_adapter()
    assert adapter_dir == tmp_path / "adapter"
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert (adapter_dir / "training_config.json").is_file()
    assert not (adapter_dir / "base_trainable_state.safetensors").exists()
    assert not (adapter_dir / "model.safetensors").exists()

    merged_dir = merge_lora(base_dir, adapter_dir, tmp_path / "merged-model")
    assert (merged_dir / "config.json").is_file()
    assert (merged_dir / "model.safetensors").is_file()
    assert not (merged_dir / "adapter_config.json").exists()

    merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
    merged_model.eval()
    with torch.no_grad():
        merged_logits = merged_model(input_ids=fixed_input).logits.detach().cpu()

    torch.testing.assert_close(merged_logits, reference_logits, rtol=1e-5, atol=1e-5)

    with pytest.raises(NotImplementedError, match="separate artifact lifecycle"):
        trainer.export_model()


@pytest.mark.integration
def test_freeze_config_restores_explicit_full_train_base_parameters_after_peft() -> None:
    model, summary = adapt_model_for_training(
        build_model(),
        freeze=hybrid_freeze_config(),
        lora=LoRAConfig(rank=4, alpha=8),
    )

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert summary.method == "lora"
    assert summary.full_train_base_parameters > 0
    assert summary.full_train_base_parameter_names
    assert len(summary.full_train_base_parameter_names) == len(
        summary.full_train_base_wrapped_parameter_names
    )
    assert any(
        name.startswith("transformer.h.1.")
        for name in summary.full_train_base_parameter_names
    )
    assert not any(
        name.startswith("transformer.h.0.")
        for name in summary.full_train_base_parameter_names
    )
    assert any("lora_" in name for name in trainable_names)
    assert any(
        "transformer.h.1" in name and "lora_" not in name
        for name in trainable_names
    )
    assert not any(
        "transformer.h.0" in name and "lora_" not in name
        for name in trainable_names
    )


@pytest.mark.integration
def test_hybrid_lora_saves_base_overrides_and_merges_numerically(tmp_path) -> None:
    from safetensors.torch import load_file

    base_dir = tmp_path / "hybrid-base-model"
    build_model().save_pretrained(base_dir, safe_serialization=True)

    trainer = Trainer(
        model=GPT2LMHeadModel.from_pretrained(base_dir),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=lora_training_config(
            tmp_path,
            freeze=hybrid_freeze_config(),
        ),
    )

    model = trainer.accelerator.unwrap_model(trainer.model)
    full_base_wrapped_names = set(
        trainer.adaptation_summary.full_train_base_wrapped_parameter_names
    )
    full_base_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in full_base_wrapped_names
    }
    assert full_base_before

    state = trainer.train()
    assert state.global_step == 2

    full_base_after = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in full_base_wrapped_names
    }
    assert any(
        not torch.equal(full_base_before[name], full_base_after[name])
        for name in full_base_before
    )

    fixed_input = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        reference_logits = model(input_ids=fixed_input).logits.detach().cpu()

    adapter_dir = trainer.save_adapter()
    base_state_path = adapter_dir / "base_trainable_state.safetensors"
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()
    assert base_state_path.is_file()

    base_state = load_file(str(base_state_path), device="cpu")
    assert set(base_state) == set(
        trainer.adaptation_summary.full_train_base_parameter_names
    )
    assert base_state
    assert all(not name.startswith("base_model.") for name in base_state)
    assert all(".base_layer." not in name for name in base_state)

    merged_dir = merge_lora(
        base_dir,
        adapter_dir,
        tmp_path / "hybrid-merged-model",
    )
    merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
    merged_model.eval()
    with torch.no_grad():
        merged_logits = merged_model(input_ids=fixed_input).logits.detach().cpu()

    torch.testing.assert_close(merged_logits, reference_logits, rtol=1e-5, atol=1e-5)
