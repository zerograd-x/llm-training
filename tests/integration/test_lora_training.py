from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

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


@pytest.mark.integration
def test_real_peft_lora_trains_only_adapter_parameters_with_gradient_checkpointing(
    tmp_path,
) -> None:
    config = TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        lora=LoRAConfig(rank=4, alpha=8, dropout=0.0),
        optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2),
    )

    trainer = Trainer(
        model=build_model(),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=config,
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

    with pytest.raises(NotImplementedError, match="Adapter-only and merged export"):
        trainer.export_model()


@pytest.mark.integration
def test_freeze_config_restores_explicit_full_train_base_parameters_after_peft() -> None:
    model, summary = adapt_model_for_training(
        build_model(),
        freeze=FreezeConfig(
            train_last_n_layers=1,
            train_embeddings=False,
            train_lm_head=False,
        ),
        lora=LoRAConfig(rank=4, alpha=8),
    )

    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert summary.method == "lora"
    assert summary.full_train_base_parameters > 0
    assert any("lora_" in name for name in trainable_names)
    assert any(
        "transformer.h.1" in name and "lora_" not in name
        for name in trainable_names
    )
    assert not any(
        "transformer.h.0" in name and "lora_" not in name
        for name in trainable_names
    )
