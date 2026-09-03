from __future__ import annotations

import pytest
import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

from llm_training import (
    CheckpointConfig,
    GradientCheckpointingConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainableTokensConfig,
    Trainer,
    TrainingConfig,
    merge_trainable_tokens,
)


pytest.importorskip("peft")


BASE_VOCAB_SIZE = 32
EXPANDED_VOCAB_SIZE = 36
N_TOKENS_ADDED = EXPANDED_VOCAB_SIZE - BASE_VOCAB_SIZE


class TensorRows(Dataset):
    def __init__(self) -> None:
        self.rows = [
            torch.tensor([1, 32, 33, 2], dtype=torch.long),
            torch.tensor([1, 34, 35, 2], dtype=torch.long),
            torch.tensor([1, 32, 35, 2], dtype=torch.long),
            torch.tensor([1, 33, 34, 2], dtype=torch.long),
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


def build_expanded_model() -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=BASE_VOCAB_SIZE,
            n_positions=8,
            n_ctx=8,
            n_embd=16,
            n_layer=2,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
            tie_word_embeddings=True,
        )
    )
    model.resize_token_embeddings(EXPANDED_VOCAB_SIZE)
    return model


def training_config(tmp_path) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(tmp_path),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        trainable_tokens=TrainableTokensConfig(
            n_tokens_added=N_TOKENS_ADDED,
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-2,
            weight_decay=0.0,
        ),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=2),
    )


@pytest.mark.integration
def test_trainable_tokens_train_save_and_offline_merge(tmp_path) -> None:
    base_dir = tmp_path / "expanded-base"
    build_expanded_model().save_pretrained(base_dir, safe_serialization=True)

    base_model = AutoModelForCausalLM.from_pretrained(base_dir)
    base_embedding = base_model.get_input_embeddings().weight.detach().cpu().clone()
    assert base_embedding.shape[0] == EXPANDED_VOCAB_SIZE

    trainer = Trainer(
        model=GPT2LMHeadModel.from_pretrained(base_dir),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=training_config(tmp_path),
    )

    assert trainer.adaptation_summary.method == "trainable_tokens"
    assert trainer.adaptation_summary.trainable_token_ids == (32, 33, 34, 35)
    assert trainer.adaptation_summary.target_modules == ("transformer.wte",)

    model = trainer.accelerator.unwrap_model(trainer.model)
    trainable_names = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    assert trainable_names
    assert all("trainable_tokens_delta" in name for name in trainable_names)
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

    deltas_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    state = trainer.train()
    assert state.global_step == 2
    assert state.epoch == 1
    assert (tmp_path / "checkpoint-00000002" / "trainer_state.json").is_file()

    deltas_after = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert any(
        not torch.equal(deltas_before[name], deltas_after[name])
        for name in deltas_before
    )

    fixed_input = torch.tensor([[1, 32, 5, 35, 2]], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        reference_logits = model(input_ids=fixed_input).logits.detach().cpu()

    adapter_dir = trainer.save_adapter()
    assert adapter_dir == tmp_path / "adapter"
    assert (adapter_dir / "adapter_config.json").is_file()
    adapter_state_path = adapter_dir / "adapter_model.safetensors"
    assert adapter_state_path.is_file()
    assert (adapter_dir / "training_config.json").is_file()
    assert not (adapter_dir / "model.safetensors").exists()
    assert not (adapter_dir / "base_trainable_state.safetensors").exists()

    adapter_state = load_file(str(adapter_state_path), device="cpu")
    assert adapter_state
    assert all("trainable_tokens" in name for name in adapter_state)
    assert all(
        tensor.shape[0] != EXPANDED_VOCAB_SIZE
        for tensor in adapter_state.values()
        if tensor.ndim > 0
    )
    assert sum(tensor.numel() for tensor in adapter_state.values()) < base_embedding.numel()

    merged_dir = merge_trainable_tokens(
        base_dir,
        adapter_dir,
        tmp_path / "merged-model",
    )
    assert (merged_dir / "config.json").is_file()
    assert (merged_dir / "model.safetensors").is_file()
    assert not (merged_dir / "adapter_config.json").exists()

    merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
    merged_model.eval()
    with torch.no_grad():
        merged_logits = merged_model(input_ids=fixed_input).logits.detach().cpu()

    torch.testing.assert_close(merged_logits, reference_logits, rtol=1e-5, atol=1e-5)

    merged_embedding = merged_model.get_input_embeddings().weight.detach().cpu()
    torch.testing.assert_close(
        merged_embedding[:BASE_VOCAB_SIZE],
        base_embedding[:BASE_VOCAB_SIZE],
        rtol=0,
        atol=0,
    )
    assert not torch.equal(
        merged_embedding[BASE_VOCAB_SIZE:],
        base_embedding[BASE_VOCAB_SIZE:],
    )

    with pytest.raises(NotImplementedError, match="TrainableTokens uses"):
        trainer.export_model()
