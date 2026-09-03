from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

from llm_training import (
    CheckpointConfig,
    DeepSpeedConfig,
    DeepSpeedOptimizerConfig,
    DeepSpeedSchedulerConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainableTokensConfig,
    Trainer,
    TrainingConfig,
)


BASE_VOCAB_SIZE = 32
EXPANDED_VOCAB_SIZE = 36


class TensorRows(Dataset):
    def __init__(self, *, trainable_tokens: bool = False) -> None:
        if trainable_tokens:
            self.rows = [
                torch.tensor([1, 32, 33, 2], dtype=torch.long),
                torch.tensor([1, 34, 35, 2], dtype=torch.long),
                torch.tensor([1, 32, 35, 2], dtype=torch.long),
                torch.tensor([1, 33, 34, 2], dtype=torch.long),
            ]
        else:
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


def build_model(*, expanded_vocab: bool = False) -> GPT2LMHeadModel:
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=BASE_VOCAB_SIZE,
            n_positions=8,
            n_ctx=8,
            n_embd=16,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
            tie_word_embeddings=True,
        )
    )
    if expanded_vocab:
        model.resize_token_embeddings(EXPANDED_VOCAB_SIZE)
    return model


def adapter_base_dir(output_dir: Path) -> Path:
    return output_dir / "adapter-base-model"


def build_training_model(
    output_dir: Path,
    *,
    adapter_training: bool,
    trainable_tokens: bool,
) -> GPT2LMHeadModel:
    if not adapter_training:
        return build_model()

    base_dir = adapter_base_dir(output_dir)
    if not (base_dir / "config.json").is_file():
        base_dir.mkdir(parents=True, exist_ok=True)
        build_model(expanded_vocab=trainable_tokens).save_pretrained(
            base_dir,
            safe_serialization=True,
        )
    return GPT2LMHeadModel.from_pretrained(base_dir)


def build_config(
    output_dir: Path,
    *,
    resume_from: str | None = None,
    lora: bool = False,
    hybrid: bool = False,
    trainable_tokens: bool = False,
) -> TrainingConfig:
    enabled_modes = sum((lora, hybrid, trainable_tokens))
    if enabled_modes > 1:
        raise ValueError("adapter test modes are mutually exclusive")
    adapter_training = bool(enabled_modes)

    freeze = None
    if hybrid:
        freeze = FreezeConfig(
            train_last_n_layers=1,
            train_embeddings=False,
            train_lm_head=False,
        )

    return TrainingConfig(
        output_dir=str(output_dir),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        freeze=freeze,
        lora=LoRAConfig(rank=4, alpha=8, dropout=0.0) if (lora or hybrid) else None,
        trainable_tokens=(
            TrainableTokensConfig(
                n_tokens_added=EXPANDED_VOCAB_SIZE - BASE_VOCAB_SIZE,
            )
            if trainable_tokens
            else None
        ),
        optimizer=OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        ),
        scheduler=SchedulerConfig(name="linear", warmup_steps=1),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=adapter_training,
            use_reentrant=False if adapter_training else None,
        ),
        deepspeed=DeepSpeedConfig(
            zero_stage=2,
            optimizer=DeepSpeedOptimizerConfig(
                type="AdamW",
                params={"torch_adam": True},
            ),
            scheduler=DeepSpeedSchedulerConfig(type="WarmupDecayLR"),
        ),
        checkpoint=CheckpointConfig(
            save_every_steps=1,
            keep_last=2,
            resume_from=resume_from,
        ),
    )


class IntentionalInterruption(RuntimeError):
    pass


class InterruptAfterFirstCheckpointTrainer(Trainer):
    def save_checkpoint(self):
        checkpoint_dir = super().save_checkpoint()
        if self.state.global_step == 1:
            raise IntentionalInterruption("simulated process failure after checkpoint 1")
        return checkpoint_dir


def _trainable_names(trainer: Trainer) -> list[str]:
    model = trainer.accelerator.unwrap_model(trainer.model)
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def _mode_name(*, lora: bool, hybrid: bool, trainable_tokens: bool) -> str:
    if hybrid:
        return "hybrid_lora"
    if lora:
        return "lora"
    if trainable_tokens:
        return "trainable_tokens"
    return "full"


def _assert_adapter_trainables(
    trainable_names: list[str],
    *,
    hybrid: bool,
    trainable_tokens: bool,
) -> None:
    assert trainable_names
    if trainable_tokens:
        assert all("trainable_tokens_delta" in name for name in trainable_names)
        return

    assert any("lora_" in name for name in trainable_names)
    if hybrid:
        assert any("lora_" not in name for name in trainable_names)
    else:
        assert all("lora_" in name for name in trainable_names)


def run_interrupt(
    output_dir: Path,
    *,
    lora: bool = False,
    hybrid: bool = False,
    trainable_tokens: bool = False,
) -> None:
    adapter_training = lora or hybrid or trainable_tokens
    trainer = InterruptAfterFirstCheckpointTrainer(
        model=build_training_model(
            output_dir,
            adapter_training=adapter_training,
            trainable_tokens=trainable_tokens,
        ),
        train_dataset=TensorRows(trainable_tokens=trainable_tokens),
        collate_fn=stack_collator,
        config=build_config(
            output_dir,
            lora=lora,
            hybrid=hybrid,
            trainable_tokens=trainable_tokens,
        ),
    )

    engine_name = type(trainer.model).__name__
    optimizer_name = type(trainer.optimizer).__name__
    scheduler_name = type(trainer.scheduler).__name__
    trainable_names = _trainable_names(trainer)
    if adapter_training:
        _assert_adapter_trainables(
            trainable_names,
            hybrid=hybrid,
            trainable_tokens=trainable_tokens,
        )

    try:
        trainer.train()
    except IntentionalInterruption:
        pass
    else:
        raise AssertionError("expected the first DeepSpeed run to stop after checkpoint 1")

    assert trainer.state.global_step == 1
    assert trainer.state.epoch == 0
    assert trainer.state.batch_in_epoch == 1
    checkpoint = output_dir / "checkpoint-00000001"
    assert (checkpoint / "trainer_state.json").is_file()

    mode = _mode_name(
        lora=lora,
        hybrid=hybrid,
        trainable_tokens=trainable_tokens,
    )
    result_name = "interrupt_result.json" if mode == "full" else f"{mode}_interrupt_result.json"
    (output_dir / result_name).write_text(
        json.dumps(
            {
                "global_step": trainer.state.global_step,
                "engine": engine_name,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
                "trainable_names": trainable_names,
            },
            indent=2,
        )
        + "\n"
    )


def run_resume(
    output_dir: Path,
    *,
    lora: bool = False,
    hybrid: bool = False,
    trainable_tokens: bool = False,
) -> None:
    adapter_training = lora or hybrid or trainable_tokens
    trainer = Trainer(
        model=build_training_model(
            output_dir,
            adapter_training=adapter_training,
            trainable_tokens=trainable_tokens,
        ),
        train_dataset=TensorRows(trainable_tokens=trainable_tokens),
        collate_fn=stack_collator,
        config=build_config(
            output_dir,
            resume_from="latest",
            lora=lora,
            hybrid=hybrid,
            trainable_tokens=trainable_tokens,
        ),
    )

    assert trainer.state.global_step == 1
    assert trainer.state.epoch == 0
    assert trainer.state.batch_in_epoch == 1
    trainable_names = _trainable_names(trainer)
    if adapter_training:
        _assert_adapter_trainables(
            trainable_names,
            hybrid=hybrid,
            trainable_tokens=trainable_tokens,
        )

    state = trainer.train()
    assert state.global_step == 2
    assert state.epoch == 1
    assert state.batch_in_epoch == 0
    assert (output_dir / "checkpoint-00000002" / "trainer_state.json").is_file()

    reference_logits = None
    adapter_dir = None
    if adapter_training:
        model = trainer.accelerator.unwrap_model(trainer.model)
        fixed_input = (
            torch.tensor([[1, 32, 5, 35, 2]], dtype=torch.long)
            if trainable_tokens
            else torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
        )
        model.eval()
        with torch.no_grad():
            reference_logits = model(input_ids=fixed_input).logits.detach().cpu().tolist()
        adapter_dir = trainer.save_adapter()
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()
        base_state_path = adapter_dir / "base_trainable_state.safetensors"
        if hybrid:
            assert base_state_path.is_file()
        else:
            assert not base_state_path.exists()
    else:
        trainer.export_model()
        final_model = output_dir / "final-model"
        assert (final_model / "config.json").is_file()
        assert any(
            (final_model / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )

    mode = _mode_name(
        lora=lora,
        hybrid=hybrid,
        trainable_tokens=trainable_tokens,
    )
    result_name = "resume_result.json" if mode == "full" else f"{mode}_resume_result.json"
    (output_dir / result_name).write_text(
        json.dumps(
            {
                "global_step": state.global_step,
                "engine": type(trainer.model).__name__,
                "optimizer": type(trainer.optimizer).__name__,
                "scheduler": type(trainer.scheduler).__name__,
                "trainable_names": trainable_names,
                "base_dir": str(adapter_base_dir(output_dir)) if adapter_training else None,
                "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
                "reference_logits": reference_logits,
                "fixed_input": fixed_input.tolist() if adapter_training else None,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: deepspeed_cpu_worker.py "
            "<interrupt|resume|lora_interrupt|lora_resume|"
            "hybrid_lora_interrupt|hybrid_lora_resume|"
            "trainable_tokens_interrupt|trainable_tokens_resume> <output-dir>"
        )
    mode = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "interrupt":
        run_interrupt(output_dir)
    elif mode == "resume":
        run_resume(output_dir)
    elif mode == "lora_interrupt":
        run_interrupt(output_dir, lora=True)
    elif mode == "lora_resume":
        run_resume(output_dir, lora=True)
    elif mode == "hybrid_lora_interrupt":
        run_interrupt(output_dir, hybrid=True)
    elif mode == "hybrid_lora_resume":
        run_resume(output_dir, hybrid=True)
    elif mode == "trainable_tokens_interrupt":
        run_interrupt(output_dir, trainable_tokens=True)
    elif mode == "trainable_tokens_resume":
        run_resume(output_dir, trainable_tokens=True)
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
