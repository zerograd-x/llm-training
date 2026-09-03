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
    GradientCheckpointingConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
)


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
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=True,
        )
    )


def lora_base_dir(output_dir: Path) -> Path:
    return output_dir / "lora-base-model"


def build_training_model(output_dir: Path, *, lora: bool) -> GPT2LMHeadModel:
    if not lora:
        return build_model()

    base_dir = lora_base_dir(output_dir)
    if not (base_dir / "config.json").is_file():
        base_dir.mkdir(parents=True, exist_ok=True)
        build_model().save_pretrained(base_dir, safe_serialization=True)
    return GPT2LMHeadModel.from_pretrained(base_dir)


def build_config(
    output_dir: Path,
    *,
    resume_from: str | None = None,
    lora: bool = False,
) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(output_dir),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        lora=LoRAConfig(rank=4, alpha=8, dropout=0.0) if lora else None,
        optimizer=OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        ),
        scheduler=SchedulerConfig(name="linear", warmup_steps=1),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=lora,
            use_reentrant=False if lora else None,
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


def run_interrupt(output_dir: Path, *, lora: bool = False) -> None:
    trainer = InterruptAfterFirstCheckpointTrainer(
        model=build_training_model(output_dir, lora=lora),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=build_config(output_dir, lora=lora),
    )

    engine_name = type(trainer.model).__name__
    optimizer_name = type(trainer.optimizer).__name__
    scheduler_name = type(trainer.scheduler).__name__
    trainable_names = _trainable_names(trainer)
    if lora:
        assert trainable_names
        assert all("lora_" in name for name in trainable_names)

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

    result_name = "lora_interrupt_result.json" if lora else "interrupt_result.json"
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


def run_resume(output_dir: Path, *, lora: bool = False) -> None:
    trainer = Trainer(
        model=build_training_model(output_dir, lora=lora),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=build_config(output_dir, resume_from="latest", lora=lora),
    )

    assert trainer.state.global_step == 1
    assert trainer.state.epoch == 0
    assert trainer.state.batch_in_epoch == 1
    trainable_names = _trainable_names(trainer)
    if lora:
        assert trainable_names
        assert all("lora_" in name for name in trainable_names)

    state = trainer.train()
    assert state.global_step == 2
    assert state.epoch == 1
    assert state.batch_in_epoch == 0
    assert (output_dir / "checkpoint-00000002" / "trainer_state.json").is_file()

    reference_logits = None
    adapter_dir = None
    if lora:
        model = trainer.accelerator.unwrap_model(trainer.model)
        fixed_input = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
        model.eval()
        with torch.no_grad():
            reference_logits = model(input_ids=fixed_input).logits.detach().cpu().tolist()
        adapter_dir = trainer.save_adapter()
        assert (adapter_dir / "adapter_config.json").is_file()
        assert (adapter_dir / "adapter_model.safetensors").is_file()
    else:
        trainer.export_model()
        final_model = output_dir / "final-model"
        assert (final_model / "config.json").is_file()
        assert any(
            (final_model / filename).is_file()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )

    result_name = "lora_resume_result.json" if lora else "resume_result.json"
    (output_dir / result_name).write_text(
        json.dumps(
            {
                "global_step": state.global_step,
                "engine": type(trainer.model).__name__,
                "optimizer": type(trainer.optimizer).__name__,
                "scheduler": type(trainer.scheduler).__name__,
                "trainable_names": trainable_names,
                "base_dir": str(lora_base_dir(output_dir)) if lora else None,
                "adapter_dir": str(adapter_dir) if adapter_dir is not None else None,
                "reference_logits": reference_logits,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: deepspeed_cpu_worker.py "
            "<interrupt|resume|lora_interrupt|lora_resume> <output-dir>"
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
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
