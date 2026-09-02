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
        )
    )


def build_config(output_dir: Path, *, resume_from: str | None = None) -> TrainingConfig:
    return TrainingConfig(
        output_dir=str(output_dir),
        num_epochs=1,
        per_device_batch_size=2,
        seed=123,
        log_every_steps=1,
        optimizer=OptimizerConfig(
            learning_rate=1e-3,
            weight_decay=0.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
        ),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        deepspeed=DeepSpeedConfig(
            zero_stage=2,
            optimizer=DeepSpeedOptimizerConfig(
                type="AdamW",
                # Keep this CPU integration test on the PyTorch AdamW code path
                # instead of compiling DeepSpeed's fused CUDA Adam extension.
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


def run_interrupt(output_dir: Path) -> None:
    trainer = InterruptAfterFirstCheckpointTrainer(
        model=build_model(),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=build_config(output_dir),
    )

    engine_name = type(trainer.model).__name__
    optimizer_name = type(trainer.optimizer).__name__
    scheduler_name = type(trainer.scheduler).__name__

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

    (output_dir / "interrupt_result.json").write_text(
        json.dumps(
            {
                "global_step": trainer.state.global_step,
                "engine": engine_name,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
            },
            indent=2,
        )
        + "\n"
    )


def run_resume(output_dir: Path) -> None:
    trainer = Trainer(
        model=build_model(),
        train_dataset=TensorRows(),
        collate_fn=stack_collator,
        config=build_config(output_dir, resume_from="latest"),
    )

    assert trainer.state.global_step == 1
    assert trainer.state.epoch == 0
    assert trainer.state.batch_in_epoch == 1

    state = trainer.train()
    assert state.global_step == 2
    assert state.epoch == 1
    assert state.batch_in_epoch == 0
    assert (output_dir / "checkpoint-00000002" / "trainer_state.json").is_file()

    trainer.export_model()
    final_model = output_dir / "final-model"
    assert (final_model / "config.json").is_file()
    assert any(
        (final_model / filename).is_file()
        for filename in ("model.safetensors", "pytorch_model.bin")
    )

    (output_dir / "resume_result.json").write_text(
        json.dumps(
            {
                "global_step": state.global_step,
                "engine": type(trainer.model).__name__,
                "optimizer": type(trainer.optimizer).__name__,
                "scheduler": type(trainer.scheduler).__name__,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: deepspeed_cpu_worker.py <interrupt|resume> <output-dir>")
    mode = sys.argv[1]
    output_dir = Path(sys.argv[2])
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "interrupt":
        run_interrupt(output_dir)
    elif mode == "resume":
        run_resume(output_dir)
    else:
        raise SystemExit(f"unknown mode: {mode}")


if __name__ == "__main__":
    main()
