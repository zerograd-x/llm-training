from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import torch
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    DeepSpeedConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    load_causal_lm,
)


class CategoryRows(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int):
        self.rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            prompt = (
                f"System: {record['system']}\n"
                f"User: {record['user']}\n"
                "Assistant:"
            )
            answer = " " + record["assistant"] + tokenizer.eos_token
            prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
            answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            input_ids = (prompt_ids + answer_ids)[:max_length]
            labels = [-100] * min(len(prompt_ids), len(input_ids))
            labels += input_ids[len(labels):]
            attention_mask = [1] * len(input_ids)
            pad = max_length - len(input_ids)
            input_ids += [tokenizer.pad_token_id] * pad
            attention_mask += [0] * pad
            labels += [-100] * pad
            self.rows.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def sync_to_s3(local_dir: Path, s3_uri: str) -> None:
    subprocess.run(
        ["aws", "s3", "sync", str(local_dir), s3_uri.rstrip("/") + "/"],
        check=True,
    )


class S3SyncTrainer(Trainer):
    def __init__(self, *args, artifact_s3: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_s3 = artifact_s3

    def save_checkpoint(self) -> Path:
        checkpoint = super().save_checkpoint()
        self.accelerator.wait_for_everyone()
        if self.artifact_s3 and self.accelerator.is_main_process:
            sync_to_s3(Path(self.config.output_dir), self.artifact_s3)
        self.accelerator.wait_for_everyone()
        return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", type=Path, default=Path("./qwen35-4b-category-sft"))
    parser.add_argument("--artifact-s3", default=None)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1 and not torch.cuda.is_available():
        raise RuntimeError("Distributed Qwen3.5 SFT requires CUDA")

    components = load_causal_lm(
        args.model,
        model_kwargs={"dtype": torch.bfloat16},
    )
    dataset = CategoryRows(args.data, components.tokenizer, args.max_length)

    config = TrainingConfig(
        output_dir=str(args.output),
        num_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
        freeze=FreezeConfig(
            train_last_n_layers=2,
            train_embeddings=False,
            train_lm_head=True,
        ),
        optimizer=OptimizerConfig(learning_rate=2e-5, weight_decay=0.1),
        scheduler=SchedulerConfig(name="cosine", warmup_ratio=0.03),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        deepspeed=DeepSpeedConfig(zero_stage=3),
        checkpoint=CheckpointConfig(save_every_steps=args.save_every, keep_last=2),
    )

    trainer = S3SyncTrainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=dataset,
        collate_fn=collate,
        config=config,
        artifact_s3=args.artifact_s3,
    )
    state = trainer.train()
    trainer.export_model()
    trainer.accelerator.wait_for_everyone()

    if args.artifact_s3 and trainer.accelerator.is_main_process:
        sync_to_s3(args.output, args.artifact_s3)
    trainer.accelerator.wait_for_everyone()

    if trainer.accelerator.is_main_process:
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
        print(
            f"Qwen3.5-4B SFT complete: world_size={trainer.accelerator.num_processes} "
            f"global_step={state.global_step} peak_allocated_gib={peak_gib:.2f} "
            "trainable=last_2_layers+lm_head"
        )


if __name__ == "__main__":
    main()
