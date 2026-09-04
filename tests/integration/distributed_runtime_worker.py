from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from llm_training import (
    CheckpointConfig,
    DistributedRuntime,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    require_single_process,
)


class IdDataset(Dataset):
    def __len__(self) -> int:
        return 8

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.tensor(index, dtype=torch.long)


class TinyLMDataset(Dataset):
    def __init__(self) -> None:
        self.rows = [
            torch.tensor([1, 3, 4, 1], dtype=torch.long),
            torch.tensor([1, 4, 5, 1], dtype=torch.long),
            torch.tensor([1, 5, 6, 1], dtype=torch.long),
            torch.tensor([1, 6, 7, 1], dtype=torch.long),
            torch.tensor([1, 7, 3, 1], dtype=torch.long),
            torch.tensor([1, 3, 5, 1], dtype=torch.long),
            torch.tensor([1, 4, 6, 1], dtype=torch.long),
            torch.tensor([1, 5, 7, 1], dtype=torch.long),
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


def build_local_hf_artifact(destination: Path, *, owner_rank: int) -> None:
    destination.mkdir(parents=True, exist_ok=True)

    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "<unk>": 2,
        "a": 3,
        "b": 4,
        "c": 5,
        "d": 6,
        "e": 7,
    }
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(destination)

    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(vocab),
            n_positions=8,
            n_ctx=8,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=0,
        )
    )
    model.save_pretrained(destination, safe_serialization=True)
    (destination / "materialized_by_rank.txt").write_text(f"{owner_rank}\n")


class RaceAmplifyingTrainer(Trainer):
    """Expose a final-checkpoint directory before rank zero writes metadata."""

    def save_checkpoint(self):
        if self.runtime.is_main_process:
            checkpoint_dir = Path(self.config.output_dir) / (
                f"checkpoint-{self.state.global_step:08d}"
            )
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            time.sleep(0.25)
        return super().save_checkpoint()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: distributed_runtime_worker.py <output-dir>")

    root = Path(sys.argv[1])
    root.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(
        output_dir=str(root / "training"),
        num_epochs=1,
        per_device_batch_size=2,
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=1e-2, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        checkpoint=CheckpointConfig(save_every_steps=None, keep_last=1),
    )
    runtime = DistributedRuntime(config)

    model_dir = root / "node-local-model"
    runtime.materialize_path(
        model_dir,
        lambda destination: build_local_hf_artifact(
            destination,
            owner_rank=runtime.process_index,
        ),
        scope="local",
    )

    components = runtime.load_causal_lm(str(model_dir))
    parameter_count = sum(parameter.numel() for parameter in components.model.parameters())

    dataloader = DataLoader(IdDataset(), batch_size=2, shuffle=False)
    prepared_loader = runtime.accelerator.prepare(dataloader)
    seen_ids: list[int] = []
    for batch in prepared_loader:
        seen_ids.extend(int(value) for value in batch.detach().cpu().tolist())

    trainer = RaceAmplifyingTrainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=TinyLMDataset(),
        collate_fn=stack_collator,
        config=config,
        runtime=runtime,
    )
    state = trainer.train()
    final_lr = float(trainer.optimizer.param_groups[0]["lr"])

    if runtime.is_main_process:
        (root / "global-main.txt").write_text(f"{runtime.process_index}\n")
    if runtime.is_local_main_process:
        (root / f"local-main-{runtime.process_index}.txt").write_text(
            f"{runtime.local_process_index}\n"
        )

    driver_guard_error = None
    try:
        require_single_process("offline merge")
    except RuntimeError as exc:
        driver_guard_error = str(exc)

    result = {
        "process_index": runtime.process_index,
        "local_process_index": runtime.local_process_index,
        "num_processes": runtime.num_processes,
        "is_main_process": runtime.is_main_process,
        "is_local_main_process": runtime.is_local_main_process,
        "seen_ids": seen_ids,
        "parameter_count": parameter_count,
        "global_step": state.global_step,
        "final_lr": final_lr,
        "driver_guard_error": driver_guard_error,
    }
    (root / f"rank-{runtime.process_index}.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    runtime.wait_for_everyone()


if __name__ == "__main__":
    main()
