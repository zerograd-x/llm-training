from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from .config_file import DataConfig, TrainingJobConfig, load_training_job
from .distributed import DistributedRuntime, active_world_size
from .inspection import build_training_plan, format_training_plan
from .trainer import Trainer


class JsonlDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError("Dataset must contain at least one non-empty JSON row")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def load_jsonl(path: str) -> JsonlDataset:
    rows: list[dict[str, Any]] = []
    try:
        handle = Path(path).open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Cannot read dataset {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Dataset row {line_number} must be a JSON object")
            rows.append(row)
    return JsonlDataset(rows)


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset field {field!r} must be a non-empty string")
    return value


def validate_dataset(dataset: JsonlDataset, data: DataConfig) -> None:
    for row in dataset.rows:
        if data.format == "sft":
            _require_text(row, data.prompt_field)
            _require_text(row, data.completion_field)
        else:
            _require_text(row, data.text_field)


def build_collator(tokenizer, data: DataConfig):
    def collate(rows: list[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
        encoded: list[tuple[list[int], list[int]]] = []
        for row in rows:
            if data.format == "sft":
                prompt = _require_text(row, data.prompt_field)
                completion = _require_text(row, data.completion_field)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                suffix = completion + (tokenizer.eos_token or "")
                completion_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                input_ids = (prompt_ids + completion_ids)[: data.max_sequence_length]
                labels = ([-100] * len(prompt_ids) + completion_ids)[: data.max_sequence_length]
            else:
                text = _require_text(row, data.text_field) + (tokenizer.eos_token or "")
                input_ids = tokenizer(text, add_special_tokens=False)["input_ids"][: data.max_sequence_length]
                labels = list(input_ids)
            if not input_ids:
                raise ValueError("Tokenization produced an empty training sequence")
            if all(label == -100 for label in labels):
                raise ValueError(
                    "SFT truncation removed every completion token; shorten the prompt "
                    "or increase data.max_sequence_length"
                )
            encoded.append((list(input_ids), list(labels)))

        width = max(len(input_ids) for input_ids, _ in encoded)
        batch_ids, batch_labels, batch_mask = [], [], []
        for input_ids, labels in encoded:
            padding = width - len(input_ids)
            batch_ids.append(input_ids + [tokenizer.pad_token_id] * padding)
            batch_labels.append(labels + [-100] * padding)
            batch_mask.append([1] * len(input_ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def _model_kwargs(job: TrainingJobConfig) -> dict[str, Any]:
    kwargs = dict(job.model.model_kwargs)
    if job.model.dtype != "auto":
        kwargs.setdefault("torch_dtype", getattr(torch, job.model.dtype))
    kwargs.setdefault("trust_remote_code", job.model.trust_remote_code)
    return kwargs


def _tokenizer_kwargs(job: TrainingJobConfig) -> dict[str, Any]:
    kwargs = dict(job.model.tokenizer_kwargs)
    kwargs.setdefault("trust_remote_code", job.model.trust_remote_code)
    return kwargs


def run_job(
    job: TrainingJobConfig,
    *,
    train_dataset: JsonlDataset | None = None,
    eval_dataset: JsonlDataset | None = None,
):
    runtime = DistributedRuntime(job.training)
    components = runtime.load_causal_lm(
        job.model.name_or_path,
        tokenizer_name_or_path=job.model.tokenizer_name_or_path,
        model_kwargs=_model_kwargs(job),
        tokenizer_kwargs=_tokenizer_kwargs(job),
    )
    train_dataset = train_dataset or load_jsonl(job.data.train_file)
    if eval_dataset is None and job.data.validation_file is not None:
        eval_dataset = load_jsonl(job.data.validation_file)
    collator = build_collator(components.tokenizer, job.data)
    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=train_dataset,
        collate_fn=collator,
        config=job.training,
        runtime=runtime,
        eval_dataset=eval_dataset,
        eval_collate_fn=collator,
    )
    state = trainer.train()

    save = job.artifacts.save
    if save == "auto":
        save = "adapter" if job.training.lora or job.training.trainable_tokens else "model"
    if save == "adapter":
        trainer.save_adapter()
    elif save == "model":
        trainer.export_model()
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="llm-train")
    parser.add_argument("--config", required=True, help="JSON or YAML job configuration")
    parser.add_argument("--dry-run", action="store_true", help="validate data/config and print the plan without loading a model")
    args = parser.parse_args(argv)
    try:
        job = load_training_job(args.config)
        train_dataset = load_jsonl(job.data.train_file)
        eval_dataset = (
            load_jsonl(job.data.validation_file)
            if job.data.validation_file is not None
            else None
        )
        validate_dataset(train_dataset, job.data)
        if eval_dataset is not None:
            validate_dataset(eval_dataset, job.data)
        plan = build_training_plan(
            job.training,
            train_examples=len(train_dataset),
            world_size=active_world_size(),
            model_name=job.model.name_or_path,
            max_sequence_length=job.data.max_sequence_length,
        )
        print(format_training_plan(plan))
        print(f"train data: {job.data.train_file}")
        if args.dry_run:
            return 0
        state = run_job(
            job,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        print(f"training completed at global_step={state.global_step}")
        return 0
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"llm-train: error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
