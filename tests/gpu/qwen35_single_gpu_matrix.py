from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    FreezeConfig,
    GradientCheckpointingConfig,
    LoRAConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    adapt_column_batch_collator,
    load_causal_lm,
)


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def stack_collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def encode_sft(tokenizer, prompt: str, answer: str, max_length: int = 128):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(
        answer + tokenizer.eos_token,
        add_special_tokens=False,
    )["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    attention_mask = [1] * len(input_ids)
    pad = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * pad
    labels += [-100] * pad
    attention_mask += [0] * pad
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def build_sft_rows(tokenizer):
    examples = [
        ("Product: men's cotton crew-neck t-shirt. Category:", " apparel"),
        ("Product: stainless steel frying pan. Category:", " cookware"),
        ("Product: wireless gaming mouse. Category:", " electronics"),
        ("Product: moisturizing face cream. Category:", " beauty"),
    ]
    return [encode_sft(tokenizer, prompt, answer) for prompt, answer in examples]


def make_common(output: Path, *, max_steps: int | None = None):
    return dict(
        output_dir=str(output),
        num_epochs=1,
        max_steps=max_steps,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_every_steps=1,
        optimizer=OptimizerConfig(
            learning_rate=1e-5,
            weight_decay=0.0,
        ),
        scheduler=SchedulerConfig(
            name="linear",
            warmup_ratio=0.0,
        ),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        checkpoint=CheckpointConfig(
            save_every_steps=None,
            keep_last=1,
        ),
    )


def run_sft_case(case: str, model: str, output: Path):
    components = load_causal_lm(
        model,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    if components.tokenizer.pad_token_id is None:
        components.tokenizer.pad_token = components.tokenizer.eos_token

    kwargs = make_common(
        output,
        max_steps=3 if case == "max_steps" else None,
    )
    if case == "lora":
        kwargs["lora"] = LoRAConfig(
            rank=4,
            alpha=8,
            dropout=0.0,
            target_last_n_layers=2,
        )
    else:
        kwargs["freeze"] = FreezeConfig(
            train_last_n_layers=1,
            train_embeddings=False,
            train_lm_head=False,
        )

    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=Rows(build_sft_rows(components.tokenizer)),
        collate_fn=stack_collate,
        config=TrainingConfig(**kwargs),
    )
    state = trainer.train()

    if case == "max_steps" and state.global_step != 3:
        raise AssertionError(f"expected max_steps=3, got {state.global_step}")
    if state.global_step < 1:
        raise AssertionError("training produced no optimizer updates")

    latest_checkpoint = output / f"checkpoint-{state.global_step:08d}"
    if not latest_checkpoint.exists():
        raise AssertionError(f"checkpoint missing: {latest_checkpoint}")

    if case == "lora":
        adapter = trainer.save_adapter()
        if not (adapter / "adapter_config.json").exists():
            raise AssertionError("LoRA adapter export is missing adapter_config.json")
    elif case == "partial_bf16":
        trainer.export_model()
        final_model = output / "final-model"
        if not (final_model / "config.json").exists():
            raise AssertionError("final model export is missing config.json")

    return state.global_step


def run_pretrain_case(model: str, output: Path):
    import llm_training_data

    components = load_causal_lm(
        model,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    if components.tokenizer.pad_token_id is None:
        components.tokenizer.pad_token = components.tokenizer.eos_token

    documents = [
        "KV cache stores previously computed key and value tensors during autoregressive decoding.",
        "Tensor parallelism splits matrix computation across multiple accelerator devices.",
        "Gradient accumulation simulates a larger effective batch using multiple microbatches.",
        "Causal language modeling predicts the next token from earlier tokens.",
    ]
    rows = llm_training_data.build_causal_lm_sequences(
        documents,
        tokenizer=components.tokenizer,
        max_sequence_length=128,
    )
    if not rows:
        raise AssertionError("causal LM sequence builder returned no rows")
    if any(row["input_ids"] != row["labels"] for row in rows):
        raise AssertionError("plain causal LM labels must equal input_ids before shift")

    collate_fn = adapt_column_batch_collator(
        llm_training_data.build_causal_lm_collator(
            components.tokenizer,
            max_sequence_length=128,
        )
    )
    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=Rows(rows),
        collate_fn=collate_fn,
        config=TrainingConfig(
            **make_common(output, max_steps=2),
            freeze=FreezeConfig(
                train_last_n_layers=1,
                train_embeddings=False,
                train_lm_head=False,
            ),
        ),
    )
    state = trainer.train()
    if state.global_step != 2:
        raise AssertionError(f"expected 2 pretraining steps, got {state.global_step}")
    checkpoint = output / f"checkpoint-{state.global_step:08d}"
    if not checkpoint.exists():
        raise AssertionError(f"pretraining checkpoint missing: {checkpoint}")
    return state.global_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=("partial_bf16", "lora", "max_steps", "pretrain_data"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output-root", default="./outputs/qwen35-gpu-smoke")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"single-GPU smoke expects exactly 1 visible GPU, got {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU is required")

    output = Path(args.output_root) / args.case
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    if args.case == "pretrain_data":
        steps = run_pretrain_case(args.model, output)
    else:
        steps = run_sft_case(args.case, args.model, output)

    peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)
    payload = {
        "ok": True,
        "case": args.case,
        "model": args.model,
        "global_step": steps,
        "gpu": torch.cuda.get_device_name(0),
        "peak_allocated_gib": round(peak_gib, 3),
    }
    print(json.dumps(payload, indent=2))
    print(
        f"PASS {args.case} model={args.model} "
        f"steps={steps} peak_allocated_gib={peak_gib:.2f}"
    )


if __name__ == "__main__":
    main()
