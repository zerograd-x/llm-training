from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

import torch
from torch.utils.data import Dataset

from llm_training import (
    CheckpointConfig,
    DeepSpeedConfig,
    DistributedRuntime,
    GradientCheckpointingConfig,
    OptimizerConfig,
    SchedulerConfig,
    Trainer,
    TrainingConfig,
    load_causal_lm,
)


class Rows(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def collate(rows):
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def encode(tokenizer, prompt: str, answer: str, max_length: int = 256):
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", default="./outputs/qwen35-4b-gpu-smoke")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        raise RuntimeError("Launch with torchrun --nproc_per_node=4")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU is required")

    config = TrainingConfig(
        output_dir=args.output,
        num_epochs=1,
        per_device_batch_size=1,
        gradient_accumulation_steps=1,
        mixed_precision="bf16",
        log_every_steps=1,
        optimizer=OptimizerConfig(learning_rate=1e-5, weight_decay=0.0),
        scheduler=SchedulerConfig(name="linear", warmup_ratio=0.0),
        gradient_checkpointing=GradientCheckpointingConfig(
            enabled=True,
            use_reentrant=False,
        ),
        deepspeed=DeepSpeedConfig(zero_stage=3),
        checkpoint=CheckpointConfig(save_every_steps=1, keep_last=1),
    )

    runtime = DistributedRuntime(config)
    components = runtime.load_causal_lm(
        args.model,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )

    examples = [
        ("Classify the product. Product: men's cotton crew-neck t-shirt. Category:", " apparel"),
        ("Classify the product. Product: stainless steel frying pan, 12 inch. Category:", " cookware"),
        ("Classify the product. Product: wireless gaming mouse with RGB lighting. Category:", " electronics"),
        ("Classify the product. Product: moisturizing face cream for dry skin. Category:", " beauty"),
        ("Classify the product. Product: waterproof hiking boots for women. Category:", " footwear"),
        ("Classify the product. Product: wooden dining table for six people. Category:", " furniture"),
        ("Classify the product. Product: USB-C fast charging cable, 2 meters. Category:", " electronics"),
        ("Classify the product. Product: ceramic coffee mug, 12 oz. Category:", " kitchen"),
    ]
    rows = [encode(components.tokenizer, p, a) for p, a in examples]

    trainer = Trainer(
        model=components.model,
        tokenizer=components.tokenizer,
        train_dataset=Rows(rows),
        collate_fn=collate,
        config=config,
        runtime=runtime,
    )

    if trainer.accelerator.num_processes != 4:
        raise AssertionError(f"expected 4 ranks, got {trainer.accelerator.num_processes}")
    if trainer.accelerator.device.type != "cuda":
        raise AssertionError(f"expected CUDA, got {trainer.accelerator.device}")

    state = trainer.train()
    if state.global_step < 1:
        raise AssertionError("training did not produce an optimizer update")

    checkpoint = Path(args.output) / f"checkpoint-{state.global_step:08d}"
    if trainer.accelerator.is_main_process and not checkpoint.exists():
        raise AssertionError(f"checkpoint missing: {checkpoint}")

    # Full fine-tuning must also produce a portable Hugging Face model.
    trainer.export_model()
    trainer.accelerator.wait_for_everyone()
    final_model = Path(args.output) / "final-model"
    if trainer.accelerator.is_main_process:
        if not (final_model / "config.json").exists():
            raise AssertionError("exported config.json missing")
        if not any(final_model.glob("*.safetensors")):
            raise AssertionError("exported safetensors weights missing")

    # Free the distributed training model before reload so the test exercises
    # the exported artifact rather than relying on the in-memory trainer state.
    del trainer
    del components
    torch.cuda.empty_cache()

    reloaded = load_causal_lm(
        str(final_model),
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    reloaded.model.to(torch.cuda.current_device())
    sample = rows[0]
    with torch.no_grad():
        outputs = reloaded.model(
            input_ids=sample["input_ids"].unsqueeze(0).to(torch.cuda.current_device()),
            attention_mask=sample["attention_mask"].unsqueeze(0).to(torch.cuda.current_device()),
        )
    if not torch.isfinite(outputs.logits).all():
        raise AssertionError("reloaded exported model produced non-finite logits")

    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
    if runtime.is_main_process:
        print(
            "PASS qwen35_4b_single_node "
            f"world_size=4 global_step={state.global_step} "
            f"peak_allocated_gib={peak:.2f} export_reload_forward=ok"
        )


if __name__ == "__main__":
    main()
