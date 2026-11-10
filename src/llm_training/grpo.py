from __future__ import annotations

"""TRL-backed Group Relative Policy Optimization for config-driven jobs."""

from dataclasses import asdict
from functools import partial, update_wrapper
import importlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from .adapters import adapt_model_for_training
from .backend import configure_gradient_checkpointing
from .config import GRPORewardConfig
from .config_file import DataConfig, TrainingJobConfig
from .dpo import dpo_deepspeed_config
from .model import load_causal_lm


def _require_grpo_dependencies() -> None:
    missing = [
        package
        for package in ("datasets", "trl")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ImportError(
            "GRPO training requires "
            + ", ".join(missing)
            + ". Install llm-training with the 'grpo' extra."
        )


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset field {field!r} must be a non-empty string")
    return value


def _render_prompt(
    prompt: str,
    *,
    tokenizer=None,
    system_prompt: str | None = None,
) -> str:
    if system_prompt is None:
        return prompt
    if tokenizer is None:
        raise ValueError("A tokenizer is required when grpo.system_prompt is set")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError(
            "grpo.system_prompt requires a tokenizer with apply_chat_template()"
        )
    return apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_prompt_rows(
    rows: Sequence[Mapping[str, Any]],
    data: DataConfig,
    *,
    tokenizer=None,
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Normalize the prompt column while preserving reward-owned metadata."""

    if data.format != "grpo":
        raise ValueError("prepare_prompt_rows requires data.format='grpo'")
    prepared: list[dict[str, Any]] = []
    for row in rows:
        normalized = dict(row)
        normalized["prompt"] = _render_prompt(
            _require_text(row, data.prompt_field),
            tokenizer=tokenizer,
            system_prompt=system_prompt,
        )
        prepared.append(normalized)
    if not prepared:
        raise ValueError("GRPO dataset must contain at least one example")
    return prepared


def _resolve_symbol(module_path: str):
    module_name, separator, symbol_name = module_path.rpartition(".")
    if not separator:
        raise ValueError(
            "Reward module_path must be a dotted path such as "
            "'my_package.rewards.correctness_reward'"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ImportError(
            f"Reward symbol {symbol_name!r} was not found in {module_name!r}"
        ) from exc


def _name_of_reward(reward: Callable[..., Any], module_path: str) -> str:
    name = getattr(reward, "__name__", None)
    return name if isinstance(name, str) and name else module_path.rsplit(".", 1)[-1]


def _normalize_reward_values(
    values: Any,
    *,
    reward_name: str,
    abstain: str,
) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    if not isinstance(values, (list, tuple)):
        raise TypeError(
            f"Reward {reward_name!r} must return a list, tuple, or tensor"
        )
    normalized: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            if abstain == "zero":
                normalized.append(0.0)
                continue
            raise ValueError(
                f"Reward {reward_name!r} abstained at row {index}; "
                "set abstain='zero' explicitly to treat abstentions as zero"
            )
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise TypeError(
                    f"Reward {reward_name!r} returned a non-scalar tensor "
                    f"at row {index}"
                )
            value = value.item()
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(
                f"Reward {reward_name!r} returned a non-finite numeric value "
                f"at row {index}"
            )
        normalized.append(float(value))
    return normalized


def load_reward(config: GRPORewardConfig) -> Callable[..., list[float]]:
    """Load and wrap one reward function with explicit abstention semantics."""

    target = _resolve_symbol(config.module_path)
    if isinstance(target, type):
        reward = target(**config.kwargs)
    elif callable(target):
        reward = partial(target, **config.kwargs) if config.kwargs else target
        if isinstance(reward, partial):
            update_wrapper(reward, target)
    else:
        raise TypeError(f"Reward target {config.module_path!r} is not callable")
    if not callable(reward):
        raise TypeError(f"Reward target {config.module_path!r} is not callable")

    reward_name = _name_of_reward(reward, config.module_path)

    def checked_reward(*args, **kwargs) -> list[float]:
        return _normalize_reward_values(
            reward(*args, **kwargs),
            reward_name=reward_name,
            abstain=config.abstain,
        )

    checked_reward.__name__ = reward_name
    checked_reward.__doc__ = getattr(reward, "__doc__", None)
    return checked_reward


def grpo_training_arguments(job: TrainingJobConfig) -> dict[str, Any]:
    """Build kwargs for trl.GRPOConfig without importing optional dependencies."""

    if job.grpo is None:
        raise ValueError("GRPO parameters are missing")
    training = job.training
    grpo = job.grpo
    evaluation = training.evaluation
    has_eval = job.data.validation_file is not None

    steps_per_generation = (
        grpo.steps_per_generation
        if grpo.steps_per_generation is not None
        else training.gradient_accumulation_steps
    )
    generation_rows = training.per_device_batch_size * steps_per_generation
    if generation_rows % grpo.num_generations != 0:
        raise ValueError(
            "training.per_device_batch_size * effective steps_per_generation "
            "must be divisible by grpo.num_generations"
        )

    kwargs: dict[str, Any] = {
        "output_dir": training.output_dir,
        "num_train_epochs": training.num_epochs,
        "per_device_train_batch_size": training.per_device_batch_size,
        "per_device_eval_batch_size": (
            evaluation.batch_size or training.per_device_batch_size
        ),
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "learning_rate": training.optimizer.learning_rate,
        "weight_decay": training.optimizer.weight_decay,
        "adam_beta1": training.optimizer.beta1,
        "adam_beta2": training.optimizer.beta2,
        "adam_epsilon": training.optimizer.eps,
        "lr_scheduler_type": training.scheduler.name,
        "logging_steps": training.log_every_steps,
        "max_steps": training.max_steps if training.max_steps is not None else -1,
        "max_grad_norm": (
            training.max_grad_norm if training.max_grad_norm is not None else 0.0
        ),
        "seed": training.seed,
        "bf16": training.mixed_precision == "bf16",
        "fp16": training.mixed_precision == "fp16",
        "gradient_checkpointing": training.gradient_checkpointing.enabled,
        "beta": grpo.beta,
        "num_generations": grpo.num_generations,
        "max_completion_length": grpo.max_completion_length,
        "steps_per_generation": grpo.steps_per_generation,
        "temperature": grpo.temperature,
        "top_p": grpo.top_p,
        "epsilon": grpo.epsilon,
        "epsilon_high": grpo.epsilon_high,
        "loss_type": grpo.loss_type,
        "scale_rewards": grpo.scale_rewards,
        "num_iterations": grpo.num_iterations,
        "mask_truncated_completions": grpo.mask_truncated_completions,
        "reward_weights": [reward.weight for reward in grpo.rewards],
        "report_to": [],
        "remove_unused_columns": False,
        "deepspeed": dpo_deepspeed_config(job),
    }
    if training.gradient_checkpointing.use_reentrant is not None:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": training.gradient_checkpointing.use_reentrant
        }
    if training.scheduler.warmup_steps is None:
        kwargs["warmup_ratio"] = training.scheduler.warmup_ratio
    else:
        kwargs["warmup_steps"] = training.scheduler.warmup_steps

    if has_eval and evaluation.every_steps is not None:
        kwargs.update(eval_strategy="steps", eval_steps=evaluation.every_steps)
    elif has_eval and evaluation.every_epochs == 1:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "no"

    if training.checkpoint.save_every_steps is None:
        kwargs["save_strategy"] = "no"
    else:
        kwargs.update(
            save_strategy="steps",
            save_steps=training.checkpoint.save_every_steps,
            save_total_limit=training.checkpoint.keep_last,
        )
    return kwargs


def run_grpo_job(
    job: TrainingJobConfig,
    *,
    train_dataset,
    eval_dataset=None,
):
    """Run one local or Accelerate/DeepSpeed-launched GRPO job."""

    if job.data.format != "grpo" or job.grpo is None:
        raise ValueError("run_grpo_job requires data.format='grpo'")
    _require_grpo_dependencies()

    from datasets import Dataset
    from trl import GRPOConfig as TRLGRPOConfig
    from trl import GRPOTrainer

    components = load_causal_lm(
        job.model.name_or_path,
        tokenizer_name_or_path=job.model.tokenizer_name_or_path,
        model_kwargs={
            **dict(job.model.model_kwargs),
            **(
                {}
                if job.model.dtype == "auto"
                else {"torch_dtype": getattr(torch, job.model.dtype)}
            ),
            "trust_remote_code": job.model.trust_remote_code,
        },
        tokenizer_kwargs={
            **dict(job.model.tokenizer_kwargs),
            "trust_remote_code": job.model.trust_remote_code,
        },
    )
    components.tokenizer.padding_side = "left"

    model, adaptation = adapt_model_for_training(
        components.model,
        freeze=job.training.freeze,
        lora=job.training.lora,
        trainable_tokens=None,
    )
    configure_gradient_checkpointing(
        model,
        job.training.gradient_checkpointing,
        require_input_grads=adaptation.method is not None,
    )

    train_rows = prepare_prompt_rows(
        train_dataset.rows,
        job.data,
        tokenizer=components.tokenizer,
        system_prompt=job.grpo.system_prompt,
    )
    eval_rows = (
        prepare_prompt_rows(
            eval_dataset.rows,
            job.data,
            tokenizer=components.tokenizer,
            system_prompt=job.grpo.system_prompt,
        )
        if eval_dataset is not None
        else None
    )
    reward_funcs = [load_reward(config) for config in job.grpo.rewards]
    args = TRLGRPOConfig(**grpo_training_arguments(job))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows is not None else None,
        processing_class=components.tokenizer,
    )
    result = trainer.train(
        resume_from_checkpoint=job.training.checkpoint.resume_from
    )

    if job.artifacts.save != "none":
        trainer.save_model(job.training.output_dir)
        if trainer.is_world_process_zero():
            output_dir = Path(job.training.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            components.tokenizer.save_pretrained(output_dir)
            (output_dir / "grpo_config.json").write_text(
                json.dumps(asdict(job.grpo), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if job.grpo.system_prompt is not None:
                (output_dir / "system_prompt.txt").write_text(
                    job.grpo.system_prompt + "\n",
                    encoding="utf-8",
                )
    return result
