from __future__ import annotations

"""TRL-backed Direct Preference Optimization for config-driven jobs."""

from dataclasses import asdict
import importlib.util
import inspect
from importlib.metadata import version as package_version
import json
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping

import torch

from .adapters import adapt_model_for_training
from .backend import configure_gradient_checkpointing
from .config_file import DataConfig, TrainingJobConfig
from .model import load_causal_lm



def _load_trl_dpo_classes():
    """Import DPO classes, retaining a compatibility shim for legacy TRL 0.23.

    TRL 0.23 imports mergekit and llm-blender callback modules eagerly. Those
    packages are not used by DPOTrainer and may be absent or incompatible with
    Transformers 5. Temporarily providing their imported symbols keeps the DPO
    dependency surface limited to the code path we actually use.
    """

    from trl import DPOConfig as TRLDPOConfig

    if not package_version("trl").startswith("0.23."):
        from trl import DPOTrainer

        return TRLDPOConfig, DPOTrainer

    missing = object()
    module_names = ("trl.mergekit_utils", "trl.trainer.judges")
    previous = {name: sys.modules.get(name, missing) for name in module_names}

    mergekit_stub = ModuleType("trl.mergekit_utils")

    class MergeConfig:
        pass

    def _unavailable_callback(*args, **kwargs):
        raise ImportError(
            "This optional TRL callback requires mergekit/llm-blender."
        )

    mergekit_stub.MergeConfig = MergeConfig
    mergekit_stub.merge_models = _unavailable_callback
    mergekit_stub.upload_model_to_hf = _unavailable_callback

    judges_stub = ModuleType("trl.trainer.judges")

    class BasePairwiseJudge:
        pass

    judges_stub.BasePairwiseJudge = BasePairwiseJudge

    sys.modules[module_names[0]] = mergekit_stub
    sys.modules[module_names[1]] = judges_stub
    try:
        from trl import DPOTrainer
    finally:
        for name, old_value in previous.items():
            if old_value is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_value
    return TRLDPOConfig, DPOTrainer


def _require_dpo_dependencies() -> None:
    missing = [
        package
        for package in ("datasets", "trl")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ImportError(
            "DPO training requires "
            + ", ".join(missing)
            + ". Install llm-training with the 'dpo' extra."
        )


def _require_text(row: Mapping[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Dataset field {field!r} must be a non-empty string")
    return value


def prepare_preference_rows(
    rows: list[Mapping[str, Any]],
    data: DataConfig,
    *,
    tokenizer=None,
    system_prompt: str | None = None,
    max_prompt_length: int | None = None,
) -> list[dict[str, str]]:
    """Normalize configurable JSONL columns to TRL's preference schema."""

    if data.format != "dpo":
        raise ValueError("prepare_preference_rows requires data.format='dpo'")
    prepared: list[dict[str, str]] = []
    for row in rows:
        prompt = _require_text(row, data.prompt_field)
        chosen = _require_text(row, data.chosen_field)
        rejected = _require_text(row, data.rejected_field)
        if chosen == rejected:
            raise ValueError("DPO chosen and rejected completions must differ")
        if system_prompt is not None:
            if tokenizer is None:
                raise ValueError("A tokenizer is required when dpo.system_prompt is set")
            apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise TypeError(
                    "dpo.system_prompt requires a tokenizer with apply_chat_template()"
                )
            prompt = apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        if max_prompt_length is not None:
            if tokenizer is None:
                raise ValueError(
                    "A tokenizer is required when dpo.max_prompt_length is set"
                )
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=False,
            )["input_ids"]
            if len(prompt_ids) > max_prompt_length:
                prompt = tokenizer.decode(
                    prompt_ids[-max_prompt_length:],
                    skip_special_tokens=False,
                )
        prepared.append(
            {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        )
    if not prepared:
        raise ValueError("DPO dataset must contain at least one example")
    return prepared


def _offload(device: str, nvme_path: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"device": device}
    if device == "nvme":
        result["nvme_path"] = nvme_path
    return result


def dpo_deepspeed_config(job: TrainingJobConfig) -> dict[str, Any] | None:
    """Translate shared ZeRO settings without duplicating Trainer ownership."""

    ds = job.training.deepspeed
    if ds is None:
        return None
    zero: dict[str, Any] = {"stage": ds.zero_stage}
    if ds.zero_stage >= 2:
        zero["offload_optimizer"] = _offload(
            ds.offload_optimizer_device,
            ds.offload_optimizer_nvme_path,
        )
    if ds.zero_stage == 3:
        zero["offload_param"] = _offload(
            ds.offload_param_device,
            ds.offload_param_nvme_path,
        )
        zero["stage3_gather_16bit_weights_on_model_save"] = (
            ds.zero3_save_16bit_model
        )
    return {
        "zero_optimization": zero,
        "gradient_accumulation_steps": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "train_batch_size": "auto",
        "gradient_clipping": "auto",
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto"},
    }


def dpo_training_arguments(job: TrainingJobConfig) -> dict[str, Any]:
    """Build kwargs for trl.DPOConfig without importing optional dependencies."""

    if job.dpo is None:
        raise ValueError("DPO parameters are missing")
    training = job.training
    dpo = job.dpo
    has_eval = job.data.validation_file is not None
    evaluation = training.evaluation

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
        "beta": dpo.beta,
        "loss_type": dpo.loss_type,
        "label_smoothing": dpo.label_smoothing,
        "max_length": dpo.max_length,
        "max_prompt_length": dpo.max_prompt_length,
        "reference_free": dpo.reference_free,
        "report_to": [],
        "remove_unused_columns": False,
        "deepspeed": dpo_deepspeed_config(job),
    }
    if training.gradient_checkpointing.use_reentrant is not None:
        kwargs["gradient_checkpointing_kwargs"] = {
            "use_reentrant": training.gradient_checkpointing.use_reentrant
        }
    # Transformers 5 represents both an absolute warmup count and a ratio
    # through warmup_steps: values below 1 are interpreted as a ratio.
    kwargs["warmup_steps"] = (
        training.scheduler.warmup_ratio
        if training.scheduler.warmup_steps is None
        else training.scheduler.warmup_steps
    )

    if has_eval and evaluation.every_steps is not None:
        kwargs.update(
            eval_strategy="steps",
            eval_steps=evaluation.every_steps,
        )
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



def _build_trl_dpo_config(config_cls, kwargs: Mapping[str, Any]):
    """Construct a DPO config across the supported TRL API transition."""

    supported = inspect.signature(config_cls).parameters
    effective = dict(kwargs)
    if "max_prompt_length" not in supported:
        # Prompt truncation is applied in prepare_preference_rows for newer TRL.
        effective.pop("max_prompt_length", None)
    if "reference_free" not in supported:
        if effective.get("reference_free"):
            raise ValueError(
                "The supported TRL release does not expose reference-free DPO"
            )
        effective.pop("reference_free", None)
    return config_cls(**effective)

def run_dpo_job(
    job: TrainingJobConfig,
    *,
    train_dataset,
    eval_dataset=None,
):
    """Run one local or accelerate/DeepSpeed-launched DPO job."""

    if job.data.format != "dpo" or job.dpo is None:
        raise ValueError("run_dpo_job requires data.format='dpo'")
    _require_dpo_dependencies()

    from datasets import Dataset

    TRLDPOConfig, DPOTrainer = _load_trl_dpo_classes()

    components = load_causal_lm(
        job.model.name_or_path,
        tokenizer_name_or_path=job.model.tokenizer_name_or_path,
        model_kwargs={
            **dict(job.model.model_kwargs),
            **(
                {}
                if job.model.dtype == "auto"
                else {
                    "torch_dtype": getattr(torch, job.model.dtype)
                }
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

    train_rows = prepare_preference_rows(
        train_dataset.rows,
        job.data,
        tokenizer=components.tokenizer,
        system_prompt=job.dpo.system_prompt,
        max_prompt_length=job.dpo.max_prompt_length,
    )
    eval_rows = (
        prepare_preference_rows(
            eval_dataset.rows,
            job.data,
            tokenizer=components.tokenizer,
            system_prompt=job.dpo.system_prompt,
            max_prompt_length=job.dpo.max_prompt_length,
        )
        if eval_dataset is not None
        else None
    )
    # Some Transformers/PEFT combinations do not create this mapping on the
    # forwarded model before TRL writes into it.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    args = _build_trl_dpo_config(TRLDPOConfig, dpo_training_arguments(job))
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
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
            Path(job.training.output_dir).mkdir(parents=True, exist_ok=True)
            output_dir = Path(job.training.output_dir)
            components.tokenizer.model_max_length = job.dpo.max_length
            components.tokenizer.save_pretrained(output_dir)
            (output_dir / "dpo_config.json").write_text(
                json.dumps(asdict(job.dpo), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if job.dpo.system_prompt is not None:
                (output_dir / "system_prompt.txt").write_text(
                    job.dpo.system_prompt + "\n",
                    encoding="utf-8",
                )
    return result

