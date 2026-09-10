from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

from .adapters import ModelAdaptationSummary
from .backend import build_deepspeed_config
from .config import TrainingConfig
from .run import RunIdentity
from .scheduler import resolve_warmup_steps


@dataclass(frozen=True)
class PlanWarning:
    code: str
    message: str


@dataclass(frozen=True)
class TrainingPlan:
    run_id: str | None
    artifact_uri: str | None
    model_name: str | None
    max_sequence_length: int | None
    train_examples: int | None
    num_epochs: int
    max_steps: int | None
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    global_batch_size: int
    batches_per_rank: int | None
    optimizer_steps_per_epoch: int | None
    optimizer_steps: int | None
    scheduler_steps: int | None
    warmup_optimizer_steps: int | None
    warmup_fraction: float | None
    checkpoint_every_steps: int | None
    expected_checkpoints: int | None
    retained_checkpoints: int | None
    evaluation_events: int | None
    output_dir: str
    mixed_precision: str
    gradient_checkpointing: bool
    adaptation_method: str
    trainable_parameters: int | None
    total_parameters: int | None
    trainable_fraction: float | None
    optimizer_owner: str
    optimizer_type: str
    learning_rate: float
    weight_decay: float
    beta1: float
    beta2: float
    eps: float
    scheduler_owner: str
    scheduler_type: str
    max_grad_norm: float | None
    gradient_clip_owner: str | None
    deepspeed_stage: int | None
    offload_optimizer_device: str | None
    offload_param_device: str | None
    zero3_save_16bit_model: bool | None
    effective_deepspeed_config: dict | None
    warnings: tuple[PlanWarning, ...]


def _requested_adaptation(config: TrainingConfig) -> str:
    if config.trainable_tokens is not None:
        return f"TrainableTokens ({config.trainable_tokens.n_tokens_added} vocabulary rows)"
    if config.lora is not None and config.freeze is not None:
        return "LoRA + explicitly trainable base parameters"
    if config.lora is not None:
        return "LoRA"
    if config.freeze is None:
        return "full fine-tuning"
    freeze = config.freeze
    if freeze.train_last_n_layers is None:
        layer_policy = "all transformer layers"
    elif freeze.train_last_n_layers == 0:
        layer_policy = "no transformer layers"
    else:
        layer_policy = f"last {freeze.train_last_n_layers} transformer layers"
    endpoints = []
    if freeze.train_embeddings:
        endpoints.append("embeddings")
    if freeze.train_lm_head:
        endpoints.append("LM head")
    if freeze.trainable_prefixes:
        endpoints.append(f"{len(freeze.trainable_prefixes)} custom prefixes")
    suffix = ", ".join(endpoints) if endpoints else "no endpoints"
    return f"selective full training: {layer_policy}; {suffix}"


def build_training_plan(
    config: TrainingConfig,
    *,
    train_examples: int | None = None,
    world_size: int = 1,
    adaptation_summary: ModelAdaptationSummary | None = None,
    model_name: str | None = None,
    max_sequence_length: int | None = None,
    has_eval_data: bool = False,
    output_exists: bool = False,
    run_identity: RunIdentity | None = None,
) -> TrainingPlan:
    """Resolve requested knobs into the effective training execution plan."""
    config.validate()
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if train_examples is not None and train_examples <= 0:
        raise ValueError("train_examples must be > 0 when provided")
    if max_sequence_length is not None and max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be > 0 when provided")

    global_batch_size = (
        config.per_device_batch_size
        * config.gradient_accumulation_steps
        * world_size
    )

    batches_per_rank = None
    updates_per_epoch = None
    optimizer_steps = None
    scheduler_steps = None
    warmup_optimizer_steps = None
    warmup_fraction = None
    expected_checkpoints = None
    retained_checkpoints = None
    evaluation_events = None

    if train_examples is not None:
        dataloader_batches = math.ceil(train_examples / config.per_device_batch_size)
        batches_per_rank = math.ceil(dataloader_batches / world_size)
        updates_per_epoch = math.ceil(
            batches_per_rank / config.gradient_accumulation_steps
        )

    if config.max_steps is not None:
        optimizer_steps = config.max_steps
    elif updates_per_epoch is not None:
        optimizer_steps = updates_per_epoch * config.num_epochs

    if optimizer_steps is not None:
        scheduler_steps = optimizer_steps * world_size
        warmup_optimizer_steps = resolve_warmup_steps(
            config.scheduler,
            num_training_steps=optimizer_steps,
        )
        warmup_fraction = warmup_optimizer_steps / optimizer_steps

        save_every = config.checkpoint.save_every_steps
        if save_every is None:
            expected_checkpoints = 1
        else:
            periodic = optimizer_steps // save_every
            final_is_new_step = optimizer_steps % save_every != 0
            expected_checkpoints = max(periodic + int(final_is_new_step), 1)
        retained_checkpoints = (
            expected_checkpoints
            if config.checkpoint.keep_last is None
            else min(expected_checkpoints, config.checkpoint.keep_last)
        )

    if has_eval_data and optimizer_steps is not None:
        eval_steps: set[int] = set()
        if config.evaluation.every_steps is not None:
            eval_steps.update(
                range(
                    config.evaluation.every_steps,
                    optimizer_steps + 1,
                    config.evaluation.every_steps,
                )
            )

        if config.evaluation.every_epochs is not None:
            if updates_per_epoch is None:
                evaluation_events = None
            else:
                completed_epochs = (
                    optimizer_steps // updates_per_epoch
                    if config.max_steps is not None
                    else config.num_epochs
                )
                for epoch in range(
                    config.evaluation.every_epochs,
                    completed_epochs + 1,
                    config.evaluation.every_epochs,
                ):
                    eval_steps.add(epoch * updates_per_epoch)
                evaluation_events = len(eval_steps)
        else:
            evaluation_events = len(eval_steps)
    elif not has_eval_data:
        evaluation_events = 0

    if adaptation_summary is None:
        adaptation_method = _requested_adaptation(config)
        trainable_parameters = None
        total_parameters = None
        trainable_fraction = None
    else:
        adaptation_method = adaptation_summary.method or "full/selective base training"
        trainable_parameters = adaptation_summary.trainable_parameters
        total_parameters = adaptation_summary.total_parameters
        trainable_fraction = adaptation_summary.trainable_fraction

    ds = config.deepspeed
    if ds is None:
        optimizer_owner = "torch"
        scheduler_owner = "torch/hf"
    else:
        optimizer_owner = (
            "deepspeed"
            if ds.optimizer is not None
            else "torch-configured / deepspeed-prepared"
        )
        scheduler_owner = (
            "deepspeed"
            if ds.scheduler is not None
            else "torch/hf-configured / deepspeed-prepared"
        )
    optimizer_type = (
        ds.optimizer.type
        if ds is not None and ds.optimizer is not None
        else "AdamW"
    )
    scheduler_type = (
        ds.scheduler.type
        if ds is not None and ds.scheduler is not None
        else config.scheduler.name
    )
    gradient_clip_owner = None
    if config.max_grad_norm is not None:
        gradient_clip_owner = "deepspeed engine" if ds is not None else "accelerate"
    effective_ds = build_deepspeed_config(config) if ds is not None else None

    warnings: list[PlanWarning] = []
    if train_examples is None:
        warnings.append(
            PlanWarning(
                "unknown_dataset_size",
                "Dataset size was not supplied; optimizer-step, warmup, checkpoint, and evaluation counts cannot be resolved.",
            )
        )
    if output_exists and config.checkpoint.resume_from is None:
        warnings.append(
            PlanWarning(
                "output_exists",
                "The output directory already contains files and no resume checkpoint is configured; use a new run directory to avoid artifact collisions.",
            )
        )
    if ds is not None and ds.zero_stage == 3:
        warnings.append(
            PlanWarning(
                "zero3_initial_load_unpartitioned",
                "ZeRO-3 partitions parameters at accelerator.prepare(); the current runtime still loads a complete model on every rank first.",
            )
        )
    if warmup_fraction is not None and warmup_fraction >= 0.5:
        warnings.append(
            PlanWarning(
                "warmup_dominates_training",
                f"Warmup covers {warmup_fraction:.0%} of optimizer updates; verify this is intentional.",
            )
        )
    if optimizer_steps is not None and optimizer_steps > 1 and expected_checkpoints == 1:
        warnings.append(
            PlanWarning(
                "final_checkpoint_only",
                "This run will have only the final checkpoint; there is no intermediate recovery point.",
            )
        )
    if has_eval_data and evaluation_events == 1:
        warnings.append(
            PlanWarning(
                "single_evaluation_point",
                "Only one evaluation point is scheduled, so this run cannot provide a validation curve.",
            )
        )
    if optimizer_steps is not None and config.log_every_steps > optimizer_steps:
        warnings.append(
            PlanWarning(
                "no_periodic_loss_log",
                f"log_every_steps={config.log_every_steps} exceeds the expected {optimizer_steps} optimizer steps; no periodic loss line will be emitted.",
            )
        )
    if (
        adaptation_summary is not None
        and config.gradient_checkpointing.enabled
        and adaptation_summary.trainable_fraction < 0.25
    ):
        warnings.append(
            PlanWarning(
                "checkpointing_on_small_trainable_fraction",
                f"Only {adaptation_summary.trainable_fraction:.1%} of parameters are trainable while gradient checkpointing is enabled; measure whether recomputation is still worthwhile.",
            )
        )
    if ds is not None and ds.offload_param_device != "none" and ds.zero_stage == 3:
        warnings.append(
            PlanWarning(
                "zero3_load_not_offloaded",
                "Parameter offload applies after DeepSpeed preparation; it does not reduce the current full-model from_pretrained() materialization requirement.",
            )
        )

    return TrainingPlan(
        run_id=run_identity.run_id if run_identity else None,
        artifact_uri=run_identity.artifact_uri if run_identity else None,
        model_name=model_name,
        max_sequence_length=max_sequence_length,
        train_examples=train_examples,
        num_epochs=config.num_epochs,
        max_steps=config.max_steps,
        world_size=world_size,
        per_device_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        global_batch_size=global_batch_size,
        batches_per_rank=batches_per_rank,
        optimizer_steps_per_epoch=updates_per_epoch,
        optimizer_steps=optimizer_steps,
        scheduler_steps=scheduler_steps,
        warmup_optimizer_steps=warmup_optimizer_steps,
        warmup_fraction=warmup_fraction,
        checkpoint_every_steps=config.checkpoint.save_every_steps,
        expected_checkpoints=expected_checkpoints,
        retained_checkpoints=retained_checkpoints,
        evaluation_events=evaluation_events,
        output_dir=config.output_dir,
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing.enabled,
        adaptation_method=adaptation_method,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_fraction=trainable_fraction,
        optimizer_owner=optimizer_owner,
        optimizer_type=optimizer_type,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        beta1=config.optimizer.beta1,
        beta2=config.optimizer.beta2,
        eps=config.optimizer.eps,
        scheduler_owner=scheduler_owner,
        scheduler_type=scheduler_type,
        max_grad_norm=config.max_grad_norm,
        gradient_clip_owner=gradient_clip_owner,
        deepspeed_stage=ds.zero_stage if ds else None,
        offload_optimizer_device=ds.offload_optimizer_device if ds else None,
        offload_param_device=ds.offload_param_device if ds else None,
        zero3_save_16bit_model=ds.zero3_save_16bit_model if ds else None,
        effective_deepspeed_config=effective_ds,
        warnings=tuple(warnings),
    )


def save_training_plan(plan: TrainingPlan, output_dir: str | Path | None = None) -> Path:
    """Persist the effective plan as machine-readable run metadata."""
    root = Path(output_dir or plan.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "training-plan.json"
    temporary = root / "training-plan.tmp"
    temporary.write_text(json.dumps(asdict(plan), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def _value(value) -> str:
    return "unknown" if value is None else str(value)


def _params(value: int | None) -> str:
    if value is None:
        return "resolved after model adaptation"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.3f}M"
    if value >= 1_000:
        return f"{value / 1_000:.3f}K"
    return str(value)


def format_training_plan(plan: TrainingPlan) -> str:
    """Render a compact human-readable execution plan."""
    backend = (
        f"DeepSpeed ZeRO-{plan.deepspeed_stage}"
        if plan.deepspeed_stage is not None
        else "Accelerate"
    )
    warmup = _value(plan.warmup_optimizer_steps)
    if plan.warmup_fraction is not None:
        warmup += f" ({plan.warmup_fraction:.1%})"
    checkpoint_every = (
        "final only"
        if plan.checkpoint_every_steps is None
        else f"every {plan.checkpoint_every_steps} optimizer steps"
    )
    trainable = _params(plan.trainable_parameters)
    if plan.trainable_fraction is not None:
        trainable += f" / {_params(plan.total_parameters)} ({plan.trainable_fraction:.2%})"

    lines = [
        "EFFECTIVE TRAINING PLAN",
        "",
        "RUN",
        f"  run id: {plan.run_id or 'not provided'}",
        f"  output: {plan.output_dir}",
        f"  artifact uri: {plan.artifact_uri or 'not configured'}",
        "",
        "MODEL",
        f"  model: {plan.model_name or 'not provided'}",
        f"  precision: {plan.mixed_precision}",
        f"  backend: {backend}",
        f"  max sequence length: {_value(plan.max_sequence_length)}",
        f"  gradient checkpointing: {str(plan.gradient_checkpointing).lower()}",
        "",
        "TRAINABLE",
        f"  policy: {plan.adaptation_method}",
        f"  parameters: {trainable}",
        "",
        "BATCH",
        f"  examples: {_value(plan.train_examples)}",
        (
            f"  stopping horizon: {plan.max_steps} optimizer steps"
            if plan.max_steps is not None
            else f"  stopping horizon: {plan.num_epochs} epochs"
        ),
        f"  world size: {plan.world_size}",
        f"  per-device batch: {plan.per_device_batch_size}",
        f"  gradient accumulation: {plan.gradient_accumulation_steps}",
        f"  effective global batch: {plan.global_batch_size}",
        f"  batches/rank/epoch: {_value(plan.batches_per_rank)}",
        f"  optimizer steps/epoch: {_value(plan.optimizer_steps_per_epoch)}",
        f"  total optimizer steps: {_value(plan.optimizer_steps)}",
        "",
        "OPTIMIZER",
        f"  owner: {plan.optimizer_owner}",
        f"  type: {plan.optimizer_type}",
        f"  learning rate: {plan.learning_rate}",
        f"  betas: ({plan.beta1}, {plan.beta2})",
        f"  eps: {plan.eps}",
        f"  weight decay: {plan.weight_decay}",
        f"  gradient clip: {_value(plan.max_grad_norm)} ({plan.gradient_clip_owner or 'disabled'})",
        "",
        "SCHEDULER",
        f"  owner: {plan.scheduler_owner}",
        f"  type: {plan.scheduler_type}",
        f"  warmup optimizer steps: {warmup}",
        f"  internal scheduler steps: {_value(plan.scheduler_steps)}",
        "",
        "DEEPSPEED",
        f"  stage: {_value(plan.deepspeed_stage)}",
        f"  optimizer offload: {_value(plan.offload_optimizer_device)}",
        f"  parameter offload: {_value(plan.offload_param_device)}",
        f"  gather/save 16-bit model: {_value(plan.zero3_save_16bit_model)}",
        "",
        "CHECKPOINTS",
        f"  cadence: {checkpoint_every}",
        f"  expected distinct checkpoints: {_value(plan.expected_checkpoints)}",
        f"  retained checkpoints: {_value(plan.retained_checkpoints)}",
        "",
        "EVALUATION",
        f"  expected evaluation points: {_value(plan.evaluation_events)}",
        "",
        "WARNINGS",
    ]
    if not plan.warnings:
        lines.append("  none")
    else:
        for warning in plan.warnings:
            lines.append(f"  [{warning.code}] {warning.message}")
    return "\n".join(lines)
