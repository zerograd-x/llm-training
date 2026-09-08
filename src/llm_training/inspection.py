from __future__ import annotations

from dataclasses import dataclass
import math

from .adapters import ModelAdaptationSummary
from .config import TrainingConfig
from .scheduler import resolve_warmup_steps


@dataclass(frozen=True)
class PlanWarning:
    code: str
    message: str


@dataclass(frozen=True)
class TrainingPlan:
    model_name: str | None
    max_sequence_length: int | None
    train_examples: int | None
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
    deepspeed_stage: int | None
    mixed_precision: str
    gradient_checkpointing: bool
    adaptation_method: str
    trainable_parameters: int | None
    total_parameters: int | None
    trainable_fraction: float | None
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
) -> TrainingPlan:
    """Resolve configured knobs into the effective training execution plan.

    The step arithmetic deliberately mirrors ``Trainer``. Accelerate's default
    ``split_batches=False`` assigns complete dataloader batches to ranks and its
    scheduler wrapper advances the underlying scheduler once per process for
    every synchronized optimizer update. ``scheduler_steps`` therefore differs
    from user-visible optimizer steps when ``world_size > 1``.
    """
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
        dataloader_batches = math.ceil(
            train_examples / config.per_device_batch_size
        )
        batches_per_rank = math.ceil(dataloader_batches / world_size)
        updates_per_epoch = math.ceil(
            batches_per_rank / config.gradient_accumulation_steps
        )
        optimizer_steps = updates_per_epoch * config.num_epochs
        scheduler_steps = optimizer_steps * world_size
        warmup_scheduler_steps = resolve_warmup_steps(
            config.scheduler,
            num_training_steps=scheduler_steps,
        )
        warmup_optimizer_steps = math.ceil(warmup_scheduler_steps / world_size)
        warmup_fraction = warmup_scheduler_steps / scheduler_steps

        save_every = config.checkpoint.save_every_steps
        if save_every is None:
            expected_checkpoints = 1
        else:
            periodic = optimizer_steps // save_every
            final_is_new_step = optimizer_steps % save_every != 0
            expected_checkpoints = periodic + int(final_is_new_step)
            expected_checkpoints = max(expected_checkpoints, 1)
        if config.checkpoint.keep_last is None:
            retained_checkpoints = expected_checkpoints
        else:
            retained_checkpoints = min(
                expected_checkpoints,
                config.checkpoint.keep_last,
            )

        if has_eval_data:
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
                for epoch in range(
                    config.evaluation.every_epochs,
                    config.num_epochs + 1,
                    config.evaluation.every_epochs,
                ):
                    eval_steps.add(epoch * updates_per_epoch)
            evaluation_events = len(eval_steps)
        else:
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
    if config.deepspeed is not None and config.deepspeed.zero_stage == 3:
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
                f"Warmup covers {warmup_fraction:.0%} of the scheduler horizon; verify this is intentional.",
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

    return TrainingPlan(
        model_name=model_name,
        max_sequence_length=max_sequence_length,
        train_examples=train_examples,
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
        deepspeed_stage=(config.deepspeed.zero_stage if config.deepspeed else None),
        mixed_precision=config.mixed_precision,
        gradient_checkpointing=config.gradient_checkpointing.enabled,
        adaptation_method=adaptation_method,
        trainable_parameters=trainable_parameters,
        total_parameters=total_parameters,
        trainable_fraction=trainable_fraction,
        warnings=tuple(warnings),
    )


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
    """Render a compact, stable human-readable execution plan."""
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
        else f"every {plan.checkpoint_every_steps} steps"
    )
    trainable = _params(plan.trainable_parameters)
    if plan.trainable_fraction is not None:
        trainable += f" / {_params(plan.total_parameters)} ({plan.trainable_fraction:.2%})"

    lines = [
        "EFFECTIVE TRAINING PLAN",
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
        f"  world size: {plan.world_size}",
        f"  per-device batch: {plan.per_device_batch_size}",
        f"  gradient accumulation: {plan.gradient_accumulation_steps}",
        f"  effective global batch: {plan.global_batch_size}",
        f"  batches/rank/epoch: {_value(plan.batches_per_rank)}",
        f"  optimizer steps/epoch: {_value(plan.optimizer_steps_per_epoch)}",
        f"  total optimizer steps: {_value(plan.optimizer_steps)}",
        "",
        "SCHEDULE",
        f"  warmup: {warmup}",
        f"  internal scheduler steps: {_value(plan.scheduler_steps)}",
        "",
        "CHECKPOINTS",
        f"  cadence: {checkpoint_every}",
        f"  expected distinct checkpoints: {_value(plan.expected_checkpoints)}",
        f"  retained checkpoints: {_value(plan.retained_checkpoints)}",
        "",
        "EVALUATION",
        f"  expected evaluation points: {_value(plan.evaluation_events)}",
        "",
        "OUTPUT",
        f"  directory: {plan.output_dir}",
    ]
    lines.extend(["", "WARNINGS"])
    if not plan.warnings:
        lines.append("  none")
    else:
        for warning in plan.warnings:
            lines.append(f"  [{warning.code}] {warning.message}")
    return "\n".join(lines)
