from __future__ import annotations

from collections.abc import Callable, Mapping
import itertools
import math
from pathlib import Path
from typing import Any

from accelerate.utils import set_seed
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from .adapters import ModelAdaptationSummary, adapt_model_for_training
from .artifacts import save_lora_adapter, save_trainable_tokens_adapter
from .backend import configure_gradient_checkpointing
from .checkpoint import (
    TrainerState,
    find_latest_checkpoint,
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
)
from .config import TrainingConfig
from .distributed import (
    DistributedRuntime,
    broadcast_main_process_bool,
    raise_if_any_rank_failed,
)
from .evaluation import EvaluationMetric, evaluate_model
from .loss import BaseLoss, load_loss, shift_causal_labels, validate_loss
from .model import save_causal_lm
from .optimizer import build_training_optimizer
from .scheduler import bind_scheduler_optimizer, build_training_scheduler


class EpochSeededRandomSampler(Sampler[int]):
    """Deterministic shuffle whose order depends only on seed + epoch."""

    def __init__(self, data_source: Dataset[Any], *, seed: int) -> None:
        self.data_source = data_source
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


class Trainer:
    """A small causal-LM training loop with checkpoint/resume support."""

    def __init__(
        self,
        *,
        model: nn.Module,
        train_dataset: Dataset[Any],
        collate_fn: Callable[[list[Any]], Mapping[str, torch.Tensor]],
        config: TrainingConfig,
        tokenizer: PreTrainedTokenizerBase | None = None,
        optimizer: Optimizer | None = None,
        scheduler: LRScheduler | None = None,
        loss_fn: BaseLoss | None = None,
        runtime: DistributedRuntime | None = None,
        eval_dataset: Dataset[Any] | None = None,
        eval_collate_fn: Callable | None = None,
        eval_metrics: tuple[EvaluationMetric, ...] = (),
    ) -> None:
        config.validate()
        set_seed(config.seed)

        if loss_fn is not None and (
            config.loss.module_path is not None or config.loss.kwargs
        ):
            raise ValueError(
                "Specify either Trainer(loss_fn=...) or config.loss, not both"
            )

        ds = config.deepspeed
        if optimizer is not None and ds is not None and ds.optimizer is not None:
            raise ValueError(
                "Trainer optimizer injection cannot be combined with a "
                "DeepSpeed-owned optimizer"
            )
        if scheduler is not None and ds is not None and ds.scheduler is not None:
            raise ValueError(
                "Trainer scheduler injection cannot be combined with a "
                "DeepSpeed-owned scheduler"
            )
        if optimizer is not None and (
            config.lora is not None or config.trainable_tokens is not None
        ):
            raise ValueError(
                "Trainer optimizer injection cannot be combined with PEFT adaptation: "
                "adapter parameters are created inside Trainer before optimizer construction"
            )

        self.config = config
        if eval_dataset is not None and len(eval_dataset) == 0:
            raise ValueError("Evaluation dataset is empty")
        self.eval_dataset = eval_dataset
        self.eval_collate_fn = eval_collate_fn or collate_fn
        self.eval_metrics = eval_metrics
        self.evaluation_history = []
        self.tokenizer = tokenizer
        self.runtime = runtime or DistributedRuntime(config)
        self.runtime.ensure_config(config)
        self.accelerator = self.runtime.accelerator
        self.loss_fn = (
            validate_loss(loss_fn)
            if loss_fn is not None
            else load_loss(
                config.loss.module_path,
                config.loss.kwargs,
            )
        )
        self.loss_fn.to(self.accelerator.device)

        if (
            config.trainable_tokens is not None
            and ds is not None
            and ds.zero_stage == 3
            and self.accelerator.device.type == "cuda"
        ):
            # PEFT initializes selected token rows with an NCCL broadcast as
            # soon as ZeRO-3 is configured, before Accelerate prepares the model.
            # Its source tensor inherits the base weight's device. Place the
            # unpartitioned model on this rank's GPU before that collective.
            # Init-time partitioning is not supported by this runtime yet.
            model.to(self.accelerator.device)

        model, self.adaptation_summary = adapt_model_for_training(
            model,
            freeze=config.freeze,
            lora=config.lora,
            trainable_tokens=config.trainable_tokens,
        )
        adapter_training = (
            config.lora is not None or config.trainable_tokens is not None
        )
        configure_gradient_checkpointing(
            model,
            config.gradient_checkpointing,
            require_input_grads=adapter_training,
        )

        self._sampler = EpochSeededRandomSampler(train_dataset, seed=config.seed)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_batch_size,
            sampler=self._sampler,
            collate_fn=collate_fn,
            # DataLoader creates a base seed whenever a new iterator starts.
            # Keep this separate from model RNG so resuming mid-epoch does not
            # consume an extra dropout/random-operation seed.
            generator=torch.Generator().manual_seed(config.seed),
        )
        if len(dataloader) == 0:
            raise ValueError("Training dataset produced an empty dataloader")

        optimizer = optimizer or build_training_optimizer(model, config)

        # Accelerate's default split_batches=False assigns different complete
        # batches to each process, then AcceleratedScheduler advances the
        # underlying scheduler ``num_processes`` times for every synchronized
        # optimizer update. Build the scheduler horizon in those underlying
        # scheduler-step units. Public step settings remain optimizer updates;
        # build_training_scheduler translates them using scheduler_step_scale.
        batches_per_process = math.ceil(
            len(dataloader) / self.accelerator.num_processes
        )
        updates_per_epoch = math.ceil(
            batches_per_process / config.gradient_accumulation_steps
        )
        optimizer_training_steps = (
            config.max_steps
            if config.max_steps is not None
            else updates_per_epoch * config.num_epochs
        )
        total_training_steps = (
            optimizer_training_steps * self.accelerator.num_processes
        )
        scheduler = scheduler or build_training_scheduler(
            optimizer,
            config,
            num_training_steps=total_training_steps,
            scheduler_step_scale=self.accelerator.num_processes,
        )

        self.model, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(model, optimizer, dataloader, scheduler)
        )
        if ds is not None:
            bind_scheduler_optimizer(scheduler, self.optimizer)
        self.state = TrainerState()

        if config.checkpoint.resume_from is not None:
            checkpoint_dir = resolve_resume_checkpoint(
                config.output_dir,
                config.checkpoint.resume_from,
            )
            self.state = load_training_checkpoint(self.accelerator, checkpoint_dir)
            if (
                config.max_steps is not None
                and self.state.global_step > config.max_steps
            ):
                raise ValueError(
                    f"Resume checkpoint global_step={self.state.global_step} exceeds "
                    f"max_steps={config.max_steps}"
                )

    def train(self) -> TrainerState:
        self.model.train()
        running_loss = 0.0
        running_updates = 0
        start_epoch = self.state.epoch
        if (
            self.config.max_steps is not None
            and self.state.global_step >= self.config.max_steps
        ):
            epoch_iterator = ()
        elif self.config.max_steps is not None:
            epoch_iterator = itertools.count(start_epoch)
        else:
            epoch_iterator = range(start_epoch, self.config.num_epochs)

        for epoch in epoch_iterator:
            self._sampler.set_epoch(epoch)
            start_batch = self.state.batch_in_epoch if epoch == start_epoch else 0
            epoch_loader = self.train_dataloader
            if start_batch:
                epoch_loader = self.accelerator.skip_first_batches(
                    self.train_dataloader,
                    start_batch,
                )

            # A fresh prepared/skip loader starts at iteration zero and may
            # otherwise reset the sampler to epoch zero inside __iter__.
            set_epoch = getattr(epoch_loader, "set_epoch", None)
            if callable(set_epoch):
                set_epoch(epoch)

            reached_step_limit = False
            for batch_index, batch in enumerate(epoch_loader, start=start_batch):
                with self.accelerator.accumulate(self.model):
                    raise_if_any_rank_failed(
                        (
                            None
                            if "labels" in batch
                            else KeyError("Training batch must contain labels")
                        ),
                        "Training batch validation",
                    )
                    labels = batch["labels"]
                    model_inputs = {
                        key: value for key, value in batch.items() if key != "labels"
                    }

                    outputs = self.model(**model_inputs)
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        raise ValueError("Model forward must return logits")

                    loss = self.loss_fn(logits, shift_causal_labels(labels))
                    error = None
                    if loss.ndim != 0:
                        error = ValueError("loss_fn must return a scalar tensor")
                    elif not bool(torch.isfinite(loss.detach()).item()):
                        error = FloatingPointError(
                            f"Non-finite loss before backward: {loss.detach().item()}"
                        )
                    raise_if_any_rank_failed(error, "Loss validation")

                    # Backends divide by the configured accumulation factor.
                    # The final window may contain fewer microbatches: normalize
                    # by its actual size so its gradient is not underweighted.
                    accumulation = self.config.gradient_accumulation_steps
                    window_start = (batch_index // accumulation) * accumulation
                    window_size = min(
                        accumulation, len(self.train_dataloader) - window_start
                    )
                    self.accelerator.backward(loss * (accumulation / window_size))

                    # Under DeepSpeed, accelerator.backward() delegates the
                    # accumulation-boundary step to DeepSpeedEngine, including
                    # configured gradient clipping. Manual clipping here would
                    # happen after that step and is therefore both redundant and
                    # too late. Non-DeepSpeed runtimes clip explicitly here.
                    if (
                        self.config.deepspeed is None
                        and self.accelerator.sync_gradients
                        and self.config.max_grad_norm is not None
                    ):
                        self.accelerator.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

                self.state.epoch = epoch
                self.state.batch_in_epoch = batch_index + 1

                if self.accelerator.sync_gradients:
                    self.state.global_step += 1
                    running_loss += float(loss.detach().item())
                    running_updates += 1

                    if self.state.global_step % self.config.log_every_steps == 0:
                        mean_loss = running_loss / max(running_updates, 1)
                        self.accelerator.print(
                            f"step={self.state.global_step} loss={mean_loss:.6f}"
                        )
                        running_loss = 0.0
                        running_updates = 0

                    save_every = self.config.checkpoint.save_every_steps
                    eval_every = self.config.evaluation.every_steps
                    if (
                        self.eval_dataset is not None
                        and eval_every
                        and self.state.global_step % eval_every == 0
                    ):
                        self.evaluate()
                    if save_every and self.state.global_step % save_every == 0:
                        self.save_checkpoint()

                    if (
                        self.config.max_steps is not None
                        and self.state.global_step >= self.config.max_steps
                    ):
                        reached_step_limit = True

                if reached_step_limit:
                    break

            epoch_completed = self.state.batch_in_epoch >= len(self.train_dataloader)
            if epoch_completed:
                self.state.epoch = epoch + 1
                self.state.batch_in_epoch = 0
                every_epochs = self.config.evaluation.every_epochs
                if (
                    self.eval_dataset is not None
                    and every_epochs
                    and self.state.epoch % every_epochs == 0
                ):
                    if self.state.last_eval_step != self.state.global_step:
                        self.evaluate()
                    self.state.last_eval_epoch = self.state.epoch

            if reached_step_limit:
                break

        needs_final_checkpoint = False
        error = None
        if self.accelerator.is_main_process:
            try:
                latest = find_latest_checkpoint(self.config.output_dir)
                latest_step = None
                if latest is not None:
                    try:
                        latest_step = int(latest.name.rsplit("-", 1)[1])
                    except (IndexError, ValueError):
                        latest_step = None
                latest_state = (
                    TrainerState.load(latest / "trainer_state.json")
                    if latest is not None
                    else None
                )
                needs_final_checkpoint = self.state.global_step > 0 and (
                    latest_step != self.state.global_step
                    or latest_state != self.state
                )
            except Exception as exc:
                error = exc
        raise_if_any_rank_failed(error, "Final checkpoint inspection")
        needs_final_checkpoint = broadcast_main_process_bool(
            needs_final_checkpoint
        )
        if needs_final_checkpoint:
            self.save_checkpoint()

        return self.state

    def evaluate(self, dataset=None) -> dict[str, float]:
        if not self.accelerator.sync_gradients:
            raise RuntimeError(
                "Evaluation requires a completed gradient accumulation boundary"
            )
        dataset = self.eval_dataset if dataset is None else dataset
        if dataset is None:
            raise ValueError("No evaluation dataset was supplied")
        result = evaluate_model(
            self.model,
            self.accelerator,
            dataset,
            self.eval_collate_fn,
            batch_size=self.config.evaluation.batch_size
            or self.config.per_device_batch_size,
            seed=self.config.seed,
            metrics=self.eval_metrics,
        )
        self.state.last_eval_step = self.state.global_step
        self.evaluation_history.append(
            dict(step=self.state.global_step, epoch=self.state.epoch, **result)
        )
        self.accelerator.print(f"evaluation step={self.state.global_step} {result}")
        return result

    def save_checkpoint(self) -> Path:
        if not self.accelerator.sync_gradients:
            raise RuntimeError(
                "Checkpoint requires a completed gradient accumulation boundary"
            )
        return save_training_checkpoint(
            self.accelerator,
            output_dir=self.config.output_dir,
            state=self.state,
            config=self.config,
            tokenizer=self.tokenizer,
        )

    def save_adapter(self, output_dir: str | Path | None = None) -> Path:
        """Save the configured PEFT adapter without consolidating the base model."""

        destination = Path(output_dir or self.config.output_dir) / "adapter"
        if self.config.lora is not None:
            return save_lora_adapter(
                self.model,
                accelerator=self.accelerator,
                output_dir=destination,
                config=self.config,
                adaptation_summary=self.adaptation_summary,
                tokenizer=self.tokenizer,
            )
        if self.config.trainable_tokens is not None:
            return save_trainable_tokens_adapter(
                self.model,
                accelerator=self.accelerator,
                output_dir=destination,
                config=self.config,
                adaptation_summary=self.adaptation_summary,
                tokenizer=self.tokenizer,
            )
        raise ValueError("save_adapter() requires a configured PEFT adaptation method")

    def export_model(self, output_dir: str | Path | None = None) -> None:
        if self.config.lora is not None:
            raise NotImplementedError(
                "LoRA uses a separate artifact lifecycle: call save_adapter() "
                "during/after training, then merge_lora(base, adapter, output) "
                "offline to produce a plain Hugging Face checkpoint."
            )
        if self.config.trainable_tokens is not None:
            raise NotImplementedError(
                "TrainableTokens uses a separate artifact lifecycle: call "
                "save_adapter() during/after training, then "
                "merge_trainable_tokens(expanded_base, adapter, output) offline."
            )

        destination = Path(output_dir or self.config.output_dir) / "final-model"
        save_causal_lm(
            self.model,
            self.tokenizer,
            destination,
            accelerator=self.accelerator,
        )
