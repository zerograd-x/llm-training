from __future__ import annotations

from collections.abc import Callable, Mapping
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
from .distributed import DistributedRuntime
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
        self.tokenizer = tokenizer
        self.runtime = runtime or DistributedRuntime(config)
        self.runtime.ensure_config(config)
        self.accelerator = self.runtime.accelerator
        self.loss_fn = validate_loss(loss_fn) if loss_fn is not None else load_loss(
            config.loss.module_path,
            config.loss.kwargs,
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
        # scheduler-step units. This also handles uneven final batch groups,
        # where Accelerate pads/evenly repeats batches across ranks.
        batches_per_process = math.ceil(
            len(dataloader) / self.accelerator.num_processes
        )
        updates_per_epoch = math.ceil(
            batches_per_process / config.gradient_accumulation_steps
        )
        scheduler_steps_per_epoch = (
            updates_per_epoch * self.accelerator.num_processes
        )
        total_training_steps = scheduler_steps_per_epoch * config.num_epochs
        scheduler = scheduler or build_training_scheduler(
            optimizer,
            config,
            num_training_steps=total_training_steps,
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

    def train(self) -> TrainerState:
        self.model.train()
        running_loss = 0.0
        running_updates = 0
        start_epoch = self.state.epoch

        for epoch in range(start_epoch, self.config.num_epochs):
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

            for batch_index, batch in enumerate(epoch_loader, start=start_batch):
                with self.accelerator.accumulate(self.model):
                    if "labels" not in batch:
                        raise KeyError("Training batch must contain labels")
                    labels = batch["labels"]
                    model_inputs = {
                        key: value for key, value in batch.items() if key != "labels"
                    }

                    outputs = self.model(**model_inputs)
                    logits = getattr(outputs, "logits", None)
                    if logits is None:
                        raise ValueError("Model forward must return logits")

                    loss = self.loss_fn(logits, shift_causal_labels(labels))
                    if loss.ndim != 0:
                        raise ValueError("loss_fn must return a scalar tensor")
                    if not bool(torch.isfinite(loss.detach()).item()):
                        raise FloatingPointError(
                            f"Non-finite loss before backward: {loss.detach().item()}"
                        )

                    self.accelerator.backward(loss)

                    if (
                        self.accelerator.sync_gradients
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
                    if save_every and self.state.global_step % save_every == 0:
                        self.save_checkpoint()

            self.state.epoch = epoch + 1
            self.state.batch_in_epoch = 0

        latest = find_latest_checkpoint(self.config.output_dir)
        latest_step = None
        if latest is not None:
            try:
                latest_step = int(latest.name.rsplit("-", 1)[1])
            except (IndexError, ValueError):
                latest_step = None
        if self.state.global_step > 0 and latest_step != self.state.global_step:
            self.save_checkpoint()

        return self.state

    def save_checkpoint(self) -> Path:
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

