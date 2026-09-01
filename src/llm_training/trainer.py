from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from pathlib import Path
from typing import Any

from accelerate import Accelerator
from accelerate.utils import set_seed
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import PreTrainedTokenizerBase

from .checkpoint import (
    TrainerState,
    find_latest_checkpoint,
    load_training_checkpoint,
    resolve_resume_checkpoint,
    save_training_checkpoint,
)
from .config import TrainingConfig
from .model import save_causal_lm
from .optimizer import build_adamw
from .scheduler import build_scheduler


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
    """A small training loop with checkpoint/resume as a first-class behavior."""

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
    ) -> None:
        config.validate()
        set_seed(config.seed)

        self.config = config
        self.tokenizer = tokenizer
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            mixed_precision=config.mixed_precision,
        )

        self._sampler = EpochSeededRandomSampler(train_dataset, seed=config.seed)
        dataloader = DataLoader(
            train_dataset,
            batch_size=config.per_device_batch_size,
            sampler=self._sampler,
            collate_fn=collate_fn,
        )
        if len(dataloader) == 0:
            raise ValueError("Training dataset produced an empty dataloader")

        optimizer = optimizer or build_adamw(model, config.optimizer)
        updates_per_epoch = math.ceil(
            len(dataloader) / config.gradient_accumulation_steps
        )
        total_training_steps = updates_per_epoch * config.num_epochs
        scheduler = scheduler or build_scheduler(
            optimizer,
            config.scheduler,
            num_training_steps=total_training_steps,
        )

        self.model, self.optimizer, self.train_dataloader, self.scheduler = (
            self.accelerator.prepare(model, optimizer, dataloader, scheduler)
        )
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

            for batch_index, batch in enumerate(epoch_loader, start=start_batch):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    if loss is None:
                        raise ValueError("Model forward returned loss=None")
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

    def export_model(self, output_dir: str | Path | None = None) -> None:
        destination = Path(output_dir or self.config.output_dir) / "final-model"
        save_causal_lm(
            self.model,
            self.tokenizer,
            destination,
            accelerator=self.accelerator,
        )
