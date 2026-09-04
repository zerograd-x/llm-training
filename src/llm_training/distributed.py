from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
import os
from pathlib import Path
from typing import Any, Literal

from accelerate import Accelerator
import torch

from .backend import build_accelerator
from .config import TrainingConfig
from .model import CausalLMComponents, load_causal_lm

MaterializationScope = Literal["global", "local"]


def raise_if_any_rank_failed(error: Exception | None, operation: str) -> None:
    """Propagate recoverable validation errors before peers enter collectives.

    All ranks must call this at the same boundary. Process crashes and failures
    inside backend collectives remain the distributed launcher's responsibility.
    """
    if (
        not torch.distributed.is_initialized()
        or torch.distributed.get_world_size() == 1
    ):
        if error is not None:
            raise error
        return
    device = (
        torch.device("cuda", torch.cuda.current_device())
        if torch.distributed.get_backend() == "nccl"
        else torch.device("cpu")
    )
    failed = torch.tensor(int(error is not None), device=device)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        messages = [None] * torch.distributed.get_world_size()
        message = None if error is None else f"{type(error).__name__}: {error}"
        torch.distributed.all_gather_object(messages, message)
        details = "; ".join(
            f"rank {rank}: {text}" for rank, text in enumerate(messages) if text
        )
        raise RuntimeError(f"{operation} failed: {details}") from error


def active_world_size() -> int:
    """Return the active distributed world size without initializing a backend."""

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_world_size())

    raw_world_size = os.environ.get("WORLD_SIZE")
    if raw_world_size is None:
        return 1
    try:
        return max(int(raw_world_size), 1)
    except ValueError:
        return 1


def require_single_process(operation: str) -> None:
    """Reject driver-only work while still inside a distributed worker group."""

    world_size = active_world_size()
    if world_size != 1:
        raise RuntimeError(
            f"{operation} is a driver-only operation and must run after distributed "
            f"workers exit; active world size is {world_size}"
        )


class DistributedRuntime:
    """Own process/rank lifecycle for one training job.

    The runtime is intentionally thin: Accelerate still owns collectives and
    backend preparation, while this class makes process ownership explicit
    before ``Trainer`` exists. That allows model/cache materialization to happen
    with local/global-main-process ordering and lets the same Accelerator be
    reused by Trainer instead of constructing a second distributed runtime.
    """

    def __init__(self, config: TrainingConfig) -> None:
        config.validate()
        self.config = config
        self.accelerator = build_accelerator(config)

    @property
    def process_index(self) -> int:
        return int(self.accelerator.process_index)

    @property
    def local_process_index(self) -> int:
        return int(self.accelerator.local_process_index)

    @property
    def num_processes(self) -> int:
        return int(self.accelerator.num_processes)

    @property
    def is_main_process(self) -> bool:
        return bool(self.accelerator.is_main_process)

    @property
    def is_local_main_process(self) -> bool:
        return bool(self.accelerator.is_local_main_process)

    def wait_for_everyone(self) -> None:
        if self.num_processes > 1:
            self.accelerator.wait_for_everyone()

    def main_process_first(self) -> AbstractContextManager[None]:
        return (
            self.accelerator.main_process_first()
            if self.num_processes > 1
            else nullcontext()
        )

    def local_main_process_first(self) -> AbstractContextManager[None]:
        return (
            self.accelerator.local_main_process_first()
            if self.num_processes > 1
            else nullcontext()
        )

    def ensure_config(self, config: TrainingConfig) -> None:
        if config != self.config:
            raise ValueError(
                "Trainer config must match the TrainingConfig used to create "
                "DistributedRuntime; one runtime owns one training-job backend"
            )

    def materialize_path(
        self,
        destination: str | Path,
        materialize: Callable[[Path], Any],
        *,
        scope: MaterializationScope = "local",
    ) -> Path:
        """Materialize a deterministic path once per node or once per job.

        ``scope='local'`` is the default for node-local disks: local rank 0 on
        every node performs the copy/download. ``scope='global'`` is for shared
        filesystems visible from every node: only global rank 0 performs it.

        The callback must write to ``destination``. Followers do not execute the
        callback; they wait until the owning process has finished and then use
        the same deterministic path in their own filesystem namespace.
        """

        if scope not in {"local", "global"}:
            raise ValueError("materialization scope must be 'local' or 'global'")

        destination = Path(destination)
        if scope == "local":
            owner = self.is_local_main_process
            first_context = self.local_main_process_first()
        else:
            owner = self.is_main_process
            first_context = self.main_process_first()

        error = None
        with first_context:
            if owner:
                try:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    materialize(destination)
                    if not destination.exists():
                        raise FileNotFoundError(
                            f"Materialization callback did not create {destination}"
                        )
                except Exception as exc:
                    error = exc

        raise_if_any_rank_failed(error, "Materialization")

        # ``*_main_process_first`` releases followers after the owner exits the
        # block. This second barrier makes the return boundary explicit: every
        # rank has observed the completed materialization before any rank moves
        # on to model construction or training.
        self.wait_for_everyone()
        error = None
        if not destination.exists():
            error = FileNotFoundError(
                "Materialized path is not visible to this rank. Check whether a "
                "global materialization was incorrectly used with node-local "
                f"storage: {destination}"
            )
        raise_if_any_rank_failed(error, "Materialization visibility")
        return destination

    @contextmanager
    def model_loading_context(self):
        """Load full weights before adapter injection and backend preparation.

        Accelerate can register a Transformers ZeRO-3 config even with
        ``zero3_init_flag=False``. Temporarily suspend that loading hook, then
        restore it for PEFT collectives and training. This does not implement
        initialization-time sharding; the initial model must fit on each rank.
        Use this context for custom ``from_pretrained`` calls made after the
        runtime is created. ``load_causal_lm`` applies it automatically.
        """
        ds = self.config.deepspeed
        if ds is None or ds.zero_stage != 3:
            yield
            return

        from transformers.integrations.deepspeed import (
            is_deepspeed_zero3_enabled,
            set_hf_deepspeed_config,
            unset_hf_deepspeed_config,
        )

        if not is_deepspeed_zero3_enabled():
            yield
            return

        hf_config = self.accelerator.state.deepspeed_plugin.dschf
        unset_hf_deepspeed_config()
        try:
            yield
        finally:
            set_hf_deepspeed_config(hf_config)

    def load_causal_lm(
        self,
        model_name_or_path: str,
        *,
        tokenizer_name_or_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer_kwargs: dict[str, Any] | None = None,
    ) -> CausalLMComponents:
        """Load one model object per rank while letting local rank 0 go first.

        For Hugging Face Hub identifiers this causes local rank 0 to populate
        the node-local HF cache before sibling ranks call ``from_pretrained``.
        All ranks still construct their own model object, which is required for
        DDP/DeepSpeed; only remote/cache materialization is ordered.
        """

        with self.local_main_process_first(), self.model_loading_context():
            components = load_causal_lm(
                model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        self.wait_for_everyone()
        return components
