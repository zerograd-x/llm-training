from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
import os
from pathlib import Path
from typing import Any, Literal

from accelerate import Accelerator
import torch

from .backend import build_accelerator
from .config import TrainingConfig
from .model import CausalLMComponents, load_causal_lm


MaterializationScope = Literal["global", "local"]


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
        self.accelerator.wait_for_everyone()

    def main_process_first(self) -> AbstractContextManager[None]:
        return self.accelerator.main_process_first()

    def local_main_process_first(self) -> AbstractContextManager[None]:
        return self.accelerator.local_main_process_first()

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

        with first_context:
            if owner:
                destination.parent.mkdir(parents=True, exist_ok=True)
                materialize(destination)
                if not destination.exists():
                    raise FileNotFoundError(
                        "Materialization callback completed without creating "
                        f"destination: {destination}"
                    )

        # ``*_main_process_first`` releases followers after the owner exits the
        # block. This second barrier makes the return boundary explicit: every
        # rank has observed the completed materialization before any rank moves
        # on to model construction or training.
        self.wait_for_everyone()
        if not destination.exists():
            raise FileNotFoundError(
                "Materialized path is not visible to this rank. Check whether a "
                "global materialization was incorrectly used with node-local "
                f"storage: {destination}"
            )
        return destination

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

        with self.local_main_process_first():
            components = load_causal_lm(
                model_name_or_path,
                tokenizer_name_or_path=tokenizer_name_or_path,
                model_kwargs=model_kwargs,
                tokenizer_kwargs=tokenizer_kwargs,
            )
        self.wait_for_everyone()
        return components
