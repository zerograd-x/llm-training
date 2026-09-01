from __future__ import annotations

import json
from pathlib import Path
import re
import shutil

from accelerate import Accelerator
from transformers import PreTrainedTokenizerBase

from ..config import TrainingConfig
from .state import TrainerState

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def list_checkpoints(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    if not root.exists():
        return []
    checkpoints = [
        path
        for path in root.iterdir()
        if path.is_dir() and _checkpoint_step(path) is not None
    ]
    return sorted(checkpoints, key=lambda path: _checkpoint_step(path) or -1)


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    checkpoints = list_checkpoints(output_dir)
    return checkpoints[-1] if checkpoints else None


def resolve_resume_checkpoint(output_dir: str | Path, resume_from: str) -> Path:
    if resume_from == "latest":
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint found under {output_dir}")
        return latest
    path = Path(resume_from)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    return path


def save_training_checkpoint(
    accelerator: Accelerator,
    *,
    output_dir: str | Path,
    state: TrainerState,
    config: TrainingConfig,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Path:
    """Save runtime state plus the logical trainer position.

    Accelerator persists prepared model/optimizer/scheduler state, scaler state
    when applicable, and RNG state. This package adds trainer position and
    configuration metadata needed to resume the logical loop.
    """

    checkpoint_dir = Path(output_dir) / f"checkpoint-{state.global_step:08d}"
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        state.save(checkpoint_dir / "trainer_state.json")
        (checkpoint_dir / "training_config.json").write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        if tokenizer is not None:
            tokenizer.save_pretrained(checkpoint_dir / "tokenizer")
        _prune_checkpoints(Path(output_dir), config.checkpoint.keep_last)

    accelerator.wait_for_everyone()
    return checkpoint_dir


def load_training_checkpoint(
    accelerator: Accelerator,
    checkpoint_dir: str | Path,
) -> TrainerState:
    checkpoint_dir = Path(checkpoint_dir)
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"Missing trainer state: {state_path}")

    accelerator.load_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    return TrainerState.load(state_path)


def _prune_checkpoints(output_dir: Path, keep_last: int | None) -> None:
    if keep_last is None:
        return
    checkpoints = list_checkpoints(output_dir)
    for path in checkpoints[:-keep_last]:
        shutil.rmtree(path)
