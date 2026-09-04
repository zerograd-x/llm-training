from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass
class TrainerState:
    """Logical position of the next batch to execute."""

    global_step: int = 0
    epoch: int = 0
    batch_in_epoch: int = 0
    last_eval_step: int = -1
    last_eval_epoch: int = -1

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "TrainerState":
        data = json.loads(Path(path).read_text())
        return cls(
            global_step=int(data["global_step"]),
            epoch=int(data["epoch"]),
            batch_in_epoch=int(data["batch_in_epoch"]),
            last_eval_step=int(data.get("last_eval_step", -1)),
            last_eval_epoch=int(data.get("last_eval_epoch", -1)),
        )
