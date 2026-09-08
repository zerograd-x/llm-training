from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .inspection import build_training_plan, format_training_plan
from .training import training_config_from_dict


def _count_jsonl(path: Path) -> int:
    count = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    if count <= 0:
        raise ValueError(f"Training data is empty: {path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve a TrainingConfig into its effective execution plan."
    )
    parser.add_argument("config", type=Path, help="TrainingConfig JSON file")
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--train-data", type=Path, help="JSONL training data to count")
    size.add_argument("--train-examples", type=int, help="Known training example count")
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.environ.get("WORLD_SIZE", "1")),
        help="Distributed world size; defaults to WORLD_SIZE or 1",
    )
    parser.add_argument("--model", default=None, help="Model name/path for display")
    parser.add_argument("--max-sequence-length", type=int, default=None)
    parser.add_argument(
        "--has-eval-data",
        action="store_true",
        help="Include scheduled evaluation-point accounting",
    )
    args = parser.parse_args()

    payload = json.loads(args.config.read_text(encoding="utf-8"))
    config = training_config_from_dict(payload)
    train_examples = args.train_examples
    if args.train_data is not None:
        train_examples = _count_jsonl(args.train_data)

    output = Path(config.output_dir)
    output_exists = output.exists() and any(output.iterdir())
    plan = build_training_plan(
        config,
        train_examples=train_examples,
        world_size=args.world_size,
        model_name=args.model,
        max_sequence_length=args.max_sequence_length,
        has_eval_data=args.has_eval_data,
        output_exists=output_exists,
    )
    print(format_training_plan(plan))


if __name__ == "__main__":
    main()
