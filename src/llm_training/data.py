from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any


def rows_to_column_batch(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[Any]]:
    """Convert PyTorch DataLoader rows into the column-batch shape used by llm-training-data."""

    if not rows:
        raise ValueError("Cannot collate an empty row batch")

    keys = tuple(rows[0].keys())
    expected = set(keys)
    for index, row in enumerate(rows[1:], start=1):
        if set(row.keys()) != expected:
            raise ValueError(
                f"Row {index} has keys {sorted(row.keys())}, expected {sorted(expected)}"
            )

    return {key: [row[key] for row in rows] for key in keys}


def adapt_column_batch_collator(
    collator: Callable[[Mapping[str, Sequence[Any]]], dict[str, Any]],
) -> Callable[[Sequence[Mapping[str, Any]]], dict[str, Any]]:
    """Adapt an llm-training-data collator for torch.utils.data.DataLoader."""

    def wrapped(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return collator(rows_to_column_batch(rows))

    return wrapped
