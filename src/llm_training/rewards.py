from __future__ import annotations

"""Small, task-neutral reward functions for GRPO examples and smoke tests."""

import re
from typing import Any, Sequence


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        parts: list[str] = []
        for message in value:
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                parts.append(message["content"])
            else:
                raise TypeError(
                    "Chat completions must contain mappings with string content"
                )
        return "".join(parts)
    raise TypeError("Completions must be strings or chat-message sequences")


def exact_match_reward(
    *,
    completions,
    answer=None,
    reference=None,
    strip: bool = True,
    case_sensitive: bool = True,
    **kwargs,
) -> list[float]:
    """Score 1 for an exact answer match and 0 otherwise.

    The dataset may use either an answer or reference column.
    """

    targets = answer if answer is not None else reference
    if targets is None:
        raise ValueError(
            "exact_match_reward requires an 'answer' or 'reference' dataset column"
        )
    if len(completions) != len(targets):
        raise ValueError("Completion and target counts must match")

    def normalize(text: str) -> str:
        text = text.strip() if strip else text
        return text if case_sensitive else text.casefold()

    return [
        float(normalize(_completion_text(completion)) == normalize(str(target)))
        for completion, target in zip(completions, targets)
    ]


class RegexReward:
    """Score whether each completion fully matches a configured regular expression."""

    def __init__(self, pattern: str, flags: int = 0) -> None:
        if not pattern:
            raise ValueError("RegexReward.pattern must be non-empty")
        self._pattern = re.compile(pattern, flags)

    def __call__(self, *, completions, **kwargs) -> list[float]:
        return [
            float(self._pattern.fullmatch(_completion_text(completion)) is not None)
            for completion in completions
        ]
