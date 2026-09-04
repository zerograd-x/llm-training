from __future__ import annotations

import pytest

from llm_training import (
    active_world_size,
    merge_lora,
    merge_trainable_tokens,
    require_single_process,
)


def test_active_world_size_defaults_to_one(monkeypatch) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert active_world_size() == 1
    require_single_process("single-process operation")


def test_require_single_process_rejects_distributed_environment(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "4")
    assert active_world_size() == 4
    with pytest.raises(RuntimeError, match="driver-only"):
        require_single_process("offline merge")


@pytest.mark.parametrize("merge_fn", [merge_lora, merge_trainable_tokens])
def test_merge_apis_reject_worker_context_before_loading_artifacts(
    monkeypatch,
    tmp_path,
    merge_fn,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RuntimeError, match="driver-only"):
        merge_fn(
            tmp_path / "missing-base",
            tmp_path / "missing-adapter",
            tmp_path / "output",
        )
