from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.integration
def test_two_rank_distributed_runtime_ownership_and_data_sharding(tmp_path: Path) -> None:
    worker = Path(__file__).with_name("distributed_runtime_worker.py")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, (
        f"2-rank runtime worker failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    rank0 = json.loads((tmp_path / "rank-0.json").read_text())
    rank1 = json.loads((tmp_path / "rank-1.json").read_text())

    assert rank0["process_index"] == 0
    assert rank1["process_index"] == 1
    assert rank0["local_process_index"] == 0
    assert rank1["local_process_index"] == 1
    assert rank0["num_processes"] == rank1["num_processes"] == 2
    assert rank0["is_main_process"] is True
    assert rank1["is_main_process"] is False
    assert rank0["is_local_main_process"] is True
    assert rank1["is_local_main_process"] is False

    # One-node local materialization is owned only by local rank 0. Both ranks
    # then construct their own model object from that completed artifact.
    assert (tmp_path / "node-local-model" / "materialized_by_rank.txt").read_text().strip() == "0"
    assert rank0["parameter_count"] == rank1["parameter_count"]
    assert rank0["parameter_count"] > 0

    assert (tmp_path / "global-main.txt").read_text().strip() == "0"
    assert (tmp_path / "local-main-0.txt").is_file()
    assert not (tmp_path / "local-main-1.txt").exists()

    seen0 = set(rank0["seen_ids"])
    seen1 = set(rank1["seen_ids"])
    assert seen0.isdisjoint(seen1)
    assert seen0 | seen1 == set(range(8))

    # Offline merge/publish work is not allowed to run while WORLD_SIZE > 1.
    assert "driver-only" in rank0["driver_guard_error"]
    assert "driver-only" in rank1["driver_guard_error"]
