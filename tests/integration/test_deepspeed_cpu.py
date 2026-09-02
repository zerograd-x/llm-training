from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest


pytest.importorskip("deepspeed")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _run_worker(mode: str, output_dir: Path) -> subprocess.CompletedProcess[str]:
    worker = Path(__file__).with_name("deepspeed_cpu_worker.py")
    env = os.environ.copy()
    env.update(
        {
            "ACCELERATE_USE_DEEPSPEED": "true",
            "DS_ACCELERATOR": "cpu",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "WORLD_SIZE": "1",
            "LOCAL_WORLD_SIZE": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(_free_port()),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    return subprocess.run(
        [sys.executable, str(worker), mode, str(output_dir)],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"DeepSpeed CPU worker failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )


@pytest.mark.integration
def test_real_deepspeed_cpu_train_resume_and_export(tmp_path: Path) -> None:
    first = _run_worker("interrupt", tmp_path)
    _assert_success(first)

    first_result = json.loads((tmp_path / "interrupt_result.json").read_text())
    assert first_result["global_step"] == 1
    assert "DeepSpeedEngine" in first_result["engine"]
    assert "DeepSpeedOptimizerWrapper" in first_result["optimizer"]
    assert "DeepSpeedSchedulerWrapper" in first_result["scheduler"]

    resumed = _run_worker("resume", tmp_path)
    _assert_success(resumed)

    resume_result = json.loads((tmp_path / "resume_result.json").read_text())
    assert resume_result["global_step"] == 2
    assert "DeepSpeedEngine" in resume_result["engine"]
    assert "DeepSpeedOptimizerWrapper" in resume_result["optimizer"]
    assert "DeepSpeedSchedulerWrapper" in resume_result["scheduler"]
