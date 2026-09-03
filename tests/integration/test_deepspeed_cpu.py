from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest
import torch
from transformers import AutoModelForCausalLM

from llm_training import merge_lora


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


def _assert_offline_merge_matches_worker(
    *,
    resume_result: dict,
    output_dir: Path,
    merged_name: str,
) -> None:
    base_dir = Path(resume_result["base_dir"])
    adapter_dir = Path(resume_result["adapter_dir"])
    assert (adapter_dir / "adapter_config.json").is_file()
    assert (adapter_dir / "adapter_model.safetensors").is_file()

    # The merge runs in the parent pytest process after both DeepSpeed worker
    # processes have exited. It is deliberately outside the distributed engine.
    merged_dir = merge_lora(base_dir, adapter_dir, output_dir / merged_name)
    merged_model = AutoModelForCausalLM.from_pretrained(merged_dir)
    merged_model.eval()
    fixed_input = torch.tensor([[1, 4, 5, 2]], dtype=torch.long)
    with torch.no_grad():
        merged_logits = merged_model(input_ids=fixed_input).logits.detach().cpu()

    reference_logits = torch.tensor(resume_result["reference_logits"])
    torch.testing.assert_close(merged_logits, reference_logits, rtol=1e-5, atol=1e-5)


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


@pytest.mark.integration
def test_real_deepspeed_cpu_lora_resume_save_adapter_and_offline_merge(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("peft") is None:
        pytest.skip("PEFT is not installed")

    first = _run_worker("lora_interrupt", tmp_path)
    _assert_success(first)

    first_result = json.loads((tmp_path / "lora_interrupt_result.json").read_text())
    assert first_result["global_step"] == 1
    assert "DeepSpeedEngine" in first_result["engine"]
    assert first_result["trainable_names"]
    assert all("lora_" in name for name in first_result["trainable_names"])

    resumed = _run_worker("lora_resume", tmp_path)
    _assert_success(resumed)

    resume_result = json.loads((tmp_path / "lora_resume_result.json").read_text())
    assert resume_result["global_step"] == 2
    assert "DeepSpeedEngine" in resume_result["engine"]
    assert resume_result["trainable_names"]
    assert all("lora_" in name for name in resume_result["trainable_names"])
    assert not (
        Path(resume_result["adapter_dir"]) / "base_trainable_state.safetensors"
    ).exists()

    _assert_offline_merge_matches_worker(
        resume_result=resume_result,
        output_dir=tmp_path,
        merged_name="merged-model",
    )


@pytest.mark.integration
def test_real_deepspeed_cpu_hybrid_lora_resume_save_and_offline_merge(
    tmp_path: Path,
) -> None:
    if importlib.util.find_spec("peft") is None:
        pytest.skip("PEFT is not installed")

    first = _run_worker("hybrid_lora_interrupt", tmp_path)
    _assert_success(first)

    first_result = json.loads(
        (tmp_path / "hybrid_lora_interrupt_result.json").read_text()
    )
    assert first_result["global_step"] == 1
    assert "DeepSpeedEngine" in first_result["engine"]
    assert any("lora_" in name for name in first_result["trainable_names"])
    assert any("lora_" not in name for name in first_result["trainable_names"])

    resumed = _run_worker("hybrid_lora_resume", tmp_path)
    _assert_success(resumed)

    resume_result = json.loads(
        (tmp_path / "hybrid_lora_resume_result.json").read_text()
    )
    assert resume_result["global_step"] == 2
    assert "DeepSpeedEngine" in resume_result["engine"]
    assert any("lora_" in name for name in resume_result["trainable_names"])
    assert any("lora_" not in name for name in resume_result["trainable_names"])

    adapter_dir = Path(resume_result["adapter_dir"])
    assert (adapter_dir / "base_trainable_state.safetensors").is_file()

    _assert_offline_merge_matches_worker(
        resume_result=resume_result,
        output_dir=tmp_path,
        merged_name="hybrid-merged-model",
    )
