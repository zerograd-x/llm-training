"""Independent processes exercise real collectives, resume and deliberate failures."""

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    sys.platform == "darwin",
    reason="Accelerate Gloo barriers select the unsupported MPS backend on macOS; run on Linux",
)
@pytest.mark.integration
@pytest.mark.parametrize(
    "case",
    [
        "boundary_ddp",
        "boundary_epoch_ddp",
        "failure_rank",
        "failure_materialize",
        "failure_checkpoint",
        "failure_masked",
    ],
)
def test_distributed_hardening(tmp_path, case):
    worker = Path(__file__).with_name("hardening_worker.py")
    env = dict(
        os.environ,
        ACCELERATE_USE_CPU="true",
        OMP_NUM_THREADS="1",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
    )
    env.pop("ACCELERATE_USE_DEEPSPEED", None)
    phases = (
        ("prepare", "baseline", "verify")
        if case.startswith("failure_")
        else ("prepare", "baseline", "interrupt", "resume", "verify")
    )
    for phase in phases:
        command = [sys.executable]
        if phase not in ("prepare", "verify"):
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            command += [
                "-m",
                "torch.distributed.run",
                "--nproc_per_node=2",
                "--master_addr=127.0.0.1",
                f"--master_port={port}",
            ]
        command += [str(worker), case, phase, str(tmp_path)]
        with (tmp_path / f"{phase}.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                pytest.fail(
                    f'{case}/{phase} timed out: {(tmp_path / f"{phase}.log").read_text()}'
                )
        negative = case.startswith("failure_") and phase == "baseline"
        assert code != 0 if negative else code == 0, (
            tmp_path / f"{phase}.log"
        ).read_text()
        files = (
            [f"{phase}.json"]
            if phase in ("prepare", "verify")
            else [f"{phase}-rank-{rank}.json" for rank in range(2)]
        )
        for filename in files:
            payload = json.loads((tmp_path / case / filename).read_text())
            assert payload["ok"], payload
            if negative:
                assert payload["expected_failure"], payload
