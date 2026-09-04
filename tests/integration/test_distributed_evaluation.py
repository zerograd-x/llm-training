import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys

import pytest


@pytest.mark.integration
@pytest.mark.skipif(
    sys.platform == "darwin", reason="Run the Gloo/Accelerate integration on Linux"
)
def test_global_custom_metric_and_collective_callback_errors(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(Path(__file__).with_name("evaluation_worker.py")),
        str(tmp_path),
    ]
    env = dict(
        os.environ,
        ACCELERATE_USE_CPU="true",
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
    )
    env.pop("ACCELERATE_USE_DEEPSPEED", None)
    with (tmp_path / "worker.log").open("w") as log:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            pytest.fail("Distributed metric test timed out")
    assert code == 0, (tmp_path / "worker.log").read_text()
    records = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)
    ]
    assert all(record["ok"] for record in records)
    assert records[0]["metrics"] == records[1]["metrics"]
