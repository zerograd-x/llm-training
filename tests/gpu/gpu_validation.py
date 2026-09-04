"""Rebuilt single-GPU suite; not a recovered copy of the temporary worker.

The driver uses only the standard library and gives every phase a fresh process.
No case silently turns a failed assertion into a successful suite exit.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

CASES = (
    "lora_bf16", "lora_zero3", "lora_gc", "lora_zero3_gc",
    "tt_bf16", "tt_zero3", "scheduler_zero3", "offload",
)


def phase_environment(case: str, phase: str, root: Path) -> dict[str, str]:
    env = dict(os.environ)
    for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT",
                "ACCELERATE_USE_DEEPSPEED", "ACCELERATE_USE_CPU"):
        env.pop(key, None)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["OMP_NUM_THREADS"] = "1"
    if phase not in {"prepare", "verify"} and ("zero3" in case or case == "offload"):
        env.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0",
                   MASTER_ADDR="127.0.0.1", MASTER_PORT="29781",
                   ACCELERATE_USE_DEEPSPEED="true", DS_ACCELERATOR="cuda")
    if case == "offload":
        # Do not hide a bypass inherited from the calling environment.
        if "DS_SKIP_CUDA_CHECK" in env:
            raise RuntimeError("Unset DS_SKIP_CUDA_CHECK before clean offload validation")
        env["TORCH_EXTENSIONS_DIR"] = str(root / "clean-extensions")
        env["MAX_JOBS"] = "2"
    return env


def run_phase(worker: Path, case: str, phase: str, root: Path,
              timeout: int = 900) -> dict:
    result_path = root / case / f"{phase}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with (result_path.with_suffix(".log")).open("w") as log:
        try:
            done = subprocess.run(
                [sys.executable, str(worker), case, phase, str(root)],
                env=phase_environment(case, phase, root), stdout=log,
                stderr=subprocess.STDOUT, timeout=timeout, check=False,
            )
            payload = json.loads(result_path.read_text()) if result_path.exists() else {}
            ok = done.returncode == 0 and payload.get("ok") is True
            return {"case": case, "phase": phase, "ok": ok,
                    "returncode": done.returncode, "result": payload,
                    "elapsed_seconds": round(time.monotonic() - started, 3)}
        except (subprocess.TimeoutExpired, ValueError, OSError, RuntimeError) as exc:
            return {"case": case, "phase": phase, "ok": False, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new, empty output directory")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This suite must be launched by a single-process driver")
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("Output directory must be empty to prevent stale-result passes")
    root.mkdir(parents=True, exist_ok=True)
    worker = Path(__file__).with_name("fresh_case.py")
    results = []
    for case in dict.fromkeys(args.cases):
        for phase in ("prepare", "baseline", "interrupt", "resume", "verify"):
            result = run_phase(worker, case, phase, root)
            results.append(result)
            (root / "matrix.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps({k: v for k, v in result.items() if k != "result"}), flush=True)
            if not result["ok"]:
                break
    return 0 if results and all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
