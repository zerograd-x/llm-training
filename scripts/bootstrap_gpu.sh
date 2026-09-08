#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
CONSTRAINTS_FILE="${CONSTRAINTS_FILE:-${REPO_ROOT}/constraints/gpu-tested.txt}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "ERROR: ${PYTHON_BIN} is not installed" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is unavailable; use an NVIDIA GPU image/AMI" >&2
  exit 1
fi

if command -v apt-get >/dev/null 2>&1; then
  packages=()
  if ! "${PYTHON_BIN}" -m venv --help >/dev/null 2>&1; then
    packages+=(python3-venv)
  fi
  command -v git >/dev/null 2>&1 || packages+=(git)
  command -v curl >/dev/null 2>&1 || packages+=(curl)
  command -v aws >/dev/null 2>&1 || packages+=(awscli)

  if ((${#packages[@]})); then
    sudo apt-get update
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
  fi
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: aws CLI is required for --artifact-s3 runs" >&2
  exit 1
fi

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/pip" install -c "${CONSTRAINTS_FILE}" -e "${REPO_ROOT}[deepspeed,dev]"

"${VENV_DIR}/bin/python" - <<'PY'
import sys
import torch
import transformers
import accelerate
import deepspeed
from transformers import Qwen3_5ForCausalLM

print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"accelerate={accelerate.__version__}")
print(f"deepspeed={deepspeed.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_runtime={torch.version.cuda}")
print(f"gpu_count={torch.cuda.device_count()}")

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA is not available")
if torch.cuda.device_count() < 1:
    raise SystemExit("ERROR: no CUDA GPU detected")
if not torch.cuda.is_bf16_supported():
    raise SystemExit("ERROR: BF16-capable GPU is required")

print("Qwen3_5ForCausalLM import=PASS")
PY

echo "GPU bootstrap complete"
echo "Repo root: ${REPO_ROOT}"
echo "Use: ${VENV_DIR}/bin/torchrun ..."
