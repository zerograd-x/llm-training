#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${VENV_DIR:-${REPO_ROOT}/.venv}"
MODEL="${MODEL:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs/qwen35-4b-gpu-smoke}"

if [[ ! -x "${VENV_DIR}/bin/torchrun" ]]; then
  echo "ERROR: ${VENV_DIR}/bin/torchrun not found. Run: bash scripts/bootstrap_gpu.sh" >&2
  exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"${VENV_DIR}/bin/torchrun" --standalone --nproc_per_node=4 \
  "${REPO_ROOT}/tests/gpu/qwen35_4b_single_node.py" \
  --model "${MODEL}" \
  --output "${OUTPUT_DIR}"

"${VENV_DIR}/bin/python" \
  "${REPO_ROOT}/tests/gpu/verify_qwen35_export.py" \
  "${OUTPUT_DIR}/final-model"

echo "PASS qwen35_smoke_complete output=${OUTPUT_DIR}"
