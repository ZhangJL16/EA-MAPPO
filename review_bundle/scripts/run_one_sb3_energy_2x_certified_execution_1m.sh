#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
SEED="${1:?usage: $0 SEED}"
PHYSICAL_GPU_INDEX="${PHYSICAL_GPU_INDEX:-0}"
OUTPUT_DIR="${ROOT_DIR}/artifacts/phase2_sb3_sac_energy_open_1m_2x_certified_execution/seed${SEED}"
CERTIFICATE_ARTIFACT="${CERTIFICATE_ARTIFACT:-${ROOT_DIR}/certificates/random_persistent_open_energy_x2_certified_execution.json}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU_INDEX}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

exec "${PYTHON_BIN}" "${ROOT_DIR}/scripts/train_sb3_energy_sac_certified_execution.py" \
  --seed "${SEED}" \
  --steps 1000000 \
  --device cuda:0 \
  --certificate-artifact "${CERTIFICATE_ARTIFACT}" \
  --output-dir "${OUTPUT_DIR}"
