#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_PREFIX="${SESSION_PREFIX:-sb3-sac-energy-2x-certified-execution}"

for seed in 0 1 2; do
  session="${SESSION_PREFIX}-seed${seed}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT_DIR}' && PHYSICAL_GPU_INDEX='${PHYSICAL_GPU_INDEX:-0}' bash scripts/run_one_sb3_energy_2x_certified_execution_1m.sh '${seed}' > 'artifacts/phase2_sb3_sac_energy_open_1m_2x_certified_execution_seed${seed}.log' 2>&1"
  echo "started ${session}"
done
