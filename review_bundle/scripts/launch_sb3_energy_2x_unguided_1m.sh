#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$root"
python_bin="${PYTHON_BIN:-$root/.venv/bin/python}"; gpu="${PHYSICAL_GPU_INDEX:-0}"
output_root="artifacts/phase2_sb3_sac_energy_open_1m_2x_unguided"
[[ ! -e "$output_root" ]] || { echo "refusing to overwrite $output_root" >&2; exit 3; }
gpu_count="$($python_bin -c 'import torch; print(torch.cuda.device_count())')"
[[ "$gpu_count" -gt "$gpu" ]] || { echo "requested GPU $gpu unavailable" >&2; exit 4; }
session="sb3-sac-energy-2x-unguided-1m"
tmux has-session -t "$session" 2>/dev/null && { echo "tmux session exists: $session" >&2; exit 5; }
tmux new-session -d -s "$session" -c "$root" \
  "for seed in 0 1 2; do bash scripts/run_one_sb3_energy_2x_1m.sh unguided \"\$seed\" \"$output_root/seed\$seed\" \"$gpu\" || exit; done"
echo "launched serial seeds=0,1,2 session=$session gpu=$gpu output=$output_root"
