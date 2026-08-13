#!/usr/bin/env bash
set -euo pipefail

echo "ERROR: this E1 launcher is superseded; use scripts/run_stage_a_energy_bootstrap.py with a fresh output directory." >&2
exit 2

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
python_bin="${PYTHON_BIN:-/home/zjl/mappo/.venv/bin/python}"
mkdir -p artifacts/new_route/logs artifacts/new_route/pids artifacts/new_route/e1_energy

for seed in 0 1 2; do
  output="artifacts/new_route/e1_energy/seed${seed}"
  checkpoint="artifacts/phase1_sb3_sac_1m_gpu/seed${seed}/checkpoint_step_1000000.zip"
  "$python_bin" scripts/run_new_route_e1.py \
    --seed "$seed" --checkpoint "$checkpoint" --output-dir "$output" \
    --sorties 100 --max-return-steps 800 --training-updates 800 --device cpu \
    --initialize-only
  nohup "$python_bin" scripts/run_new_route_e1.py \
    --seed "$seed" --checkpoint "$checkpoint" --output-dir "$output" \
    --sorties 100 --max-return-steps 800 --training-updates 800 --device cpu \
    > "artifacts/new_route/logs/e1_seed${seed}.log" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "artifacts/new_route/pids/e1_seed${seed}.pid"
done
