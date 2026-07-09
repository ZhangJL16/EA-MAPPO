#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
RUN_NAME="${RUN_NAME:-energy_aware_greedy_$(date +%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-logs/uav_energy_delivery_comparisons/$RUN_NAME}"
mkdir -p "$LOG_DIR"

CMD=(
  "$PYTHON_BIN" scripts/UAVEnergyDeliveryLevel/comparisons/run_heuristic_baseline.py
  --method energy_aware_greedy
  --episodes "${EPISODES:-20}"
  --seed "${SEED:-123}"
  --episode_limit "${EPISODE_LIMIT:-400}"
  --uav_n_agents "${UAV_N_AGENTS:-4}"
  --uav_total_orders "${UAV_TOTAL_ORDERS:-16}"
  --uav_max_active_orders "${UAV_MAX_ACTIVE_ORDERS:-8}"
  --charge_threshold "${CHARGE_THRESHOLD:-0.35}"
  --charge_release_threshold "${CHARGE_RELEASE_THRESHOLD:-0.65}"
  --energy_reserve_ratio "${ENERGY_RESERVE_RATIO:-0.04}"
  --output_csv "$LOG_DIR/heuristic_eval.csv"
  "$@"
)

printf "%q " "${CMD[@]}" > "$LOG_DIR/train.cmd"
echo >> "$LOG_DIR/train.cmd"
echo "METHOD=energy_aware_greedy"
echo "LOG=$LOG_DIR/train.log"
echo "CSV=$LOG_DIR/heuristic_eval.csv"
printf "COMMAND=%s\n" "$(cat "$LOG_DIR/train.cmd")"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${CMD[@]}" > "$LOG_DIR/train.log" 2>&1
