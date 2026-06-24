#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
GPU_ID="${GPU_ID:-0}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-lab}"
export MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
RUN_DIR="${RUN_DIR:-logs/uav_delivery_all_methods/$(date +%Y%m%d_%H%M%S)}"

if [[ -n "${ALGS:-}" ]]; then
  read -r -a ALG_LIST <<< "$ALGS"
else
  ALG_LIST=(
    # Core policy branches in agent/agent.py.
    mappo
    qmix
    vdn
    iql
    gmix
    coma
    central_v
    reinforce
    qtran_base
    qtran_alt
    maven
    macpo
    rgmcomm

    # CommNet/G2ANet runner branches.
    coma+commnet
    central_v+commnet
    reinforce+commnet
    coma+g2anet
    central_v+g2anet
    reinforce+g2anet

    # Safety, communication plugin, and reshape variants used by this repo.
    mappo_reshape
    mappo_Comm
    mappo_safe
    mappo_safe_Comm
    qmix_Comm
    qmix_safe
    qmix_safe_Comm
    qmix_reshape_Comm
    vdn_Comm
    vdn_safe
    vdn_safe_Comm
    vdn_reshape
    vdn_reshape_Comm
    iql_Comm
    iql_reshape_Comm
    gmix_reshape
    gmix_Comm
    gmix_reshape_Comm
    macpo_Comm
  )
fi

COMMON_ARGS=(
  --map UAVDelivery
  --uav_n_agents 4
  --uav_total_orders 8
  --uav_max_active_orders 4
  --cuda True
  --gpu_id "$GPU_ID"
  --experiment_device "$EXPERIMENT_DEVICE"
)

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python3 or create .venv/bin/python3 first." >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
SUMMARY_FILE="$RUN_DIR/summary.tsv"
printf "alg\tstatus\texit_code\tlog\n" > "$SUMMARY_FILE"

overall_status=0

echo "Run directory: $RUN_DIR"
echo "Experiment device: $EXPERIMENT_DEVICE"
echo "Algorithms: ${ALG_LIST[*]}"

for alg in "${ALG_LIST[@]}"; do
  log_file="$RUN_DIR/${alg}.log"
  cmd=(
    "$PYTHON_BIN" main.py
    --alg "$alg"
    "${COMMON_ARGS[@]}"
    "$@"
  )

  printf "\n[%s] START %s\n" "$(date '+%F %T')" "$alg"
  printf "%q " "${cmd[@]}" > "$RUN_DIR/${alg}.cmd"
  printf "\n" >> "$RUN_DIR/${alg}.cmd"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf "[DRY_RUN] "
    printf "%q " "${cmd[@]}"
    printf "\n"
    printf "%s\t%s\t%s\t%s\n" "$alg" "dry_run" "0" "$log_file" >> "$SUMMARY_FILE"
    continue
  fi

  if "${cmd[@]}" > "$log_file" 2>&1; then
    printf "[%s] DONE  %s\n" "$(date '+%F %T')" "$alg"
    printf "%s\t%s\t%s\t%s\n" "$alg" "ok" "0" "$log_file" >> "$SUMMARY_FILE"
  else
    exit_code=$?
    overall_status=1
    printf "[%s] FAIL  %s exit_code=%s log=%s\n" "$(date '+%F %T')" "$alg" "$exit_code" "$log_file" >&2
    printf "%s\t%s\t%s\t%s\n" "$alg" "fail" "$exit_code" "$log_file" >> "$SUMMARY_FILE"
  fi
done

printf "\nSummary: %s\n" "$SUMMARY_FILE"
exit "$overall_status"
