#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_uav_delivery_method.sh <alg> [main.py args...]
  ./run_uav_delivery_method.sh --alg <alg> [main.py args...]

Examples:
  ./run_uav_delivery_method.sh mappo --n_steps 600000 --evaluate_cycle 20
  ./run_uav_delivery_method.sh --alg qmix --gpu_id 1 --n_steps 200000
  EXPERIMENT_DEVICE=lab RUN_DIR=logs/my_run ./run_uav_delivery_method.sh vdn_safe_Comm

Defaults added when omitted:
  --map UAVDelivery
  --uav_n_agents 4
  --uav_total_orders 8
  --uav_max_active_orders 4
  --cuda True
  --gpu_id ${GPU_ID:-0}
  --experiment_device ${EXPERIMENT_DEVICE:-lab}
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python3}"
GPU_ID="${GPU_ID:-0}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-lab}"
export MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
RUN_DIR="${RUN_DIR:-logs/uav_delivery_single_method/$(date +%Y%m%d_%H%M%S)}"

ALG=""
USER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --alg)
      if [[ $# -lt 2 ]]; then
        echo "Missing value after --alg" >&2
        exit 2
      fi
      ALG="$2"
      shift 2
      ;;
    --alg=*)
      ALG="${1#--alg=}"
      shift
      ;;
    *)
      if [[ -z "$ALG" && "$1" != --* ]]; then
        ALG="$1"
      else
        USER_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "$ALG" ]]; then
  usage >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python3 or create .venv/bin/python3 first." >&2
  exit 2
fi

has_arg() {
  local flag="$1"
  local arg
  for arg in "${USER_ARGS[@]}"; do
    if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

DEFAULT_ARGS=()
has_arg --map || DEFAULT_ARGS+=(--map UAVDelivery)
has_arg --uav_n_agents || DEFAULT_ARGS+=(--uav_n_agents 4)
has_arg --uav_total_orders || DEFAULT_ARGS+=(--uav_total_orders 8)
has_arg --uav_max_active_orders || DEFAULT_ARGS+=(--uav_max_active_orders 4)
has_arg --cuda || DEFAULT_ARGS+=(--cuda True)
has_arg --gpu_id || DEFAULT_ARGS+=(--gpu_id "$GPU_ID")
has_arg --experiment_device || DEFAULT_ARGS+=(--experiment_device "$EXPERIMENT_DEVICE")

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/${ALG}.log"
CMD_FILE="$RUN_DIR/${ALG}.cmd"

cmd=(
  "$PYTHON_BIN" main.py
  --alg "$ALG"
  "${DEFAULT_ARGS[@]}"
  "${USER_ARGS[@]}"
)

printf "%q " "${cmd[@]}" > "$CMD_FILE"
printf "\n" >> "$CMD_FILE"

echo "Run directory: $RUN_DIR"
echo "Algorithm: $ALG"
echo "Experiment device: $EXPERIMENT_DEVICE"
echo "Log: $LOG_FILE"
printf "Command: "
printf "%q " "${cmd[@]}"
printf "\n"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" > "$LOG_FILE" 2>&1
