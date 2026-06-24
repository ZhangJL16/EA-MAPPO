#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/UAVEnergyDelivery/run_all_methods.sh [main.py args...]

Examples:
  ./scripts/UAVEnergyDelivery/run_all_methods.sh
  ./scripts/UAVEnergyDelivery/run_all_methods.sh --n_steps 600000 --evaluate_cycle 20
  ./scripts/UAVEnergyDelivery/run_all_methods.sh --algs "mappo rgmcomm macpo iql qmix"
  ./scripts/UAVEnergyDelivery/run_all_methods.sh --parallel 3
  DRY_RUN=1 ./scripts/UAVEnergyDelivery/run_all_methods.sh --n_steps 1000
  EXPERIMENT_DEVICE=lab RUN_DIR=logs/my_energy_all ./scripts/UAVEnergyDelivery/run_all_methods.sh

Script options:
  --algs "a b c"      Override the default algorithm list.
  --parallel N        Run up to N algorithms concurrently.
  --max-parallel N    Alias for --parallel.
  -j N                Alias for --parallel.
  --dry-run           Print and record commands without running training.
  -h, --help          Show this help.

Environment overrides:
  ALGS="a b c"               Alternative way to override the algorithm list.
  PYTHON_BIN=.venv/bin/python Python executable.
  GPU_ID=0                   GPU id passed to main.py.
  MARL_EXPERIMENT_DEVICE=dorm Device label written to experiment summary CSV.
  EXPERIMENT_DEVICE=lab      One-command override for this script invocation.
  RUN_DIR=...                Output directory for logs, commands, summary.
  SEED=123                   Training seed.
  EVAL_SEED=100123           Evaluation seed.
  EVALUATE_EPOCH=20          Evaluation episodes per evaluation point.
  MAX_PARALLEL=1             Same as --parallel.
  DRY_RUN=1                  Same as --dry-run.

Defaults added when omitted:
  --map UAVEnergyDelivery
  --uav_n_agents 4
  --uav_total_orders 8
  --uav_max_active_orders 4
  --seed ${SEED:-123}
  --eval_seed ${EVAL_SEED:-$((SEED + 100000))}
  --evaluate_epoch ${EVALUATE_EPOCH:-20}
  --cuda True
  --gpu_id ${GPU_ID:-0}
  --experiment_device ${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN=".venv/bin/python3"
  fi
fi

GPU_ID="${GPU_ID:-0}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}"
SEED="${SEED:-123}"
EVAL_SEED="${EVAL_SEED:-$((SEED + 100000))}"
EVALUATE_EPOCH="${EVALUATE_EPOCH:-20}"
RUN_DIR="${RUN_DIR:-logs/uav_energy_delivery_all_methods/$(date +%Y%m%d_%H%M%S)}"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"
DRY_RUN="${DRY_RUN:-0}"
MAX_PARALLEL="${MAX_PARALLEL:-1}"

DEFAULT_ALGS=(
  # High-priority core methods. In this repo, rgmcomm is the runnable
  # entrypoint that loads policy/maddpg.py.
  mappo
  rgmcomm
  macpo
  iql
  qmix

  # Remaining core policy branches in agent/agent.py.
  vdn
  gmix
  coma
  central_v
  reinforce
  qtran_base
  qtran_alt
  maven
)

ALG_OVERRIDE="${ALGS:-}"
USER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --algs)
      if [[ $# -lt 2 ]]; then
        echo "Missing value after --algs" >&2
        exit 2
      fi
      ALG_OVERRIDE="$2"
      shift 2
      ;;
    --algs=*)
      ALG_OVERRIDE="${1#--algs=}"
      shift
      ;;
    --parallel|--max-parallel|-j)
      if [[ $# -lt 2 ]]; then
        echo "Missing value after $1" >&2
        exit 2
      fi
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --parallel=*|--max-parallel=*)
      MAX_PARALLEL="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *)
      USER_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -n "$ALG_OVERRIDE" ]]; then
  read -r -a ALG_LIST <<< "$ALG_OVERRIDE"
else
  ALG_LIST=("${DEFAULT_ALGS[@]}")
fi

case "$MAX_PARALLEL" in
  ''|*[!0-9]*)
    echo "--parallel must be a positive integer, got: $MAX_PARALLEL" >&2
    exit 2
    ;;
esac
if (( MAX_PARALLEL < 1 )); then
  echo "--parallel must be >= 1, got: $MAX_PARALLEL" >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  echo "Set PYTHON_BIN=/path/to/python or create .venv/bin/python first." >&2
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

arg_value() {
  local flag="$1"
  local default_value="$2"
  local value="$default_value"
  local index=0
  local arg
  while (( index < ${#USER_ARGS[@]} )); do
    arg="${USER_ARGS[$index]}"
    if [[ "$arg" == "$flag="* ]]; then
      value="${arg#*=}"
    elif [[ "$arg" == "$flag" && $((index + 1)) -lt ${#USER_ARGS[@]} ]]; then
      index=$((index + 1))
      value="${USER_ARGS[$index]}"
    fi
    index=$((index + 1))
  done
  printf "%s" "$value"
}

COMMON_ARGS=()
has_arg --map || COMMON_ARGS+=(--map UAVEnergyDelivery)
has_arg --uav_n_agents || COMMON_ARGS+=(--uav_n_agents 4)
has_arg --uav_total_orders || COMMON_ARGS+=(--uav_total_orders 8)
has_arg --uav_max_active_orders || COMMON_ARGS+=(--uav_max_active_orders 4)
has_arg --seed || COMMON_ARGS+=(--seed "$SEED")
has_arg --eval_seed || COMMON_ARGS+=(--eval_seed "$EVAL_SEED")
has_arg --evaluate_epoch || COMMON_ARGS+=(--evaluate_epoch "$EVALUATE_EPOCH")
has_arg --cuda || COMMON_ARGS+=(--cuda True)
has_arg --gpu_id || COMMON_ARGS+=(--gpu_id "$GPU_ID")
has_arg --experiment_device || COMMON_ARGS+=(--experiment_device "$EXPERIMENT_DEVICE")

export MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"

DISPLAY_MAP="$(arg_value --map UAVEnergyDelivery)"
DISPLAY_SEED="$(arg_value --seed "$SEED")"
DISPLAY_EVAL_SEED="$(arg_value --eval_seed "$EVAL_SEED")"
DISPLAY_EVALUATE_EPOCH="$(arg_value --evaluate_epoch "$EVALUATE_EPOCH")"

mkdir -p "$RUN_DIR"
SUMMARY_FILE="$RUN_DIR/summary.tsv"
STATUS_DIR="$RUN_DIR/.status"
rm -rf "$STATUS_DIR"
mkdir -p "$STATUS_DIR"

overall_status=0

echo "Run directory: $RUN_DIR"
echo "Map: $DISPLAY_MAP"
echo "Experiment device: $EXPERIMENT_DEVICE"
echo "Seed: $DISPLAY_SEED"
echo "Evaluation seed: $DISPLAY_EVAL_SEED"
echo "Evaluation episodes: $DISPLAY_EVALUATE_EPOCH"
echo "Python: $PYTHON_BIN"
echo "Parallel jobs: $MAX_PARALLEL"
echo "Algorithms: ${ALG_LIST[*]}"

status_file_for_alg() {
  local alg="$1"
  local safe_alg="${alg//\//_}"
  printf "%s/%s.tsv" "$STATUS_DIR" "$safe_alg"
}

write_status() {
  local alg="$1"
  local status="$2"
  local exit_code="$3"
  local log_file="$4"
  printf "%s\t%s\t%s\t%s\n" "$alg" "$status" "$exit_code" "$log_file" > "$(status_file_for_alg "$alg")"
}

run_one() {
  local alg="$1"
  local log_file run_command exit_code
  local -a cmd
  log_file="$RUN_DIR/${alg}.log"
  cmd=(
    "$PYTHON_BIN" main.py
    --alg "$alg"
    "${COMMON_ARGS[@]}"
    "${USER_ARGS[@]}"
  )

  printf "\n[%s] START %s\n" "$(date '+%F %T')" "$alg"
  run_command="$(printf "%q " "${cmd[@]}")"
  printf "%s\n" "$run_command" > "$RUN_DIR/${alg}.cmd"

  if [[ "$DRY_RUN" == "1" ]]; then
    printf "[DRY_RUN] %s\n" "$run_command"
    write_status "$alg" "dry_run" "0" "$log_file"
    return 0
  fi

  if MARL_RUN_SCRIPT="$SCRIPT_PATH" MARL_RUN_COMMAND="$run_command" "${cmd[@]}" > "$log_file" 2>&1; then
    printf "[%s] DONE  %s\n" "$(date '+%F %T')" "$alg"
    write_status "$alg" "ok" "0" "$log_file"
    return 0
  else
    exit_code=$?
    printf "[%s] FAIL  %s exit_code=%s log=%s\n" "$(date '+%F %T')" "$alg" "$exit_code" "$log_file" >&2
    write_status "$alg" "fail" "$exit_code" "$log_file"
    return "$exit_code"
  fi
}

active_jobs=0
for alg in "${ALG_LIST[@]}"; do
  if (( MAX_PARALLEL == 1 )); then
    if ! run_one "$alg"; then
      overall_status=1
    fi
    continue
  fi

  run_one "$alg" &
  active_jobs=$((active_jobs + 1))
  if (( active_jobs >= MAX_PARALLEL )); then
    if ! wait -n; then
      overall_status=1
    fi
    active_jobs=$((active_jobs - 1))
  fi
done

while (( active_jobs > 0 )); do
  if ! wait -n; then
    overall_status=1
  fi
  active_jobs=$((active_jobs - 1))
done

printf "alg\tstatus\texit_code\tlog\n" > "$SUMMARY_FILE"
for alg in "${ALG_LIST[@]}"; do
  status_file="$(status_file_for_alg "$alg")"
  if [[ -f "$status_file" ]]; then
    cat "$status_file" >> "$SUMMARY_FILE"
  else
    log_file="$RUN_DIR/${alg}.log"
    printf "%s\t%s\t%s\t%s\n" "$alg" "missing" "127" "$log_file" >> "$SUMMARY_FILE"
    overall_status=1
  fi
done

printf "\nSummary: %s\n" "$SUMMARY_FILE"
exit "$overall_status"
