#!/usr/bin/env bash
set -uo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./run_uav_energy_delivery_hmappo.sh [main_level.py args...]

Examples:
  ./run_uav_energy_delivery_hmappo.sh
  ./run_uav_energy_delivery_hmappo.sh --gpu_id 1 --n_steps 600000
  META_PERIOD=10 EXPERIMENT_DEVICE=lab RUN_DIR=logs/my_hmappo_run ./run_uav_energy_delivery_hmappo.sh

Defaults added when omitted:
  --alg hmappo
  --map UAVEnergyDeliveryLevel
  --uav_n_agents 4
  --uav_total_orders 8
  --uav_max_active_orders 4
  --hmappo_meta_period ${META_PERIOD:-5}
  --high_lr_actor ${HIGH_LR_ACTOR:-3e-4}
  --high_lr_critic ${HIGH_LR_CRITIC:-3e-4}
  --high_actor_hidden_dim ${HIGH_ACTOR_HIDDEN_DIM:-128}
  --high_critic_hidden_dim ${HIGH_CRITIC_HIDDEN_DIM:-128}
  --seed ${SEED:-123}
  --eval_seed ${EVAL_SEED:-$((SEED + 100000))}
  --evaluate_epoch ${EVALUATE_EPOCH:-20}
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
SEED="${SEED:-123}"
EVAL_SEED="${EVAL_SEED:-$((SEED + 100000))}"
EVALUATE_EPOCH="${EVALUATE_EPOCH:-20}"
META_PERIOD="${META_PERIOD:-5}"
HIGH_LR_ACTOR="${HIGH_LR_ACTOR:-3e-4}"
HIGH_LR_CRITIC="${HIGH_LR_CRITIC:-3e-4}"
HIGH_ACTOR_HIDDEN_DIM="${HIGH_ACTOR_HIDDEN_DIM:-128}"
HIGH_CRITIC_HIDDEN_DIM="${HIGH_CRITIC_HIDDEN_DIM:-128}"
RUN_DIR="${RUN_DIR:-logs/uav_energy_delivery_hmappo/$(date +%Y%m%d_%H%M%S)}"
SCRIPT_PATH="$SCRIPT_DIR/$(basename "${BASH_SOURCE[0]}")"

USER_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    *)
      USER_ARGS+=("$1")
      shift
      ;;
  esac
done

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
has_arg --alg || DEFAULT_ARGS+=(--alg hmappo)
has_arg --map || DEFAULT_ARGS+=(--map UAVEnergyDeliveryLevel)
has_arg --uav_n_agents || DEFAULT_ARGS+=(--uav_n_agents 4)
has_arg --uav_total_orders || DEFAULT_ARGS+=(--uav_total_orders 8)
has_arg --uav_max_active_orders || DEFAULT_ARGS+=(--uav_max_active_orders 4)
has_arg --hmappo_meta_period || DEFAULT_ARGS+=(--hmappo_meta_period "$META_PERIOD")
has_arg --high_lr_actor || DEFAULT_ARGS+=(--high_lr_actor "$HIGH_LR_ACTOR")
has_arg --high_lr_critic || DEFAULT_ARGS+=(--high_lr_critic "$HIGH_LR_CRITIC")
has_arg --high_actor_hidden_dim || DEFAULT_ARGS+=(--high_actor_hidden_dim "$HIGH_ACTOR_HIDDEN_DIM")
has_arg --high_critic_hidden_dim || DEFAULT_ARGS+=(--high_critic_hidden_dim "$HIGH_CRITIC_HIDDEN_DIM")
has_arg --seed || DEFAULT_ARGS+=(--seed "$SEED")
has_arg --eval_seed || DEFAULT_ARGS+=(--eval_seed "$EVAL_SEED")
has_arg --evaluate_epoch || DEFAULT_ARGS+=(--evaluate_epoch "$EVALUATE_EPOCH")
has_arg --cuda || DEFAULT_ARGS+=(--cuda True)
has_arg --gpu_id || DEFAULT_ARGS+=(--gpu_id "$GPU_ID")
has_arg --experiment_device || DEFAULT_ARGS+=(--experiment_device "$EXPERIMENT_DEVICE")

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/hmappo.log"
CMD_FILE="$RUN_DIR/hmappo.cmd"

cmd=(
  "$PYTHON_BIN" main_level.py
  "${DEFAULT_ARGS[@]}"
  "${USER_ARGS[@]}"
)

RUN_COMMAND="$(printf "%q " "${cmd[@]}")"
printf "%s\n" "$RUN_COMMAND" > "$CMD_FILE"

export MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
export MARL_RUN_SCRIPT="$SCRIPT_PATH"
export MARL_RUN_COMMAND="$RUN_COMMAND"

echo "Run directory: $RUN_DIR"
echo "Algorithm: hmappo"
echo "Map: UAVEnergyDeliveryLevel"
echo "Experiment device: $EXPERIMENT_DEVICE"
echo "Seed: $SEED"
echo "Evaluation seed: $EVAL_SEED"
echo "Evaluation episodes: $EVALUATE_EPOCH"
echo "Meta period: $META_PERIOD"
echo "Log: $LOG_FILE"
printf "Command: %s\n" "$RUN_COMMAND"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}" > "$LOG_FILE" 2>&1
