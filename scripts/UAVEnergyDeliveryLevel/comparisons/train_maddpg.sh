#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

USER_ARGS=("$@")
DEFAULT_ARGS=()

# The repository wires policy/maddpg.py through the RGMComm branch.
# This script is the runnable MADDPG-style baseline available in this codebase.
append_default_arg DEFAULT_ARGS --alg RGMComm "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --map UAVEnergyDelivery "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_n_agents 4 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --episode_limit 400 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_total_orders 16 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_max_active_orders 8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --n_steps 600000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --time_steps 600000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_rate 5000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_epoch 20 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_episode_len 400 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --cuda True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --gpu_id "${GPU_ID:-0}" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_device "${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}" "${USER_ARGS[@]}"

comparison_run_command \
  "maddpg_repo_rgm_branch" \
  "main.py" \
  "${DEFAULT_ARGS[@]}" \
  "${USER_ARGS[@]}"
