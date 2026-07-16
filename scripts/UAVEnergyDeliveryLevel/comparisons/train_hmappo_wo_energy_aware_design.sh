#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

USER_ARGS=("$@")
DEFAULT_ARGS=()

append_default_arg DEFAULT_ARGS --alg hmappo "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --map UAVEnergyDeliveryLevel "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_n_agents 4 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --episode_limit 400 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_total_orders 16 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_max_active_orders 8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_meta_period 5 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_high_mode_policy continuous "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_meta_update_on_subgoal_done False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_dense_reward_scale 1.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_queue_enabled False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_enabled False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --n_steps 600000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_cycle 5000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_epoch 20 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --cuda True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --gpu_id "${GPU_ID:-0}" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_device "${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}" "${USER_ARGS[@]}"

comparison_run_command \
  "hmappo_wo_energy_aware_design" \
  "main_level.py" \
  "${DEFAULT_ARGS[@]}" \
  "${USER_ARGS[@]}"
