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
append_default_arg DEFAULT_ARGS --hrl_meta_update_on_subgoal_done False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_order_progress_override 0.65 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_delivery_intrinsic_progress_bonus 0.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_intrinsic_collision_penalty 0.8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_dense_reward_scale 0.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_shield_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_margin_reserve_ratio 0.04 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_margin_loss_coef 0.1 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_margin_charge_beta 0.2 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_queue_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_queue_radius 0.24 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_margin 0.04 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --n_steps 600000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_cycle 5000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_epoch 20 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --cuda True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --gpu_id "${GPU_ID:-0}" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_device "${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}" "${USER_ARGS[@]}"

comparison_run_command \
  "ours_full" \
  "main_level.py" \
  "${DEFAULT_ARGS[@]}" \
  "${USER_ARGS[@]}"
