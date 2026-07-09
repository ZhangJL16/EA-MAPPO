#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_common.sh"

USER_ARGS=("$@")
DEFAULT_ARGS=()

sanitize_label() {
  printf "%s" "$1" | tr -c 'A-Za-z0-9_.-' '_'
}

EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-$(sanitize_label "$(hostname)")}}"
EXPERIMENT_LOG_CSV="${EXPERIMENT_LOG_CSV:-train_logs/$(sanitize_label "$EXPERIMENT_DEVICE")_train_log.csv}"

append_default_arg DEFAULT_ARGS --alg hmappo "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --map UAVEnergyDeliveryLevel "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_n_agents 4 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --episode_limit 400 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_total_orders 16 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_max_active_orders 8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_meta_period 5 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_oracle_high_level True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_freeze_high_level True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_freeze_low_level False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_meta_update_on_subgoal_done False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_delivery_intrinsic_progress_bonus 0.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_intrinsic_collision_penalty 0.8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_dense_reward_scale 0.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_shield_enabled False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_energy_margin_loss_coef 0.0 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_queue_enabled False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_margin 0.04 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --n_steps 180000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_cycle 5000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_epoch 20 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --cuda True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --gpu_id "${GPU_ID:-0}" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_device "$EXPERIMENT_DEVICE" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_log_csv "$EXPERIMENT_LOG_CSV" "${USER_ARGS[@]}"

comparison_run_command \
  "hmappo_low_pretrain_oracle" \
  "main_level.py" \
  "${DEFAULT_ARGS[@]}" \
  "${USER_ARGS[@]}"
