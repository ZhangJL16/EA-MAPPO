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
PRETRAINED_LOW_DIR="${PRETRAINED_LOW_DIR:-model_runs/model_runs_low_delivery_pretrain/low_delivery_pretrain_collision08_180k_selected_0704_025743}"
EXPERIMENT_LOG_CSV="${EXPERIMENT_LOG_CSV:-train_logs/$(sanitize_label "$EXPERIMENT_DEVICE")_train_log.csv}"

append_default_arg DEFAULT_ARGS --alg hmappo "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --map UAVEnergyDeliveryLevel "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_n_agents 4 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --episode_limit 400 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_total_orders 16 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --uav_max_active_orders 8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_meta_period 5 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_high_mode_policy continuous "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_pretrained_low_model_dir "$PRETRAINED_LOW_DIR" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_freeze_low_level False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hmappo_freeze_high_level False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_meta_update_on_subgoal_done False "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_intrinsic_collision_penalty 0.8 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_charge_queue_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --hrl_safe_action_guard_enabled True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --n_steps 120000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_cycle 5000 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --evaluate_epoch 20 "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --cuda True "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --gpu_id "${GPU_ID:-0}" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_device "$EXPERIMENT_DEVICE" "${USER_ARGS[@]}"
append_default_arg DEFAULT_ARGS --experiment_log_csv "$EXPERIMENT_LOG_CSV" "${USER_ARGS[@]}"

comparison_run_command \
  "hmappo_warmstart_low180k" \
  "main_level.py" \
  "${DEFAULT_ARGS[@]}" \
  "${USER_ARGS[@]}"
