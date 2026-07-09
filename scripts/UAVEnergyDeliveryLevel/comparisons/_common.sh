#!/usr/bin/env bash

comparison_repo_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "$script_dir/../../.." >/dev/null 2>&1
  pwd
}

has_arg() {
  local flag="$1"
  shift
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$flag" || "$arg" == "$flag="* ]]; then
      return 0
    fi
  done
  return 1
}

append_default_arg() {
  local -n target_args="$1"
  local flag="$2"
  local value="$3"
  shift 3
  if ! has_arg "$flag" "$@"; then
    target_args+=("$flag" "$value")
  fi
}

comparison_run_command() {
  local method_id="$1"
  local entrypoint="$2"
  shift 2
  local user_args=("$@")

  local repo_root
  repo_root="$(comparison_repo_root)"
  cd "$repo_root"

  local python_bin="${PYTHON_BIN:-.venv/bin/python}"
  if [[ ! -x "$python_bin" ]]; then
    echo "Python executable not found or not executable: $python_bin" >&2
    exit 2
  fi

  local run_stamp="${RUN_STAMP:-$(date +%m%d_%H%M%S)}"
  local run_name="${RUN_NAME:-${method_id}_${run_stamp}}"
  local log_dir="${LOG_DIR:-logs/uav_energy_delivery_comparisons/$run_name}"
  local model_dir="${MODEL_DIR:-model_runs/comparisons/$run_name}"
  mkdir -p "$log_dir" "$model_dir"

  local log_file="$log_dir/train.log"
  local cmd_file="$log_dir/train.cmd"

  local final_args=("${user_args[@]}")
  if ! has_arg --model_dir "${final_args[@]}"; then
    final_args+=(--model_dir "$model_dir")
  fi
  if ! has_arg --replay_dir "${final_args[@]}"; then
    final_args+=(--replay_dir "")
  fi

  local cmd=(
    "$python_bin" "$entrypoint"
    "${final_args[@]}"
  )

  local run_command
  run_command="$(printf "%q " "${cmd[@]}")"
  printf "%s\n" "$run_command" > "$cmd_file"

  export MARL_RUN_SCRIPT="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  export MARL_RUN_COMMAND="$run_command"
  export MARL_EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-dorm}}"

  echo "METHOD=$method_id"
  echo "RUN_NAME=$run_name"
  echo "ENTRYPOINT=$entrypoint"
  echo "LOG=$log_file"
  echo "CMD=$cmd_file"
  echo "MODEL_DIR=$model_dir"
  printf "COMMAND=%s\n" "$run_command"

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  "${cmd[@]}" > "$log_file" 2>&1
}
