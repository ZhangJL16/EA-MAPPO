#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SUITE_ID="${SUITE_ID:-uedl_hmappo_twostage_$(date +%m%d_%H%M%S)}"
COMPUTER_ID="${COMPUTER_ID:-$(hostname | tr -c 'A-Za-z0-9_.-' '_')}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-2}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
CUDA="${CUDA:-True}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-180000}"
FINETUNE_STEPS="${FINETUNE_STEPS:-120000}"
EVALUATE_CYCLE="${EVALUATE_CYCLE:-5000}"
EVALUATE_EPOCH="${EVALUATE_EPOCH:-20}"
SEEDS="${SEEDS:-123,456,789,101112,131415,161718,192021,222324}"
UAV_N_AGENTS="${UAV_N_AGENTS:-4}"
UAV_TOTAL_ORDERS="${UAV_TOTAL_ORDERS:-24}"
UAV_MAX_ACTIVE_ORDERS="${UAV_MAX_ACTIVE_ORDERS:-8}"
EPISODE_LIMIT="${EPISODE_LIMIT:-600}"
UAV_ENERGY_DECAY="${UAV_ENERGY_DECAY:-0.5}"
UAV_CHARGING_RATE="${UAV_CHARGING_RATE:-4.0}"
DRY_RUN="${DRY_RUN:-0}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-$COMPUTER_ID}"
EXPERIMENT_LOG_CSV="${EXPERIMENT_LOG_CSV:-train_logs/${EXPERIMENT_DEVICE}_train_log.csv}"
LOW_PRETRAIN_SCRIPT="${LOW_PRETRAIN_SCRIPT:-scripts/UAVEnergyDeliveryLevel/comparisons/train_hmappo_low_pretrain_oracle.sh}"
FINETUNE_SCRIPT="${FINETUNE_SCRIPT:-scripts/UAVEnergyDeliveryLevel/comparisons/train_hmappo_warmstart_low180k.sh}"

usage() {
  cat <<'EOF'
Usage:
  scripts/UAVEnergyDeliveryLevel/run_hmappo_twostage_2pc.sh [options] [-- extra main_level.py args]

Runs two-stage H-MAPPO without relying on any existing pretrained model:
  1. low-level oracle pretraining
  2. H-MAPPO fine-tuning initialized from the just-trained low-level model

Use the same SUITE_ID on two computers and SHARD_INDEX=0/1 to split seeds.
Each computer writes its own CSV: train_logs/<COMPUTER_ID>_train_log.csv.

Options:
  --suite-id ID              Shared experiment id.
  --computer-id ID           Machine label. Defaults to hostname.
  --num-shards N             Total computers. Defaults to 2.
  --shard-index I            This computer's zero-based shard index.
  -j, --max-parallel N       Concurrent seed pipelines. Defaults to 4.
  --gpu-ids LIST             Comma/space-separated GPU ids. Defaults to 0.
  --seeds LIST               Comma/space-separated seeds.
  --pretrain-steps N         Low-level pretraining steps. Defaults to 180000.
  --finetune-steps N         H-MAPPO fine-tuning steps. Defaults to 120000.
  --cuda True|False          Defaults to True.
  --dry-run                  Print commands and write status without training.
EOF
}

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --suite-id)
      SUITE_ID="$2"
      shift 2
      ;;
    --suite-id=*)
      SUITE_ID="${1#*=}"
      shift
      ;;
    --computer-id)
      COMPUTER_ID="$2"
      shift 2
      ;;
    --computer-id=*)
      COMPUTER_ID="${1#*=}"
      shift
      ;;
    --num-shards)
      NUM_SHARDS="$2"
      shift 2
      ;;
    --num-shards=*)
      NUM_SHARDS="${1#*=}"
      shift
      ;;
    --shard-index)
      SHARD_INDEX="$2"
      shift 2
      ;;
    --shard-index=*)
      SHARD_INDEX="${1#*=}"
      shift
      ;;
    -j|--max-parallel)
      MAX_PARALLEL="$2"
      shift 2
      ;;
    --max-parallel=*)
      MAX_PARALLEL="${1#*=}"
      shift
      ;;
    --gpu-ids)
      GPU_IDS="$2"
      shift 2
      ;;
    --gpu-ids=*)
      GPU_IDS="${1#*=}"
      shift
      ;;
    --seeds)
      SEEDS="$2"
      shift 2
      ;;
    --seeds=*)
      SEEDS="${1#*=}"
      shift
      ;;
    --pretrain-steps|--pretrain_steps)
      PRETRAIN_STEPS="$2"
      shift 2
      ;;
    --pretrain-steps=*|--pretrain_steps=*)
      PRETRAIN_STEPS="${1#*=}"
      shift
      ;;
    --finetune-steps|--finetune_steps|--n-steps|--n_steps)
      FINETUNE_STEPS="$2"
      shift 2
      ;;
    --finetune-steps=*|--finetune_steps=*|--n-steps=*|--n_steps=*)
      FINETUNE_STEPS="${1#*=}"
      shift
      ;;
    --evaluate-cycle|--evaluate_cycle)
      EVALUATE_CYCLE="$2"
      shift 2
      ;;
    --evaluate-cycle=*|--evaluate_cycle=*)
      EVALUATE_CYCLE="${1#*=}"
      shift
      ;;
    --evaluate-epoch|--evaluate_epoch)
      EVALUATE_EPOCH="$2"
      shift 2
      ;;
    --evaluate-epoch=*|--evaluate_epoch=*)
      EVALUATE_EPOCH="${1#*=}"
      shift
      ;;
    --cuda)
      CUDA="$2"
      shift 2
      ;;
    --cuda=*)
      CUDA="${1#*=}"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

require_positive_int() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || "$value" == *[!0-9]* || "$value" -lt 1 ]]; then
    echo "$name must be a positive integer, got: $value" >&2
    exit 2
  fi
}

require_nonnegative_int() {
  local name="$1"
  local value="$2"
  if [[ -z "$value" || "$value" == *[!0-9]* ]]; then
    echo "$name must be a non-negative integer, got: $value" >&2
    exit 2
  fi
}

normalize_list() {
  printf "%s" "$1" | tr ',;' '  ' | xargs -n1
}

require_positive_int MAX_PARALLEL "$MAX_PARALLEL"
require_positive_int NUM_SHARDS "$NUM_SHARDS"
require_nonnegative_int SHARD_INDEX "$SHARD_INDEX"
require_positive_int PRETRAIN_STEPS "$PRETRAIN_STEPS"
require_positive_int FINETUNE_STEPS "$FINETUNE_STEPS"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "SHARD_INDEX must be smaller than NUM_SHARDS." >&2
  exit 2
fi
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi
if [[ ! -x "$LOW_PRETRAIN_SCRIPT" ]]; then
  echo "Low pretrain script not found or not executable: $LOW_PRETRAIN_SCRIPT" >&2
  exit 2
fi
if [[ ! -x "$FINETUNE_SCRIPT" ]]; then
  echo "Fine-tune script not found or not executable: $FINETUNE_SCRIPT" >&2
  exit 2
fi

mapfile -t SEED_LIST < <(normalize_list "$SEEDS")
mapfile -t GPU_LIST < <(normalize_list "$GPU_IDS")
if (( ${#GPU_LIST[@]} == 0 )); then
  GPU_LIST=(0)
fi

SUITE_DIR="logs/uav_energy_delivery_comparison_suites/$SUITE_ID"
RUNS_DIR="$SUITE_DIR/runs"
STATUS_DIR="$SUITE_DIR/status"
mkdir -p "$RUNS_DIR" "$STATUS_DIR" "$(dirname "$EXPERIMENT_LOG_CSV")"

MANIFEST="$SUITE_DIR/manifest_${COMPUTER_ID}.tsv"
printf "suite_id\tcomputer_id\tshard_index\tnum_shards\tmax_parallel\tseeds\tpretrain_steps\tfinetune_steps\tevaluate_cycle\tevaluate_epoch\tcuda\tgpu_ids\texperiment_log_csv\tstarted_at\n" > "$MANIFEST"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$SUITE_ID" "$COMPUTER_ID" "$SHARD_INDEX" "$NUM_SHARDS" "$MAX_PARALLEL" \
  "$SEEDS" "$PRETRAIN_STEPS" "$FINETUNE_STEPS" "$EVALUATE_CYCLE" "$EVALUATE_EPOCH" \
  "$CUDA" "$GPU_IDS" "$EXPERIMENT_LOG_CSV" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MANIFEST"

echo "SUITE_ID=$SUITE_ID"
echo "SUITE_DIR=$SUITE_DIR"
echo "COMPUTER_ID=$COMPUTER_ID"
echo "SHARD=$SHARD_INDEX/$NUM_SHARDS"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "SEEDS=${SEED_LIST[*]}"
echo "PRETRAIN_STEPS=$PRETRAIN_STEPS"
echo "FINETUNE_STEPS=$FINETUNE_STEPS"
echo "EXPERIMENT_LOG_CSV=$EXPERIMENT_LOG_CSV"

active_job_count() {
  jobs -rp | wc -l
}

wait_for_slot() {
  while true; do
    if (( $(active_job_count) < MAX_PARALLEL )); then
      return
    fi
    wait -n || true
  done
}

run_stage() {
  local stage="$1"
  local script="$2"
  local run_name="$3"
  local log_dir="$4"
  local model_dir="$5"
  local gpu_id="$6"
  shift 6
  local args=("$@")

  mkdir -p "$log_dir" "$model_dir"
  echo "START stage=$stage run=$run_name"
  (
    export RUN_NAME="$run_name" LOG_DIR="$log_dir" MODEL_DIR="$model_dir"
    export PYTHON_BIN="$PYTHON_BIN" GPU_ID="$gpu_id"
    export EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE" MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
    export EXPERIMENT_LOG_CSV="$EXPERIMENT_LOG_CSV" MARL_EXPERIMENT_LOG_CSV="$EXPERIMENT_LOG_CSV"
    "$script" "${args[@]}"
  )
}

run_one_pipeline() {
  local work_index="$1"
  local seed="$2"
  local gpu_id="$3"
  local method="hmappo_twostage_lowpretrain"
  local run_prefix="${SUITE_ID}_${COMPUTER_ID}_${method}_seed${seed}"
  local low_run_name="${run_prefix}_low"
  local high_run_name="${run_prefix}_high"
  local low_log_dir="$RUNS_DIR/$low_run_name"
  local high_log_dir="$RUNS_DIR/$high_run_name"
  local low_model_dir="model_runs/twostage/$low_run_name"
  local high_model_dir="model_runs/twostage/$high_run_name"
  local status_file="$STATUS_DIR/status_${COMPUTER_ID}_${work_index}_${method}_seed${seed}.tsv"
  local start_time end_time start_epoch end_epoch elapsed status exit_code low_exit high_exit command_text

  printf "suite_id\tcomputer_id\tshard_index\tnum_shards\twork_index\tmethod\tseed\trun_name\tstatus\texit_code\telapsed_sec\tlow_model_dir\thigh_model_dir\tstart_time\tend_time\tcommand\n" > "$status_file"

  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  start_epoch="$(date +%s)"
  echo "START pipeline seed=$seed gpu=$gpu_id low=$low_run_name high=$high_run_name"

  local eval_seed=$((seed + 100000))
  local common_args=(
    --seed "$seed"
    --eval_seed "$eval_seed"
    --time_steps "$PRETRAIN_STEPS"
    --evaluate_cycle "$EVALUATE_CYCLE"
    --evaluate_rate "$EVALUATE_CYCLE"
    --evaluate_epoch "$EVALUATE_EPOCH"
    --evaluate_episode_len "$EPISODE_LIMIT"
    --episode_limit "$EPISODE_LIMIT"
    --uav_n_agents "$UAV_N_AGENTS"
    --uav_total_orders "$UAV_TOTAL_ORDERS"
    --uav_max_active_orders "$UAV_MAX_ACTIVE_ORDERS"
    --uav_energy_decay "$UAV_ENERGY_DECAY"
    --uav_charging_rate "$UAV_CHARGING_RATE"
    --cuda "$CUDA"
    --gpu_id "$gpu_id"
    --experiment_log_csv "$EXPERIMENT_LOG_CSV"
  )
  local low_args=(
    "${common_args[@]}"
    --n_steps "$PRETRAIN_STEPS"
    "${EXTRA_ARGS[@]}"
  )
  local high_args=(
    --seed "$seed"
    --eval_seed "$eval_seed"
    --n_steps "$FINETUNE_STEPS"
    --time_steps "$FINETUNE_STEPS"
    --evaluate_cycle "$EVALUATE_CYCLE"
    --evaluate_rate "$EVALUATE_CYCLE"
    --evaluate_epoch "$EVALUATE_EPOCH"
    --evaluate_episode_len "$EPISODE_LIMIT"
    --episode_limit "$EPISODE_LIMIT"
    --uav_n_agents "$UAV_N_AGENTS"
    --uav_total_orders "$UAV_TOTAL_ORDERS"
    --uav_max_active_orders "$UAV_MAX_ACTIVE_ORDERS"
    --uav_energy_decay "$UAV_ENERGY_DECAY"
    --uav_charging_rate "$UAV_CHARGING_RATE"
    --cuda "$CUDA"
    --gpu_id "$gpu_id"
    --experiment_log_csv "$EXPERIMENT_LOG_CSV"
    --hmappo_pretrained_low_model_dir "$low_model_dir"
    "${EXTRA_ARGS[@]}"
  )

  command_text="LOW: RUN_NAME=$low_run_name MODEL_DIR=$low_model_dir $LOW_PRETRAIN_SCRIPT ${low_args[*]} ; HIGH: RUN_NAME=$high_run_name MODEL_DIR=$high_model_dir PRETRAINED_LOW_DIR=$low_model_dir $FINETUNE_SCRIPT ${high_args[*]}"

  if [[ "$DRY_RUN" == "1" ]]; then
    (
      export DRY_RUN=1
      run_stage low "$LOW_PRETRAIN_SCRIPT" "$low_run_name" "$low_log_dir" "$low_model_dir" "$gpu_id" "${low_args[@]}"
      export PRETRAINED_LOW_DIR="$low_model_dir"
      run_stage high "$FINETUNE_SCRIPT" "$high_run_name" "$high_log_dir" "$high_model_dir" "$gpu_id" "${high_args[@]}"
    )
    low_exit=0
    high_exit=0
    exit_code=0
    status="dry_run"
  else
    run_stage low "$LOW_PRETRAIN_SCRIPT" "$low_run_name" "$low_log_dir" "$low_model_dir" "$gpu_id" "${low_args[@]}"
    low_exit=$?
    if [[ "$low_exit" == "0" ]]; then
      export PRETRAINED_LOW_DIR="$low_model_dir"
      run_stage high "$FINETUNE_SCRIPT" "$high_run_name" "$high_log_dir" "$high_model_dir" "$gpu_id" "${high_args[@]}"
      high_exit=$?
    else
      high_exit=99
    fi
    if [[ "$low_exit" == "0" && "$high_exit" == "0" ]]; then
      exit_code=0
      status="ok"
    else
      exit_code=1
      status="fail"
    fi
  fi

  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$SUITE_ID" "$COMPUTER_ID" "$SHARD_INDEX" "$NUM_SHARDS" "$work_index" \
    "$method" "$seed" "$high_run_name" "$status" "$exit_code" "$elapsed" \
    "$low_model_dir" "$high_model_dir" "$start_time" "$end_time" "$command_text" >> "$status_file"
  echo "DONE pipeline seed=$seed status=$status elapsed=${elapsed}s low_exit=$low_exit high_exit=$high_exit"
  return "$exit_code"
}

work_index=0
launched=0
for seed in "${SEED_LIST[@]}"; do
  if (( work_index % NUM_SHARDS == SHARD_INDEX )); then
    wait_for_slot
    gpu_id="${GPU_LIST[$((launched % ${#GPU_LIST[@]}))]}"
    run_one_pipeline "$work_index" "$seed" "$gpu_id" &
    launched=$((launched + 1))
  fi
  work_index=$((work_index + 1))
done

while (( $(active_job_count) > 0 )); do
  wait -n || true
done

if [[ -f "$EXPERIMENT_LOG_CSV" ]]; then
  cp "$EXPERIMENT_LOG_CSV" "$SUITE_DIR/$(basename "$EXPERIMENT_LOG_CSV")"
fi

scripts/UAVEnergyDeliveryLevel/aggregate_comparison_results.sh --suite-dir "$SUITE_DIR"
echo "Two-stage H-MAPPO suite complete: $SUITE_DIR"
