#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
SUITE_ID="${SUITE_ID:-uedl_suite_$(date +%m%d_%H%M%S)}"
COMPUTER_ID="${COMPUTER_ID:-$(hostname | tr -c 'A-Za-z0-9_.-' '_')}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
GPU_IDS="${GPU_IDS:-${GPU_ID:-0}}"
CUDA="${CUDA:-True}"
N_STEPS="${N_STEPS:-600000}"
EVALUATE_CYCLE="${EVALUATE_CYCLE:-5000}"
EVALUATE_EPOCH="${EVALUATE_EPOCH:-20}"
SEEDS="${SEEDS:-123}"
METHODS="${METHODS:-all}"
HEURISTIC_EPISODES="${HEURISTIC_EPISODES:-20}"
UAV_N_AGENTS="${UAV_N_AGENTS:-4}"
UAV_TOTAL_ORDERS="${UAV_TOTAL_ORDERS:-24}"
UAV_MAX_ACTIVE_ORDERS="${UAV_MAX_ACTIVE_ORDERS:-8}"
EPISODE_LIMIT="${EPISODE_LIMIT:-600}"
UAV_ENERGY_DECAY="${UAV_ENERGY_DECAY:-0.5}"
UAV_CHARGING_RATE="${UAV_CHARGING_RATE:-4.0}"
DRY_RUN="${DRY_RUN:-0}"
EXPERIMENT_DEVICE="${EXPERIMENT_DEVICE:-${MARL_EXPERIMENT_DEVICE:-$COMPUTER_ID}}"

usage() {
  cat <<'EOF'
Usage:
  scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh [options] [-- extra training args]

Runs UAVEnergyDeliveryLevel comparison methods with up to 4 concurrent jobs by
default. Use NUM_SHARDS/SHARD_INDEX or the flags below to split work across
different computers.

Common options:
  --suite-id ID          Shared experiment id. Use the same id on all computers.
  --computer-id ID       Label for this machine. Defaults to hostname.
  --num-shards N         Total number of computers/shards.
  --shard-index I        This computer's zero-based shard index.
  -j, --max-parallel N   Concurrent processes on this computer. Defaults to 4.
  --gpu-ids LIST         Comma/space-separated GPU ids, e.g. 0 or 0,1.
  --methods LIST         Comma/space-separated methods, or all.
  --seeds LIST           Comma/space-separated seeds. Defaults to 123.
  --n-steps N            Training steps for RL methods. Defaults to 600000.
  --evaluate-cycle N     Evaluation interval. Defaults to 5000.
  --evaluate-epoch N     Evaluation episodes. Defaults to 20.
  --cuda True|False      Defaults to True.
  --dry-run              Print commands and write status without training.

Environment equivalents:
  SUITE_ID, COMPUTER_ID, NUM_SHARDS, SHARD_INDEX, MAX_PARALLEL, GPU_IDS,
  METHODS, SEEDS, N_STEPS, CUDA.

Examples:
  # Single computer, 4 concurrent jobs
  scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh

  # Three computers. Run one command on each computer:
  SUITE_ID=paper_cmp NUM_SHARDS=3 SHARD_INDEX=0 scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh
  SUITE_ID=paper_cmp NUM_SHARDS=3 SHARD_INDEX=1 scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh
  SUITE_ID=paper_cmp NUM_SHARDS=3 SHARD_INDEX=2 scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh

  # Two seeds, two GPUs, still 4 concurrent jobs
  SEEDS=123,456 GPU_IDS=0,1 scripts/UAVEnergyDeliveryLevel/run_comparison_suite_4proc.sh
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
    --methods)
      METHODS="$2"
      shift 2
      ;;
    --methods=*)
      METHODS="${1#*=}"
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
    --n-steps)
      N_STEPS="$2"
      shift 2
      ;;
    --n-steps=*)
      N_STEPS="${1#*=}"
      shift
      ;;
    --evaluate-cycle)
      EVALUATE_CYCLE="$2"
      shift 2
      ;;
    --evaluate-cycle=*)
      EVALUATE_CYCLE="${1#*=}"
      shift
      ;;
    --evaluate-epoch)
      EVALUATE_EPOCH="$2"
      shift 2
      ;;
    --evaluate-epoch=*)
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

require_positive_int MAX_PARALLEL "$MAX_PARALLEL"
require_positive_int NUM_SHARDS "$NUM_SHARDS"
require_nonnegative_int SHARD_INDEX "$SHARD_INDEX"
if (( SHARD_INDEX >= NUM_SHARDS )); then
  echo "SHARD_INDEX must be smaller than NUM_SHARDS." >&2
  exit 2
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable not found or not executable: $PYTHON_BIN" >&2
  exit 2
fi

normalize_list() {
  printf "%s" "$1" | tr ',;' '  ' | xargs -n1
}

ALL_METHODS=(
  greedy_threshold_charge
  energy_aware_greedy
  auction_threshold_charge
  ippo
  mappo
  mappo_lagrangian
  mappo_safety_shield
  maddpg
  matd3
  macpo
  hsd
  hmappo_basic
  hmappo_wo_energy_aware_design
  ours_wo_hierarchy
  ours_wo_energy_constraint
  ours_wo_charging_resource_modeling
  ours_wo_safety_layer
  ours_wo_auction_module
  ours_wo_high_level_temporal_abstraction
  ours_full
)

script_for_method() {
  case "$1" in
    greedy_threshold_charge) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_greedy_threshold_charge.sh" ;;
    energy_aware_greedy) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_energy_aware_greedy.sh" ;;
    auction_threshold_charge) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_auction_threshold_charge.sh" ;;
    ippo) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ippo.sh" ;;
    mappo) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_mappo.sh" ;;
    mappo_lagrangian) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_mappo_lagrangian.sh" ;;
    mappo_safety_shield) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_mappo_safety_shield.sh" ;;
    maddpg) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_maddpg.sh" ;;
    matd3) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_matd3.sh" ;;
    macpo) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_macpo.sh" ;;
    hsd) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_hsd.sh" ;;
    hmappo_basic) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_hmappo_basic.sh" ;;
    hmappo_wo_energy_aware_design) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_hmappo_wo_energy_aware_design.sh" ;;
    ours_wo_hierarchy) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_hierarchy.sh" ;;
    ours_wo_energy_constraint) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_energy_constraint.sh" ;;
    ours_wo_charging_resource_modeling) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_charging_resource_modeling.sh" ;;
    ours_wo_safety_layer) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_safety_layer.sh" ;;
    ours_wo_auction_module) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_auction_module.sh" ;;
    ours_wo_high_level_temporal_abstraction) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_wo_high_level_temporal_abstraction.sh" ;;
    ours_full) echo "scripts/UAVEnergyDeliveryLevel/comparisons/train_ours_full.sh" ;;
    *) return 1 ;;
  esac
}

is_heuristic_method() {
  case "$1" in
    greedy_threshold_charge|energy_aware_greedy|auction_threshold_charge) return 0 ;;
    *) return 1 ;;
  esac
}

if [[ "$METHODS" == "all" ]]; then
  SELECTED_METHODS=("${ALL_METHODS[@]}")
else
  mapfile -t SELECTED_METHODS < <(normalize_list "$METHODS")
fi
mapfile -t SEED_LIST < <(normalize_list "$SEEDS")
mapfile -t GPU_LIST < <(normalize_list "$GPU_IDS")
if (( ${#GPU_LIST[@]} == 0 )); then
  GPU_LIST=(0)
fi

SUITE_DIR="logs/uav_energy_delivery_comparison_suites/$SUITE_ID"
RUNS_DIR="$SUITE_DIR/runs"
STATUS_DIR="$SUITE_DIR/status"
mkdir -p "$RUNS_DIR" "$STATUS_DIR"

MANIFEST="$SUITE_DIR/manifest.tsv"
printf "suite_id\tcomputer_id\tshard_index\tnum_shards\tmax_parallel\tmethods\tseeds\tn_steps\tevaluate_cycle\tevaluate_epoch\tcuda\tgpu_ids\tstarted_at\n" > "$MANIFEST"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
  "$SUITE_ID" "$COMPUTER_ID" "$SHARD_INDEX" "$NUM_SHARDS" "$MAX_PARALLEL" \
  "$METHODS" "$SEEDS" "$N_STEPS" "$EVALUATE_CYCLE" "$EVALUATE_EPOCH" \
  "$CUDA" "$GPU_IDS" "$(date '+%Y-%m-%d %H:%M:%S')" >> "$MANIFEST"

echo "SUITE_ID=$SUITE_ID"
echo "SUITE_DIR=$SUITE_DIR"
echo "COMPUTER_ID=$COMPUTER_ID"
echo "SHARD=$SHARD_INDEX/$NUM_SHARDS"
echo "MAX_PARALLEL=$MAX_PARALLEL"
echo "METHODS=${SELECTED_METHODS[*]}"
echo "SEEDS=${SEED_LIST[*]}"

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

run_one() {
  local work_index="$1"
  local method="$2"
  local seed="$3"
  local gpu_id="$4"
  local script="$5"
  local run_name="${SUITE_ID}_${COMPUTER_ID}_${method}_seed${seed}"
  local log_dir="$RUNS_DIR/$run_name"
  local model_dir="model_runs/comparisons/$run_name"
  local status_file="$STATUS_DIR/status_${COMPUTER_ID}_${work_index}_${method}_seed${seed}.tsv"
  local start_time end_time start_epoch end_epoch elapsed exit_code status command_text

  mkdir -p "$log_dir" "$model_dir"
  printf "suite_id\tcomputer_id\tshard_index\tnum_shards\twork_index\tmethod\tseed\trun_name\tstatus\texit_code\telapsed_sec\tscript\tlog_dir\tmodel_dir\tstart_time\tend_time\tcommand\n" > "$status_file"

  start_time="$(date '+%Y-%m-%d %H:%M:%S')"
  start_epoch="$(date +%s)"
  echo "START method=$method seed=$seed gpu=$gpu_id run=$run_name"

  if is_heuristic_method "$method"; then
    command_text="RUN_NAME=$run_name LOG_DIR=$log_dir SEED=$seed EPISODES=$HEURISTIC_EPISODES EPISODE_LIMIT=$EPISODE_LIMIT UAV_N_AGENTS=$UAV_N_AGENTS UAV_TOTAL_ORDERS=$UAV_TOTAL_ORDERS UAV_MAX_ACTIVE_ORDERS=$UAV_MAX_ACTIVE_ORDERS UAV_ENERGY_DECAY=$UAV_ENERGY_DECAY UAV_CHARGING_RATE=$UAV_CHARGING_RATE $script"
    if [[ "$DRY_RUN" == "1" ]]; then
      (
        export DRY_RUN=1 RUN_NAME="$run_name" LOG_DIR="$log_dir"
        export SEED="$seed" EPISODES="$HEURISTIC_EPISODES"
        export EPISODE_LIMIT="$EPISODE_LIMIT" UAV_N_AGENTS="$UAV_N_AGENTS"
        export UAV_TOTAL_ORDERS="$UAV_TOTAL_ORDERS" UAV_MAX_ACTIVE_ORDERS="$UAV_MAX_ACTIVE_ORDERS"
        export UAV_ENERGY_DECAY="$UAV_ENERGY_DECAY" UAV_CHARGING_RATE="$UAV_CHARGING_RATE"
        "$script"
      )
      exit_code=0
      status="dry_run"
    else
      (
        export RUN_NAME="$run_name" LOG_DIR="$log_dir"
        export SEED="$seed" EPISODES="$HEURISTIC_EPISODES"
        export EPISODE_LIMIT="$EPISODE_LIMIT" UAV_N_AGENTS="$UAV_N_AGENTS"
        export UAV_TOTAL_ORDERS="$UAV_TOTAL_ORDERS" UAV_MAX_ACTIVE_ORDERS="$UAV_MAX_ACTIVE_ORDERS"
        export UAV_ENERGY_DECAY="$UAV_ENERGY_DECAY" UAV_CHARGING_RATE="$UAV_CHARGING_RATE"
        "$script"
      )
      exit_code=$?
      status=$([[ "$exit_code" == "0" ]] && echo ok || echo fail)
    fi
  else
    local eval_seed=$((seed + 100000))
    local rl_args=(
      --seed "$seed"
      --eval_seed "$eval_seed"
      --n_steps "$N_STEPS"
      --time_steps "$N_STEPS"
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
      "${EXTRA_ARGS[@]}"
    )
    command_text="RUN_NAME=$run_name LOG_DIR=$log_dir MODEL_DIR=$model_dir EXPERIMENT_DEVICE=$EXPERIMENT_DEVICE $script ${rl_args[*]}"
    if [[ "$DRY_RUN" == "1" ]]; then
      (
        export DRY_RUN=1 RUN_NAME="$run_name" LOG_DIR="$log_dir" MODEL_DIR="$model_dir"
        export EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE" MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
        "$script" "${rl_args[@]}"
      )
      exit_code=0
      status="dry_run"
    else
      (
        export RUN_NAME="$run_name" LOG_DIR="$log_dir" MODEL_DIR="$model_dir"
        export EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE" MARL_EXPERIMENT_DEVICE="$EXPERIMENT_DEVICE"
        "$script" "${rl_args[@]}"
      )
      exit_code=$?
      status=$([[ "$exit_code" == "0" ]] && echo ok || echo fail)
    fi
  fi

  end_time="$(date '+%Y-%m-%d %H:%M:%S')"
  end_epoch="$(date +%s)"
  elapsed=$((end_epoch - start_epoch))
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$SUITE_ID" "$COMPUTER_ID" "$SHARD_INDEX" "$NUM_SHARDS" "$work_index" \
    "$method" "$seed" "$run_name" "$status" "$exit_code" "$elapsed" "$script" \
    "$log_dir" "$model_dir" "$start_time" "$end_time" "$command_text" >> "$status_file"
  echo "DONE method=$method seed=$seed status=$status elapsed=${elapsed}s"
  return "$exit_code"
}

work_index=0
launched=0
for seed in "${SEED_LIST[@]}"; do
  for method in "${SELECTED_METHODS[@]}"; do
    script="$(script_for_method "$method")" || {
      echo "Unknown method: $method" >&2
      exit 2
    }
    if [[ ! -x "$script" ]]; then
      echo "Script not found or not executable for $method: $script" >&2
      exit 2
    fi
    if (( work_index % NUM_SHARDS == SHARD_INDEX )); then
      wait_for_slot
      gpu_id="${GPU_LIST[$((launched % ${#GPU_LIST[@]}))]}"
      run_one "$work_index" "$method" "$seed" "$gpu_id" "$script" &
      launched=$((launched + 1))
    fi
    work_index=$((work_index + 1))
  done
done

while (( $(active_job_count) > 0 )); do
  wait -n || true
done

if [[ -f train_logs/uav_delivery_experiments.csv ]]; then
  cp train_logs/uav_delivery_experiments.csv "$SUITE_DIR/uav_delivery_experiments_${COMPUTER_ID}.csv"
fi

scripts/UAVEnergyDeliveryLevel/aggregate_comparison_results.sh --suite-dir "$SUITE_DIR"
echo "Suite complete: $SUITE_DIR"
