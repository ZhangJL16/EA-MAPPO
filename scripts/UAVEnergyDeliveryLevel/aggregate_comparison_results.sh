#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SUITE_ID="${SUITE_ID:-}"
SUITE_DIR="${SUITE_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

usage() {
  cat <<'EOF'
Usage:
  scripts/UAVEnergyDeliveryLevel/aggregate_comparison_results.sh [options]

Options:
  --suite-id ID       Suite id under logs/uav_energy_delivery_comparison_suites/.
  --suite-dir DIR     Explicit suite directory. Overrides --suite-id.
  --output-dir DIR    Directory for aggregate outputs. Defaults to <suite-dir>/aggregate.

This script aggregates:
  - status TSV files produced by run_comparison_suite_4proc.sh
  - heuristic_eval.csv files under the suite directory
  - train_logs/uav_delivery_experiments.csv snapshots copied into the suite

For multi-computer aggregation, copy each computer's suite directory into the
same suite directory on one machine, then run this script with --suite-dir.
EOF
}

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
    --suite-dir)
      SUITE_DIR="$2"
      shift 2
      ;;
    --suite-dir=*)
      SUITE_DIR="${1#*=}"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --output-dir=*)
      OUTPUT_DIR="${1#*=}"
      shift
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$SUITE_DIR" ]]; then
  if [[ -n "$SUITE_ID" ]]; then
    SUITE_DIR="logs/uav_energy_delivery_comparison_suites/$SUITE_ID"
  else
    SUITE_DIR="$(find logs/uav_energy_delivery_comparison_suites -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1 || true)"
  fi
fi

if [[ -z "$SUITE_DIR" || ! -d "$SUITE_DIR" ]]; then
  echo "Suite directory not found. Pass --suite-id or --suite-dir." >&2
  exit 2
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$SUITE_DIR/aggregate"
fi
mkdir -p "$OUTPUT_DIR"

SUITE_DIR="$SUITE_DIR" OUTPUT_DIR="$OUTPUT_DIR" .venv/bin/python - <<'PY'
import csv
import math
import os
from collections import defaultdict
from pathlib import Path

suite_dir = Path(os.environ["SUITE_DIR"]).resolve()
output_dir = Path(os.environ["OUTPUT_DIR"]).resolve()
output_dir.mkdir(parents=True, exist_ok=True)
repo_root = Path.cwd()

def read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

def to_float(value):
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

def stats(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    if not values:
        return "", ""
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)

status_rows = []
for path in sorted(suite_dir.rglob("status_*.tsv")):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            row["_status_file"] = str(path)
            status_rows.append(row)

combined_status = output_dir / "status.csv"
status_fields = [
    "suite_id", "computer_id", "shard_index", "num_shards", "work_index",
    "method", "seed", "run_name", "status", "exit_code", "elapsed_sec",
    "script", "log_dir", "model_dir", "start_time", "end_time", "command",
    "_status_file",
]
write_csv(combined_status, status_rows, status_fields)

run_by_name = {row.get("run_name", ""): row for row in status_rows if row.get("run_name")}

experiment_csvs = []
snapshot_names = {"uav_delivery_experiments.csv"}
experiment_csvs.extend(sorted(suite_dir.rglob("uav_delivery_experiments*.csv")))
global_csv = repo_root / "train_logs" / "uav_delivery_experiments.csv"
if global_csv.exists():
    experiment_csvs.append(global_csv)

experiment_rows = []
seen = set()
for path in experiment_csvs:
    for row in read_csv(path):
        key = (
            row.get("run_command", ""),
            row.get("timestamp", ""),
            row.get("rl_algorithm", ""),
            row.get("final_timestep", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        run_name = ""
        command = row.get("run_command", "")
        for candidate in run_by_name:
            if candidate and candidate in command:
                run_name = candidate
                break
        if run_name:
            row["_run_name"] = run_name
            row["_source_csv"] = str(path)
            experiment_rows.append(row)

heuristic_rows = []
for status in status_rows:
    method = status.get("method", "")
    run_name = status.get("run_name", "")
    log_dir = status.get("log_dir", "")
    candidates = []
    if log_dir:
        candidates.append(Path(log_dir) / "heuristic_eval.csv")
        candidates.append(suite_dir / log_dir / "heuristic_eval.csv")
    if run_name:
        candidates.extend(suite_dir.rglob(f"{run_name}/heuristic_eval.csv"))
    found = None
    for candidate in candidates:
        if candidate.exists():
            found = candidate
            break
    if not found:
        continue
    rows = read_csv(found)
    numeric_keys = [
        "episode_reward", "episode_steps", "orders_completed", "total_orders",
        "collision_count", "obstacle_collision_count", "agent_collision_count",
        "charging_agents", "mean_energy", "depleted_agents",
    ]
    agg = {
        "_run_name": run_name,
        "_source_csv": str(found),
        "device": status.get("computer_id", ""),
        "rl_algorithm": method,
        "map": "UAVEnergyDeliveryLevel",
        "seed": status.get("seed", ""),
        "run_script": status.get("script", ""),
        "run_command": status.get("command", ""),
    }
    for key in numeric_keys:
        mean, _ = stats([to_float(row.get(key)) for row in rows])
        agg[key] = mean
    total_orders = to_float(agg.get("total_orders"))
    orders_completed = to_float(agg.get("orders_completed"))
    agg["delivery_success_rate"] = (
        orders_completed / total_orders
        if total_orders and orders_completed is not None
        else ""
    )
    agg["eval_episode_reward"] = agg.get("episode_reward", "")
    heuristic_rows.append(agg)

runs = []
metric_map = [
    ("eval_episode_reward", "eval_episode_reward"),
    ("delivery_success_rate", "delivery_success_rate"),
    ("orders_completed", "orders_completed"),
    ("total_orders", "total_orders"),
    ("collision_count", "collision_count"),
    ("obstacle_collision_count", "obstacle_collision_count"),
    ("agent_collision_count", "agent_collision_count"),
    ("eval_episode_steps", "eval_episode_steps"),
]

for row in experiment_rows:
    status = run_by_name.get(row.get("_run_name", ""), {})
    out = {
        "method": status.get("method", row.get("rl_algorithm", "")),
        "seed": status.get("seed", row.get("seed", "")),
        "run_name": row.get("_run_name", ""),
        "status": status.get("status", ""),
        "exit_code": status.get("exit_code", ""),
        "computer_id": status.get("computer_id", row.get("device", "")),
        "rl_algorithm": row.get("rl_algorithm", ""),
        "map": row.get("map", ""),
        "n_steps": row.get("n_steps", ""),
        "final_timestep": row.get("final_timestep", ""),
        "train_step": row.get("train_step", ""),
        "run_script": row.get("run_script", status.get("script", "")),
        "source_csv": row.get("_source_csv", ""),
    }
    for src, dst in metric_map:
        out[dst] = row.get(src, "")
    runs.append(out)

for row in heuristic_rows:
    status = run_by_name.get(row.get("_run_name", ""), {})
    out = {
        "method": status.get("method", row.get("rl_algorithm", "")),
        "seed": status.get("seed", row.get("seed", "")),
        "run_name": row.get("_run_name", ""),
        "status": status.get("status", ""),
        "exit_code": status.get("exit_code", ""),
        "computer_id": status.get("computer_id", row.get("device", "")),
        "rl_algorithm": row.get("rl_algorithm", ""),
        "map": row.get("map", ""),
        "n_steps": "",
        "final_timestep": "",
        "train_step": "",
        "run_script": row.get("run_script", status.get("script", "")),
        "source_csv": row.get("_source_csv", ""),
        "eval_episode_reward": row.get("eval_episode_reward", ""),
        "delivery_success_rate": row.get("delivery_success_rate", ""),
        "orders_completed": row.get("orders_completed", ""),
        "total_orders": row.get("total_orders", ""),
        "collision_count": row.get("collision_count", ""),
        "obstacle_collision_count": row.get("obstacle_collision_count", ""),
        "agent_collision_count": row.get("agent_collision_count", ""),
        "eval_episode_steps": row.get("episode_steps", ""),
    }
    runs.append(out)

run_fields = [
    "method", "seed", "run_name", "status", "exit_code", "computer_id",
    "rl_algorithm", "map", "n_steps", "final_timestep", "train_step",
    "eval_episode_reward", "delivery_success_rate", "orders_completed",
    "total_orders", "collision_count", "obstacle_collision_count",
    "agent_collision_count", "eval_episode_steps", "run_script", "source_csv",
]
write_csv(output_dir / "runs_metrics.csv", runs, run_fields)

grouped = defaultdict(list)
for row in runs:
    grouped[row.get("method", "")].append(row)

summary_rows = []
summary_metrics = [
    "eval_episode_reward", "delivery_success_rate", "orders_completed",
    "collision_count", "obstacle_collision_count", "agent_collision_count",
    "eval_episode_steps",
]
for method, rows in sorted(grouped.items()):
    out = {
        "method": method,
        "run_count": len(rows),
        "ok_count": sum(1 for row in rows if row.get("status") == "ok"),
        "fail_count": sum(1 for row in rows if row.get("status") == "fail"),
    }
    seeds = sorted({row.get("seed", "") for row in rows if row.get("seed", "") != ""})
    out["seeds"] = " ".join(seeds)
    for metric in summary_metrics:
        mean, std = stats([to_float(row.get(metric)) for row in rows])
        out[f"{metric}_mean"] = mean
        out[f"{metric}_std"] = std
    summary_rows.append(out)

summary_fields = [
    "method", "run_count", "ok_count", "fail_count", "seeds",
]
for metric in summary_metrics:
    summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
write_csv(output_dir / "method_summary.csv", summary_rows, summary_fields)

failed = [
    row for row in status_rows
    if row.get("status") not in {"ok", "dry_run"}
]
write_csv(output_dir / "failed_runs.csv", failed, status_fields)

print(f"Suite: {suite_dir}")
print(f"Status rows: {len(status_rows)}")
print(f"Metric rows: {len(runs)}")
print(f"Method rows: {len(summary_rows)}")
print(f"Wrote: {output_dir / 'status.csv'}")
print(f"Wrote: {output_dir / 'runs_metrics.csv'}")
print(f"Wrote: {output_dir / 'method_summary.csv'}")
print(f"Wrote: {output_dir / 'failed_runs.csv'}")
PY
