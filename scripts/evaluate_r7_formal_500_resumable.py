from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.r7_cppo_pid import R7ActorCritic, R7ModelConfig
from scripts.train_r7_cppo_pid_navigation import (
    DISTANCE_BUCKETS,
    PROTOCOL,
    atomic_json,
    evaluate,
    file_sha256,
    load_evaluation_tasks,
    utc_now,
)
from scripts.train_uav_energy_delivery_sac import (
    navigation_energy_gate_passed,
    navigation_safety_gate_passed,
)


FORMAL_PROTOCOL = "R7_CPPO_PID_NAVIGATION_FORMAL_500_V1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable five-bucket, 500-task formal R7 evaluation"
    )
    parser.add_argument("--training-completed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--evaluation-num-envs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=20)
    args = parser.parse_args()
    if min(args.evaluation_num_envs, args.batch_size) <= 0:
        parser.error("evaluation worker and batch counts must be positive")
    if not Path(args.training_completed).is_file():
        parser.error("training COMPLETED.json does not exist")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    return args


def aggregate(
    records: list[dict[str, object]],
    *,
    task_source: Path,
    wall_seconds: float,
) -> dict[str, object]:
    records.sort(key=lambda row: int(row["evaluation_index"]))
    bucket_success: dict[str, float] = {}
    for bucket in DISTANCE_BUCKETS:
        rows = [row for row in records if row["distance_bucket"] == bucket]
        if len(rows) != 100:
            raise ValueError(f"formal bucket {bucket} has {len(rows)} rather than 100 tasks")
        bucket_success[bucket] = float(
            np.mean([bool(row["success"]) for row in rows])
        )
    total_steps = sum(int(row["policy_steps"]) for row in records)
    summary: dict[str, object] = {
        "protocol": PROTOCOL,
        "formal_protocol": FORMAL_PROTOCOL,
        "development_subset": False,
        "evaluation_distance_buckets": list(DISTANCE_BUCKETS),
        "task_source": str(task_source),
        "task_source_sha256": file_sha256(task_source),
        "num_tasks": len(records),
        "tasks_per_distance_bucket": 100,
        "overall_success_rate": float(
            np.mean([bool(row["success"]) for row in records])
        ),
        "distance_bucket_success": bucket_success,
        "mean_path_ratio": float(
            np.mean([float(row["path_ratio"]) for row in records])
        ),
        "mean_policy_steps": float(
            np.mean([int(row["policy_steps"]) for row in records])
        ),
        "hocbf_intervention_step_rate": float(
            sum(
                float(row["hocbf_intervention_rate"])
                * int(row["policy_steps"])
                for row in records
            )
            / max(total_steps, 1)
        ),
        "obstacle_collision_steps": int(
            sum(int(row["obstacle_collision_steps"]) for row in records)
        ),
        "boundary_contact_steps": int(
            sum(int(row["boundary_contact_steps"]) for row in records)
        ),
        "boundary_contact_step_rate": float(
            sum(int(row["boundary_contact_steps"]) for row in records)
            / max(total_steps, 1)
        ),
        "evaluation_env_transitions": total_steps,
        "wall_clock_seconds": wall_seconds,
        "records": records,
    }
    summary["formal_navigation_energy_gate_passed"] = navigation_energy_gate_passed(
        summary
    )
    summary["formal_navigation_safety_gate_passed"] = navigation_safety_gate_passed(
        summary
    )
    return summary


def main() -> int:
    cli = parse_args()
    training_completed_path = Path(cli.training_completed).resolve()
    training_completed = json.loads(
        training_completed_path.read_text(encoding="utf-8")
    )
    if training_completed.get("status") != "COMPLETED":
        raise ValueError("training artifact is not complete")
    if int(training_completed.get("actual_training_transitions", -1)) != 500_000:
        raise ValueError("formal evaluation requires the exact 500k checkpoint")
    checkpoint_path = Path(training_completed["final_checkpoint"]).resolve()
    checkpoint_sha256 = file_sha256(checkpoint_path)
    task_source = Path(training_completed["args"]["evaluation_tasks"]).resolve()
    task_source_sha256 = file_sha256(task_source)
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "COMPLETED.json"
    if completed_path.is_file():
        prior = json.loads(completed_path.read_text(encoding="utf-8"))
        if prior.get("checkpoint_sha256") != checkpoint_sha256:
            raise ValueError("completed formal evaluation belongs to another checkpoint")
        print(json.dumps({"status": "ALREADY_COMPLETED", "output": str(output)}))
        return 0
    (output / "FAILED.json").unlink(missing_ok=True)
    (output / "INTERRUPTED.json").unlink(missing_ok=True)
    progress_path = output / "PROGRESS.json"
    records: list[dict[str, object]] = []
    elapsed_before = 0.0
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress["checkpoint_sha256"] != checkpoint_sha256:
            raise ValueError("formal progress checkpoint identity mismatch")
        if progress["task_source_sha256"] != task_source_sha256:
            raise ValueError("formal progress task-source identity mismatch")
        records = list(progress["records"])
        elapsed_before = float(progress["evaluation_wall_clock_seconds"])
    source_ids = [int(row["source_task_index"]) for row in records]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("formal progress contains duplicate source tasks")
    all_tasks, _base_seed = load_evaluation_tasks(
        task_source, 500, tuple(DISTANCE_BUCKETS)
    )
    evaluation_index_by_source = {
        int(task["source_task_index"]): index for index, task in enumerate(all_tasks)
    }
    unknown = set(source_ids) - set(evaluation_index_by_source)
    if unknown:
        raise ValueError(f"formal progress contains unknown task indices: {sorted(unknown)}")
    manifest = {
        "status": "RUNNING",
        "protocol": FORMAL_PROTOCOL,
        "started_or_resumed_at": utc_now(),
        "training_completed": str(training_completed_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "task_source": str(task_source),
        "task_source_sha256": task_source_sha256,
        "target_tasks": 500,
        "tasks_per_distance_bucket": 100,
        "evaluation_num_envs": cli.evaluation_num_envs,
        "batch_size": cli.batch_size,
        "completed_tasks_at_resume": len(records),
        "wall_clock_limit": None,
    }
    atomic_json(output / "RUNNING.json", manifest)
    device = torch.device(cli.device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("R7 checkpoint protocol mismatch")
    model = R7ActorCritic(R7ModelConfig(**payload["model_config"])).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    evaluation_args = argparse.Namespace(**training_completed["args"])
    evaluation_args.device = cli.device
    evaluation_args.evaluation_tasks = str(task_source)
    evaluation_args.evaluation_num_envs = cli.evaluation_num_envs
    evaluation_args.evaluation_distance_buckets = list(DISTANCE_BUCKETS)
    completed_sources = set(source_ids)
    pending = [
        task
        for task in all_tasks
        if int(task["source_task_index"]) not in completed_sources
    ]
    started = time.perf_counter()
    try:
        for offset in range(0, len(pending), cli.batch_size):
            batch = pending[offset : offset + cli.batch_size]
            selected = [int(task["source_task_index"]) for task in batch]
            evaluation_args.eval_tasks = len(selected)
            evaluation_args.evaluation_source_indices = selected
            result = evaluate(model=model, args=evaluation_args, device=device)
            for record in result["records"]:
                source_index = int(record["source_task_index"])
                record["evaluation_index"] = evaluation_index_by_source[source_index]
                records.append(record)
            elapsed = elapsed_before + time.perf_counter() - started
            atomic_json(
                progress_path,
                {
                    **manifest,
                    "status": "IN_PROGRESS",
                    "updated_at": utc_now(),
                    "completed_tasks": len(records),
                    "evaluation_wall_clock_seconds": elapsed,
                    "records": records,
                },
            )
        if len(records) != 500:
            raise RuntimeError(f"formal evaluation ended with {len(records)} tasks")
        elapsed = elapsed_before + time.perf_counter() - started
        summary = aggregate(records, task_source=task_source, wall_seconds=elapsed)
        atomic_json(output / "formal_navigation_500.json", summary)
        completed = {
            **manifest,
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "completed_tasks": 500,
            "evaluation_wall_clock_seconds": elapsed,
            "formal_navigation": {
                key: value for key, value in summary.items() if key != "records"
            },
        }
        atomic_json(completed_path, completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except KeyboardInterrupt:
        atomic_json(
            output / "INTERRUPTED.json",
            {
                **manifest,
                "status": "INTERRUPTED",
                "interrupted_at": utc_now(),
                "completed_tasks": len(records),
                "resume_from": str(progress_path),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise
    except BaseException as error:
        atomic_json(
            output / "FAILED.json",
            {
                **manifest,
                "status": "FAILED",
                "failed_at": utc_now(),
                "completed_tasks": len(records),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "resume_from": str(progress_path),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
