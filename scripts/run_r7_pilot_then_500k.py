from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DISTANCE_BUCKETS = ("100-500", "500-1500", "1500-2500", "2500-4000", ">4000")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def promotion_decision(completed: dict[str, Any]) -> dict[str, Any]:
    navigation = completed["final_navigation"]
    training = completed["training_episode_summary"]
    per_bucket = navigation["distance_bucket_success"]
    checks = {
        "training_has_real_goal_success": int(training["successes"]) > 0,
        "overall_success_rate_at_least_0_80": (
            float(navigation["overall_success_rate"]) >= 0.80
        ),
        "every_distance_bucket_success_at_least_0_60": all(
            per_bucket.get(bucket) is not None
            and float(per_bucket[bucket]) >= 0.60
            for bucket in DISTANCE_BUCKETS
        ),
        "mean_path_ratio_at_most_1_50": (
            float(navigation["mean_path_ratio"]) <= 1.50
        ),
        "zero_obstacle_collision_steps": (
            int(navigation["obstacle_collision_steps"]) == 0
        ),
        "zero_boundary_contact_steps": (
            int(navigation["boundary_contact_steps"]) == 0
        ),
        "hocbf_intervention_rate_at_most_0_20": (
            float(navigation["hocbf_intervention_step_rate"]) <= 0.20
        ),
    }
    return {
        "promote_to_500k": all(checks.values()),
        "checks": checks,
        "observed": {
            "training_successes": int(training["successes"]),
            "overall_success_rate": float(navigation["overall_success_rate"]),
            "distance_bucket_success": per_bucket,
            "mean_path_ratio": float(navigation["mean_path_ratio"]),
            "obstacle_collision_steps": int(navigation["obstacle_collision_steps"]),
            "boundary_contact_steps": int(navigation["boundary_contact_steps"]),
            "hocbf_intervention_step_rate": float(
                navigation["hocbf_intervention_step_rate"]
            ),
        },
    }


def training_command(
    output: Path,
    *,
    transition_budget: int,
    resume_checkpoint: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/train_r7_cppo_pid_navigation.py"),
        "--output-dir",
        str(output),
        "--device",
        "cuda",
        "--seed",
        "7002",
        "--rollout-seed-base",
        "720000",
        "--transition-budget",
        str(transition_budget),
        "--num-envs",
        "8",
        "--rollout-steps",
        "250",
        "--checkpoint-freq",
        "50000",
        "--eval-tasks",
        "25",
        "--evaluation-num-envs",
        "6",
        "--evaluation-distance-buckets",
        *DISTANCE_BUCKETS,
    ]
    if resume_checkpoint is not None:
        command.extend(["--resume-checkpoint", str(resume_checkpoint)])
    return command


def run_command(
    command: list[str], log_path: Path, timeout_seconds: int | None
) -> int:
    executed = command
    if timeout_seconds is not None:
        executed = [
            "timeout",
            "--signal=INT",
            "--kill-after=60s",
            f"{timeout_seconds}s",
            *command,
        ]
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run(
            executed,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-hour R7 pilot and conditionally resume to 500k"
    )
    parser.add_argument("--output-root", required=True)
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--pilot-resume-checkpoint")
    resume_group.add_argument(
        "--promoted-resume-checkpoint",
        help="Skip the pilot gate and resume the promoted 500k stage directly",
    )
    parser.add_argument("--pilot-timeout-seconds", type=int, default=4200)
    parser.add_argument("--long-timeout-seconds", type=int, default=32400)
    parser.add_argument("--no-timeouts", action="store_true")
    parser.add_argument("--formal-evaluation-500", action="store_true")
    parser.add_argument("--formal-evaluation-num-envs", type=int, default=8)
    parser.add_argument("--formal-evaluation-batch-size", type=int, default=20)
    args = parser.parse_args()
    if min(
        args.pilot_timeout_seconds,
        args.long_timeout_seconds,
        args.formal_evaluation_num_envs,
        args.formal_evaluation_batch_size,
    ) <= 0:
        parser.error("timeouts must be positive")
    if args.pilot_resume_checkpoint and not Path(args.pilot_resume_checkpoint).is_file():
        parser.error("pilot resume checkpoint does not exist")
    if args.promoted_resume_checkpoint and not Path(
        args.promoted_resume_checkpoint
    ).is_file():
        parser.error("promoted resume checkpoint does not exist")
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    pilot_output = output_root / "pilot_196k"
    long_output = output_root / "promoted_500k"
    formal_output = output_root / "formal_evaluation_500"
    pilot_timeout = None if args.no_timeouts else args.pilot_timeout_seconds
    long_timeout = None if args.no_timeouts else args.long_timeout_seconds
    pilot_resume_checkpoint = (
        None
        if args.pilot_resume_checkpoint is None
        else Path(args.pilot_resume_checkpoint).resolve()
    )
    promoted_resume_checkpoint = (
        None
        if args.promoted_resume_checkpoint is None
        else Path(args.promoted_resume_checkpoint).resolve()
    )
    running = {
        "status": (
            "RUNNING_PILOT"
            if promoted_resume_checkpoint is None
            else "RUNNING_PROMOTED_500K"
        ),
        "started_at": utc_now(),
        "pilot_transition_budget": 196_000,
        "promoted_total_transition_budget": 500_000,
        "pilot_timeout_seconds": pilot_timeout,
        "long_timeout_seconds": long_timeout,
        "formal_evaluation_timeout_seconds": None,
        "formal_evaluation_500_enabled": args.formal_evaluation_500,
        "pilot_output": str(pilot_output),
        "long_output": str(long_output),
        "formal_output": str(formal_output),
        "pilot_resume_checkpoint": (
            None if pilot_resume_checkpoint is None else str(pilot_resume_checkpoint)
        ),
        "promoted_resume_checkpoint": (
            None
            if promoted_resume_checkpoint is None
            else str(promoted_resume_checkpoint)
        ),
        "promotion_policy": {
            "training_successes": ">0",
            "overall_success_rate": ">=0.80",
            "every_distance_bucket_success": ">=0.60",
            "mean_path_ratio": "<=1.50",
            "obstacle_collision_steps": "==0",
            "boundary_contact_steps": "==0",
            "hocbf_intervention_step_rate": "<=0.20",
        },
    }
    atomic_json(output_root / "PIPELINE_RUNNING.json", running)
    if promoted_resume_checkpoint is None:
        pilot_command = training_command(
            pilot_output,
            transition_budget=196_000,
            resume_checkpoint=pilot_resume_checkpoint,
        )
        atomic_json(
            output_root / "PILOT_COMMAND.json",
            {"command": pilot_command, "started_at": utc_now()},
        )
        pilot_return_code = run_command(
            pilot_command, output_root / "pilot.log", pilot_timeout
        )
        pilot_completed_path = pilot_output / "COMPLETED.json"
        if pilot_return_code != 0 or not pilot_completed_path.is_file():
            result = {
                **running,
                "status": "PILOT_FAILED_OR_TIMED_OUT",
                "finished_at": utc_now(),
                "pilot_return_code": pilot_return_code,
            }
            atomic_json(output_root / "PIPELINE_FAILED.json", result)
            (output_root / "PIPELINE_RUNNING.json").unlink(missing_ok=True)
            return 1
        pilot_completed = json.loads(pilot_completed_path.read_text(encoding="utf-8"))
        decision = promotion_decision(pilot_completed)
        decision["decided_at"] = utc_now()
        atomic_json(output_root / "PILOT_DECISION.json", decision)
        if not decision["promote_to_500k"]:
            result = {
                **running,
                "status": "PILOT_COMPLETED_NOT_PROMOTED",
                "finished_at": utc_now(),
                "decision": decision,
            }
            atomic_json(output_root / "PIPELINE_COMPLETED.json", result)
            (output_root / "PIPELINE_RUNNING.json").unlink(missing_ok=True)
            return 0
        checkpoint = pilot_output / "checkpoints/checkpoint_transition_000196000.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"pilot checkpoint missing: {checkpoint}")
    else:
        checkpoint = promoted_resume_checkpoint
        decision = {
            "promote_to_500k": True,
            "basis": "explicit_resume_from_saved_pilot_checkpoint",
            "checkpoint": str(checkpoint),
            "decided_at": utc_now(),
        }
        atomic_json(output_root / "PILOT_DECISION.json", decision)
    long_command = training_command(
        long_output, transition_budget=500_000, resume_checkpoint=checkpoint
    )
    atomic_json(
        output_root / "LONG_COMMAND.json",
        {"command": long_command, "started_at": utc_now()},
    )
    running["status"] = "RUNNING_PROMOTED_500K"
    atomic_json(output_root / "PIPELINE_RUNNING.json", running)
    long_return_code = run_command(
        long_command, output_root / "long.log", long_timeout
    )
    long_completed = long_output / "COMPLETED.json"
    status = (
        "PROMOTED_500K_COMPLETED"
        if long_return_code == 0 and long_completed.is_file()
        else "PROMOTED_500K_FAILED_OR_TIMED_OUT"
    )
    result: dict[str, Any] = {
        **running,
        "status": status,
        "finished_at": utc_now(),
        "decision": decision,
        "long_return_code": long_return_code,
    }
    if status != "PROMOTED_500K_COMPLETED":
        atomic_json(output_root / "PIPELINE_FAILED.json", result)
        (output_root / "PIPELINE_RUNNING.json").unlink(missing_ok=True)
        return 1
    if args.formal_evaluation_500:
        formal_command = [
            sys.executable,
            str(ROOT / "scripts/evaluate_r7_formal_500_resumable.py"),
            "--training-completed",
            str(long_completed),
            "--output-dir",
            str(formal_output),
            "--device",
            "cuda",
            "--evaluation-num-envs",
            str(args.formal_evaluation_num_envs),
            "--batch-size",
            str(args.formal_evaluation_batch_size),
        ]
        atomic_json(
            output_root / "FORMAL_EVALUATION_COMMAND.json",
            {"command": formal_command, "started_at": utc_now()},
        )
        running["status"] = "RUNNING_FORMAL_EVALUATION_500"
        atomic_json(output_root / "PIPELINE_RUNNING.json", running)
        formal_return_code = run_command(
            formal_command, output_root / "formal_evaluation.log", None
        )
        formal_completed = formal_output / "COMPLETED.json"
        result["formal_evaluation_return_code"] = formal_return_code
        result["status"] = (
            "PROMOTED_500K_AND_FORMAL_500_COMPLETED"
            if formal_return_code == 0 and formal_completed.is_file()
            else "FORMAL_500_FAILED_OR_INTERRUPTED"
        )
        result["finished_at"] = utc_now()
        if result["status"] != "PROMOTED_500K_AND_FORMAL_500_COMPLETED":
            atomic_json(output_root / "PIPELINE_FAILED.json", result)
            (output_root / "PIPELINE_RUNNING.json").unlink(missing_ok=True)
            return 1
    terminal = (
        output_root / "PIPELINE_COMPLETED.json"
        if result["status"] in {
            "PROMOTED_500K_COMPLETED",
            "PROMOTED_500K_AND_FORMAL_500_COMPLETED",
        }
        else output_root / "PIPELINE_FAILED.json"
    )
    atomic_json(terminal, result)
    (output_root / "PIPELINE_RUNNING.json").unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
