#!/usr/bin/env python3
"""Evaluate a fixed non-learning navigation platform against the formal Gate.

This is a new platform contract, not an R6 SAC repair.  It keeps the environment,
HOCBF, task generator, and navigation thresholds inherited from a frozen source
artifact while replacing only the navigation policy with the existing
deterministic go-to-goal controller.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_jseb_checkpoints import migrate_legacy_jseb_command
from scripts.run_jacobian_safety_energy_1m import parse_args as parse_jseb_args
from scripts.train_uav_energy_delivery_sac import (
    HeuristicGoalPolicy,
    evaluate_navigation_tasks,
    generate_stratified_navigation_tasks,
    navigation_energy_gate_passed,
    navigation_safety_gate_passed,
    save_navigation_tasks,
)


FORMAL_TASKS = 500
FORMAL_SEED = 170_002
PLATFORM_NAME = "deterministic_go_to_goal_v1"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reconstruct_platform_environment_args(
    artifact: Path,
    *,
    device: str,
    seed: int,
):
    """Rebuild the R5 environment while removing evaluation-irrelevant CLI flags."""
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    command = list(config["exact_command"])[1:]
    command, migration = migrate_legacy_jseb_command(command)
    flags_with_values = {
        "--navigation-repair-variant",
        "--training-vec-env",
        "--training-vec-start-method",
    }
    boolean_flags = {"--hocbf-sampled-data-robust"}
    filtered: list[str] = []
    removed: dict[str, object] = {}
    index = 0
    while index < len(command):
        flag = command[index]
        if flag in flags_with_values:
            if index + 1 >= len(command):
                raise ValueError(f"source command is missing a value for {flag}")
            removed[flag] = command[index + 1]
            index += 2
            continue
        if flag in boolean_flags:
            removed[flag] = True
            index += 1
            continue
        filtered.append(flag)
        index += 1
    if removed.get("--navigation-repair-variant") != "R5":
        raise ValueError("the fixed platform contract requires the frozen R5 source")
    if removed.get("--hocbf-sampled-data-robust") is not True:
        raise ValueError("the R5 source must declare sampled-data robust HOCBF")
    args = parse_jseb_args(filtered)
    args.device = device
    args.eval_task_seed = seed
    args.eval_navigation_tasks = FORMAL_TASKS
    args.hocbf_sampled_data_robust = True
    audit = {
        **dict(config),
        "command_migration": migration,
        "evaluation_irrelevant_source_flags_removed": removed,
        "sampled_data_robust_hocbf_preserved": True,
    }
    return args, audit


def _gate_summary(result: dict[str, object]) -> dict[str, object]:
    energy_passed = navigation_energy_gate_passed(result)
    safety_passed = navigation_safety_gate_passed(result)
    failures: list[str] = []
    if not energy_passed:
        failures.append("navigation_energy_readiness_gate_failed")
    if not safety_passed:
        failures.append("navigation_collision_boundary_gate_failed")
    bucket_values = [
        float(value)
        for value in result["distance_bucket_success"].values()
        if value is not None
    ]
    return {
        "platform_contract": PLATFORM_NAME,
        "num_tasks": int(result["num_tasks"]),
        "overall_success_rate": float(result["overall_success_rate"]),
        "minimum_distance_bucket_success": min(bucket_values),
        "distance_bucket_success": result["distance_bucket_success"],
        "mean_path_ratio": float(result["mean_path_ratio"]),
        "obstacle_collision_steps": int(result["obstacle_collision_steps"]),
        "obstacle_collision_episode_rate": float(
            result["obstacle_collision_episode_rate"]
        ),
        "boundary_contact_step_rate": float(
            result["boundary_contact_step_rate"]
        ),
        "hocbf_intervention_step_rate": float(
            result["hocbf_intervention_step_rate"]
        ),
        "hocbf_emergency_brake_step_rate": float(
            result["hocbf_emergency_brake_step_rate"]
        ),
        "nominal_safe_action_rate": float(result["nominal_safe_action_rate"]),
        "projection_valid_step_rate": float(
            result["projection_valid_step_rate"]
        ),
        "evaluation_env_transitions": int(result["evaluation_env_transitions"]),
        "navigation_energy_gate_passed": bool(energy_passed),
        "navigation_safety_gate_passed": bool(safety_passed),
        "navigation_gate_passed": bool(energy_passed and safety_passed),
        "navigation_gate_failures": failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--selection-seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--num-tasks", type=int, default=FORMAL_TASKS)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument("--progress-interval-tasks", type=int, default=25)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.num_tasks = 10
        args.evaluation_num_envs = min(args.evaluation_num_envs, 2)
        args.progress_interval_tasks = 2
    elif args.num_tasks != FORMAL_TASKS:
        parser.error("the formal fixed-platform Gate requires exactly 500 tasks")
    if args.evaluation_num_envs <= 0 or args.progress_interval_tasks <= 0:
        parser.error("evaluation worker and progress counts must be positive")
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source_artifact = args.source_artifact.resolve()
    output_dir = args.output_dir.resolve()
    plan = {
        "platform_contract": PLATFORM_NAME,
        "source_artifact": str(source_artifact),
        "output_dir": str(output_dir),
        "selection_seed": int(args.selection_seed),
        "num_tasks": int(args.num_tasks),
        "evaluation_num_envs": int(args.evaluation_num_envs),
        "smoke": bool(args.smoke),
        "policy_training": False,
        "policy_updates": False,
        "replay_writes": False,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not fresh: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(args.torch_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(args.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.torch_threads))
    environment_args, source_config = _reconstruct_platform_environment_args(
        source_artifact,
        device=args.device,
        seed=args.selection_seed,
    )
    environment_args.evaluation_num_envs = int(args.evaluation_num_envs)
    environment_args.evaluation_progress_interval_tasks = int(
        args.progress_interval_tasks
    )
    tasks = generate_stratified_navigation_tasks(
        num_tasks=args.num_tasks,
        seed=args.selection_seed,
    )
    save_navigation_tasks(
        output_dir / "platform_navigation_tasks.json",
        tasks,
        seed=args.selection_seed,
        role=(
            "execution_smoke"
            if args.smoke
            else "predeclared_fixed_platform_navigation_gate"
        ),
    )
    policy_source = Path(__file__).resolve().parent / "train_uav_energy_delivery_sac.py"
    running = {
        **plan,
        "status": "RUNNING",
        "pid": os.getpid(),
        "source_git_sha": source_config.get("git_sha"),
        "evaluator_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "evaluator_git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "policy_source": str(policy_source),
        "policy_source_sha256": _file_sha256(policy_source),
        "exact_command": [sys.executable, *sys.argv],
        "gate_thresholds": {
            "overall_success_rate_minimum": 0.98,
            "each_distance_bucket_success_minimum": 0.95,
            "mean_path_ratio_maximum": 1.10,
            "boundary_contact_step_rate_strict_maximum": 0.01,
            "obstacle_collision_steps": 0,
        },
    }
    _write_json(output_dir / "RUNNING.json", running)
    started = time.perf_counter()
    try:
        result = evaluate_navigation_tasks(
            HeuristicGoalPolicy(),
            environment_args,
            tasks,
            global_env_transitions=0,
            output_path=output_dir / "navigation_result.json",
        )
        gate = _gate_summary(result)
        completed = {
            **running,
            **gate,
            "status": "PASS" if gate["navigation_gate_passed"] else "FAIL",
            "completion_semantics": (
                "fixed_nonlearning_platform_navigation_gate_completed"
            ),
            "wall_clock_seconds": time.perf_counter() - started,
            "downstream_navigation_ready": bool(
                gate["navigation_gate_passed"] and not args.smoke
            ),
            "downstream_stages_authorized": bool(
                gate["navigation_gate_passed"] and not args.smoke
            ),
            "smoke_not_formal_evidence": bool(args.smoke),
        }
        _write_json(output_dir / "COMPLETED.json", completed)
        if args.smoke:
            _write_json(output_dir / "SMOKE_COMPLETED.json", completed)
        elif gate["navigation_gate_passed"]:
            _write_json(output_dir / "NAVIGATION_READY.json", completed)
        else:
            _write_json(
                output_dir / "STOPPED_NAVIGATION_NOT_READY.json", completed
            )
        (output_dir / "RUNNING.json").unlink(missing_ok=True)
        return 0 if args.smoke or gate["navigation_gate_passed"] else 4
    except Exception as error:
        _write_json(
            output_dir / "FAILED.json",
            {
                **running,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (output_dir / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
