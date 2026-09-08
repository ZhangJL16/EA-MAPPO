from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.run_jacobian_safety_energy_1m import parse_args as parse_jseb_args
from scripts.train_uav_energy_delivery_sac import (
    evaluate_navigation_tasks,
    generate_stratified_navigation_tasks,
    navigation_energy_gate_passed,
    navigation_safety_gate_passed,
    save_navigation_tasks,
)


DEFAULT_SELECTION_SEED = 170_001
DEFAULT_SELECTION_TASKS = 500
LEGACY_PHASE2_BUDGET_FLAGS = (
    "--phase2a-transitions",
    "--phase2b-transitions",
    "--phase2c-transitions",
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_transition(path: Path) -> int:
    prefix = "checkpoint_transition_"
    if path.suffix != ".zip" or not path.stem.startswith(prefix):
        raise ValueError(f"invalid checkpoint filename: {path.name}")
    return int(path.stem.removeprefix(prefix))


def discover_checkpoints(artifact: Path) -> list[tuple[int, Path]]:
    root = artifact / "phase1_navigation"
    checkpoints = sorted(
        (checkpoint_transition(path), path)
        for path in root.glob("checkpoint_transition_*.zip")
    )
    if not checkpoints:
        raise FileNotFoundError(f"no Phase-1 checkpoints found in {root}")
    return checkpoints


def select_checkpoints(
    checkpoints: list[tuple[int, Path]],
    requested_transitions: list[int] | None,
) -> list[tuple[int, Path]]:
    if requested_transitions is None:
        return checkpoints
    requested = [int(value) for value in requested_transitions]
    if len(set(requested)) != len(requested):
        raise ValueError("checkpoint transitions must be unique")
    available = {transition: path for transition, path in checkpoints}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"requested checkpoints are missing: {missing}")
    return [(transition, available[transition]) for transition in requested]


def migrate_legacy_jseb_command(arguments: list[str]) -> tuple[list[str], dict[str, object]]:
    migrated: list[str] = []
    legacy_budgets: dict[str, int] = {}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument not in LEGACY_PHASE2_BUDGET_FLAGS:
            migrated.append(argument)
            index += 1
            continue
        if argument in legacy_budgets:
            raise ValueError(f"duplicate legacy argument: {argument}")
        if index + 1 >= len(arguments):
            raise ValueError(f"legacy argument is missing its value: {argument}")
        try:
            value = int(arguments[index + 1])
        except ValueError as error:
            raise ValueError(f"legacy budget must be an integer: {argument}") from error
        if value < 0:
            raise ValueError(f"legacy budget must be nonnegative: {argument}")
        legacy_budgets[argument] = value
        index += 2
    if legacy_budgets:
        if "--phase2-energy-transitions" in migrated:
            raise ValueError(
                "artifact command mixes legacy phase2 budgets with the unified budget"
            )
        missing = set(LEGACY_PHASE2_BUDGET_FLAGS) - set(legacy_budgets)
        if missing:
            raise ValueError(
                "artifact command has an incomplete legacy phase2 budget set: "
                + ", ".join(sorted(missing))
            )
        unified_budget = sum(legacy_budgets.values())
        migrated.extend(["--phase2-energy-transitions", str(unified_budget)])
    else:
        unified_budget = None
    audit = {
        "legacy_phase2_budget_migrated": bool(legacy_budgets),
        "legacy_phase2_budgets": legacy_budgets,
        "unified_phase2_energy_transitions": unified_budget,
    }
    return migrated, audit


def normalize_environment_reconstruction_command(
    arguments: list[str],
) -> tuple[list[str], dict[str, object]]:
    """Remove training-runtime flags while preserving the R5 environment.

    R5 was launched by an orchestration wrapper with flags that the canonical
    JSEB parser does not own. For reconstruction, its base parser contract is
    R4 plus sampled-data robust HOCBF. The returned args are relabeled R5 after
    parsing; no training is performed through this compatibility path.
    """

    flags_with_values = {
        "--training-vec-env",
        "--training-vec-start-method",
    }
    boolean_flags = {"--hocbf-sampled-data-robust"}
    normalized: list[str] = []
    removed: dict[str, object] = {}
    source_variant: str | None = None
    index = 0
    while index < len(arguments):
        flag = arguments[index]
        if flag in flags_with_values:
            if index + 1 >= len(arguments):
                raise ValueError(f"source command is missing a value for {flag}")
            removed[flag] = arguments[index + 1]
            index += 2
            continue
        if flag in boolean_flags:
            removed[flag] = True
            index += 1
            continue
        if flag == "--navigation-repair-variant":
            if index + 1 >= len(arguments):
                raise ValueError(
                    "source command is missing a navigation repair variant"
                )
            source_variant = arguments[index + 1]
            normalized.extend(
                [flag, "R4" if source_variant == "R5" else source_variant]
            )
            index += 2
            continue
        normalized.append(flag)
        index += 1
    return normalized, {
        "source_navigation_repair_variant": source_variant,
        "parser_compatibility_variant": (
            "R4" if source_variant == "R5" else source_variant
        ),
        "evaluation_irrelevant_source_flags_removed": removed,
        "sampled_data_robust_hocbf_preserved": bool(
            removed.get("--hocbf-sampled-data-robust") is True
        ),
    }


def reconstruct_environment_args(artifact: Path, *, device: str, seed: int):
    config = json.loads((artifact / "config.json").read_text(encoding="utf-8"))
    command = list(config["exact_command"])
    migrated_command, migration_audit = migrate_legacy_jseb_command(command[1:])
    reconstruction_command, reconstruction_audit = (
        normalize_environment_reconstruction_command(migrated_command)
    )
    args = parse_jseb_args(reconstruction_command)
    if reconstruction_audit["source_navigation_repair_variant"] == "R5":
        if not reconstruction_audit["sampled_data_robust_hocbf_preserved"]:
            raise ValueError("R5 reconstruction requires sampled-data robust HOCBF")
        args.navigation_repair_variant = "R5"
        args.hocbf_sampled_data_robust = True
    args.device = device
    args.eval_task_seed = seed
    args.eval_navigation_tasks = DEFAULT_SELECTION_TASKS
    config = dict(config)
    config["command_migration"] = migration_audit
    config["environment_reconstruction"] = reconstruction_audit
    return args, config


def compact_result(transition: int, checkpoint: Path, result: dict, elapsed: float) -> dict:
    bucket_values = [
        float(value)
        for value in result["distance_bucket_success"].values()
        if value is not None
    ]
    energy_gate_passed = navigation_energy_gate_passed(result)
    safety_gate_passed = navigation_safety_gate_passed(result)
    gate_failures = []
    if not energy_gate_passed:
        gate_failures.append("navigation_energy_readiness_gate_failed")
    if not safety_gate_passed:
        gate_failures.append("navigation_collision_boundary_gate_failed")
    return {
        "checkpoint_transition": transition,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "num_tasks": int(result["num_tasks"]),
        "global_env_transitions": int(result["global_env_transitions"]),
        "evaluation_env_transitions": int(result["evaluation_env_transitions"]),
        "overall_success_rate": float(result["overall_success_rate"]),
        "minimum_distance_bucket_success": min(bucket_values),
        "distance_bucket_success": result["distance_bucket_success"],
        "mean_path_ratio": float(result["mean_path_ratio"]),
        "obstacle_collision_steps": int(result["obstacle_collision_steps"]),
        "obstacle_collision_episode_rate": float(
            result["obstacle_collision_episode_rate"]
        ),
        "boundary_contact_step_rate": float(result["boundary_contact_step_rate"]),
        "hocbf_intervention_step_rate": float(
            result["hocbf_intervention_step_rate"]
        ),
        "hocbf_emergency_brake_step_rate": float(
            result["hocbf_emergency_brake_step_rate"]
        ),
        "nominal_safe_action_rate": float(result["nominal_safe_action_rate"]),
        "projection_valid_step_rate": float(result["projection_valid_step_rate"]),
        "navigation_energy_gate_passed": energy_gate_passed,
        "navigation_safety_gate_passed": safety_gate_passed,
        "navigation_gate_passed": energy_gate_passed and safety_gate_passed,
        "navigation_gate_failures": gate_failures,
        "wall_clock_seconds": elapsed,
    }


def write_summary_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "checkpoint_transition",
        "overall_success_rate",
        "minimum_distance_bucket_success",
        "mean_path_ratio",
        "obstacle_collision_steps",
        "obstacle_collision_episode_rate",
        "boundary_contact_step_rate",
        "hocbf_intervention_step_rate",
        "hocbf_emergency_brake_step_rate",
        "nominal_safe_action_rate",
        "projection_valid_step_rate",
        "navigation_energy_gate_passed",
        "navigation_safety_gate_passed",
        "navigation_gate_passed",
        "evaluation_env_transitions",
        "wall_clock_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate every JSEB Phase-1 checkpoint on one held-out task set"
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--num-tasks", type=int, default=DEFAULT_SELECTION_TASKS)
    parser.add_argument(
        "--checkpoint-transitions",
        type=int,
        nargs="+",
        help="evaluate only the named environment-transition checkpoints",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        default=6,
        help=(
            "parallel environment workers; policy inference remains central and "
            "deterministic"
        ),
    )
    parser.add_argument(
        "--evaluation-progress-interval-tasks",
        type=int,
        default=25,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(arguments)
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.evaluation_num_envs <= 0:
        parser.error("--evaluation-num-envs must be positive")
    if args.evaluation_progress_interval_tasks <= 0:
        parser.error("--evaluation-progress-interval-tasks must be positive")
    return args


def main() -> int:
    cli = parse_args()
    artifact = Path(cli.artifact).resolve()
    output = (
        Path(cli.output_dir).resolve()
        if cli.output_dir
        else artifact / "checkpoint_selection_500tasks"
    )
    checkpoints = select_checkpoints(
        discover_checkpoints(artifact),
        cli.checkpoint_transitions,
    )
    if cli.num_tasks != DEFAULT_SELECTION_TASKS:
        raise ValueError("the formal checkpoint-selection sweep requires exactly 500 tasks")
    if cli.dry_run:
        print(json.dumps({"checkpoints": [step for step, _ in checkpoints]}, indent=2))
        return 0
    if output.exists() and any(output.iterdir()):
        completed = output / "COMPLETED.json"
        if completed.exists():
            print(f"checkpoint sweep already completed: {completed}")
            return 0
        raise FileExistsError(f"checkpoint sweep output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(cli.torch_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cli.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cli.torch_threads))
    environment_args, source_config = reconstruct_environment_args(
        artifact,
        device=cli.device,
        seed=cli.selection_seed,
    )
    environment_args.evaluation_num_envs = int(cli.evaluation_num_envs)
    environment_args.evaluation_progress_interval_tasks = int(
        cli.evaluation_progress_interval_tasks
    )
    tasks = generate_stratified_navigation_tasks(
        num_tasks=cli.num_tasks,
        seed=cli.selection_seed,
    )
    save_navigation_tasks(
        output / "checkpoint_selection_tasks.json",
        tasks,
        seed=cli.selection_seed,
        role="checkpoint_selection_not_final_test",
    )
    protocol = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "source_artifact": str(artifact),
        "source_git_sha": source_config["git_sha"],
        "evaluator_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "evaluator_git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "selection_seed": cli.selection_seed,
        "num_tasks_per_checkpoint": cli.num_tasks,
        "checkpoint_transitions": [step for step, _ in checkpoints],
        "task_set_role": "model_selection_not_final_paper_test",
        "training_replay_writes": False,
        "policy_updates": False,
        "td_updates": False,
        "device": cli.device,
        "torch_threads": cli.torch_threads,
        "evaluation_num_envs": cli.evaluation_num_envs,
        "evaluation_execution": (
            "parallel_environments_central_batched_deterministic_policy"
            if cli.evaluation_num_envs > 1
            else "serial_environment_deterministic_policy"
        ),
        "evaluation_progress_interval_tasks": (
            cli.evaluation_progress_interval_tasks
        ),
        "exact_command": sys.argv,
    }
    write_json(output / "RUNNING.json", protocol)
    rows: list[dict] = []
    try:
        for index, (transition, checkpoint) in enumerate(checkpoints, start=1):
            result_path = output / f"eval_checkpoint_{transition:06d}.json"
            started = time.perf_counter()
            model = JacobianBridgeSAC.load(checkpoint, device=cli.device)
            result = evaluate_navigation_tasks(
                model,
                environment_args,
                tasks,
                global_env_transitions=transition,
                output_path=result_path,
            )
            elapsed = time.perf_counter() - started
            row = compact_result(transition, checkpoint, result, elapsed)
            rows.append(row)
            write_json(
                output / "PROGRESS.json",
                {
                    "status": "RUNNING",
                    "completed_checkpoints": index,
                    "total_checkpoints": len(checkpoints),
                    "latest": row,
                },
            )
            write_summary_csv(output / "checkpoint_summary.csv", rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        ranked = sorted(
            rows,
            key=lambda row: (
                -row["overall_success_rate"],
                row["obstacle_collision_steps"],
                row["mean_path_ratio"],
                row["hocbf_intervention_step_rate"],
            ),
        )
        summary = {
            "status": "COMPLETED",
            "completion_semantics": "evaluation_completed_not_automatic_gate_pass",
            "selection_warning": (
                "The selected checkpoint must be evaluated on a fresh final test set; "
                "these 500 tasks are now validation data."
            ),
            "ranking_rule": (
                "lexicographic: success descending, collision steps ascending, "
                "path ratio ascending, intervention rate ascending"
            ),
            "best_checkpoint_by_selection_rule": ranked[0],
            "results": rows,
        }
        formal_500k = next(
            (row for row in rows if row["checkpoint_transition"] == 500_000),
            None,
        )
        summary["navigation_training_completed"] = formal_500k is not None
        summary["downstream_navigation_ready"] = (
            None
            if formal_500k is None
            else bool(formal_500k["navigation_gate_passed"])
        )
        summary["formal_500k_navigation_gate"] = formal_500k
        write_json(output / "checkpoint_summary.json", summary)
        write_json(output / "COMPLETED.json", summary)
        if formal_500k is not None and not formal_500k["navigation_gate_passed"]:
            write_json(
                output / "STOPPED_NAVIGATION_NOT_READY.json",
                {
                    "status": "STOPPED_NAVIGATION_NOT_READY",
                    "navigation_training_completed": True,
                    "downstream_navigation_ready": False,
                    "formal_500k_navigation_gate": formal_500k,
                    "downstream_stages_authorized": False,
                },
            )
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "completed_results": rows,
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
