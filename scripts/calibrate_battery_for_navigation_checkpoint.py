from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from experiments.energy_mc.gate_b_prerequisites import navigation_artifact_view
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    freeze_navigation_policy,
    generate_stratified_navigation_tasks,
    navigation_energy_gate_passed,
    navigation_safety_gate_passed,
    run_battery_calibration,
    run_battery_validation,
    save_navigation_tasks,
)


FORMAL_NAVIGATION_TRANSITIONS = 500_000
FORMAL_CALIBRATION_TASKS = 500
FORMAL_VALIDATION_RUNS = 100


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate battery capacity for one navigation-ready frozen checkpoint"
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navigation-evaluation-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--battery-calibration-tasks", type=int, default=500)
    parser.add_argument("--battery-validation-runs", type=int, default=100)
    parser.add_argument("--battery-calibration-seed", type=int, default=80_001)
    parser.add_argument("--battery-validation-seed", type=int, default=90_001)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument("--evaluation-progress-interval-tasks", type=int, default=25)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.battery_calibration_tasks != FORMAL_CALIBRATION_TASKS:
        parser.error("formal calibration requires exactly 500 tasks")
    if args.battery_validation_runs != FORMAL_VALIDATION_RUNS:
        parser.error("formal endurance validation requires exactly 100 runs")
    if args.evaluation_num_envs <= 0:
        parser.error("evaluation worker count must be positive")
    if args.target_nominal_endurance_minutes != 30.0:
        parser.error("the preregistered first Gate-B calibration uses 30 minutes")
    return args


def navigation_readiness_failures(
    evaluation: dict[str, object],
    *,
    require_wrapped_artifact: bool = False,
) -> list[str]:
    failures: list[str] = []
    view = navigation_artifact_view(evaluation)
    metrics = view.metrics
    if require_wrapped_artifact and not view.wrapped:
        failures.append(
            "formal calibration requires a navigation completion wrapper"
        )
    if view.wrapped and view.authorization_passed is not True:
        failures.append("navigation completion wrapper does not authorize calibration")
    if int(metrics.get("num_tasks", -1)) != 500:
        failures.append("navigation readiness evaluation must contain 500 tasks")
    if int(metrics.get("global_env_transitions", -1)) != FORMAL_NAVIGATION_TRANSITIONS:
        failures.append("navigation readiness evaluation must use the 500k checkpoint")
    if view.fixed_baseline_contract:
        required_telemetry = {
            "overall_success_rate",
            "distance_bucket_success",
            "mean_path_ratio",
            "boundary_contact_step_rate",
            "obstacle_collision_steps",
            "episodes_with_obstacle_collision",
        }
        missing = sorted(required_telemetry.difference(metrics))
        if missing:
            failures.append(
                f"fixed-baseline navigation telemetry is incomplete: {missing}"
            )
    else:
        if not navigation_energy_gate_passed(metrics):
            failures.append("navigation energy/readiness gate failed")
        if not navigation_safety_gate_passed(metrics):
            failures.append("navigation collision/boundary gate failed")
    return failures


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    artifact = cli.artifact.expanduser().resolve()
    checkpoint = cli.checkpoint.expanduser().resolve()
    navigation_path = cli.navigation_evaluation_json.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    if checkpoint.name != "checkpoint_transition_500000.zip":
        raise ValueError("formal battery calibration requires checkpoint_transition_500000.zip")
    for path in (artifact, checkpoint, navigation_path):
        if not path.exists():
            raise FileNotFoundError(path)
    navigation = json.loads(navigation_path.read_text(encoding="utf-8"))
    navigation_view = navigation_artifact_view(navigation)
    checkpoint_digest = file_sha256(checkpoint)
    navigation_digest = file_sha256(navigation_path)
    artifact_config_path = artifact / "config.json"
    if not artifact_config_path.is_file():
        raise FileNotFoundError(artifact_config_path)
    artifact_config_digest = file_sha256(artifact_config_path)
    failures = navigation_readiness_failures(
        navigation,
        require_wrapped_artifact=True,
    )
    if navigation_view.checkpoint_sha256 != checkpoint_digest:
        failures.append(
            "navigation completion wrapper checkpoint SHA does not match "
            "the calibration checkpoint"
        )
    if cli.dry_run:
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": checkpoint_digest,
                    "navigation_failures": failures,
                    "would_run": not failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return int(bool(failures))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"battery calibration output is not fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "source_artifact": str(artifact),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest,
        "navigation_evaluation": str(navigation_path),
        "navigation_readiness_failures": failures,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).splitlines(),
        "exact_command": [sys.executable, *sys.argv],
    }
    write_json(output / "RUNNING.json", provenance)
    if failures:
        stopped = {
            **provenance,
            "status": "STOPPED_NAVIGATION_NOT_READY",
        }
        write_json(output / "STOPPED_NAVIGATION_NOT_READY.json", stopped)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 2
    try:
        args, _ = reconstruct_environment_args(
            artifact,
            device=cli.device,
            seed=cli.battery_calibration_seed,
        )
        args.device = cli.device
        args.battery_calibration_tasks = cli.battery_calibration_tasks
        args.battery_validation_runs = cli.battery_validation_runs
        args.battery_calibration_seed = cli.battery_calibration_seed
        args.battery_validation_seed = cli.battery_validation_seed
        args.evaluation_num_envs = cli.evaluation_num_envs
        args.evaluation_progress_interval_tasks = (
            cli.evaluation_progress_interval_tasks
        )
        args.target_nominal_endurance_minutes = (
            cli.target_nominal_endurance_minutes
        )
        args.smoke = False
        args.pilot = False
        torch.set_num_threads(cli.torch_threads)
        os.environ.setdefault("OMP_NUM_THREADS", str(cli.torch_threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(cli.torch_threads))
        policy = JacobianBridgeSAC.load(checkpoint, device=cli.device)
        if int(policy.num_timesteps) != FORMAL_NAVIGATION_TRANSITIONS:
            raise ValueError("checkpoint payload does not record 500k transitions")
        freeze_navigation_policy(policy)
        tasks = generate_stratified_navigation_tasks(
            num_tasks=cli.battery_calibration_tasks,
            seed=cli.battery_calibration_seed,
        )
        save_navigation_tasks(
            output / "battery_calibration_tasks.json",
            tasks,
            seed=cli.battery_calibration_seed,
            role="battery_calibration_not_navigation_training",
        )
        calibration = run_battery_calibration(policy, args, tasks, output)
        task_rows = calibration.get("tasks")
        taxonomy_fields = {
            "task_index",
            "distance_bucket",
            "success",
            "end_reason",
            "total_realized_energy",
            "simulation_flight_time",
        }
        taxonomy_complete = bool(
            isinstance(task_rows, list)
            and len(task_rows) == FORMAL_CALIBRATION_TASKS
            and all(
                isinstance(row, dict) and taxonomy_fields.issubset(row)
                for row in task_rows
            )
        )
        calibration.update(
            {
                "legacy_battery_calibration_navigation_valid": calibration.get(
                    "battery_calibration_navigation_valid"
                ),
                "fixed_baseline_calibration_evaluable": bool(
                    navigation_view.fixed_baseline_contract
                    and taxonomy_complete
                    and int(calibration.get("num_successful_tasks", 0)) > 0
                ),
                "calibration_failure_taxonomy_complete": taxonomy_complete,
            }
        )
        calibration.update(
            {
                "navigation_checkpoint_sha256": checkpoint_digest,
                "navigation_evaluation_sha256": navigation_digest,
                "navigation_artifact_config_sha256": artifact_config_digest,
            }
        )
        write_json(output / "battery_calibration.json", calibration)
        if (
            not navigation_view.fixed_baseline_contract
            and not bool(calibration["battery_calibration_navigation_valid"])
        ):
            stopped = {
                "status": "STOPPED_CALIBRATION_NAVIGATION_INVALID",
                "calibration_success_rate": calibration["calibration_success_rate"],
                "calibration_distance_bucket_success": calibration[
                    "calibration_distance_bucket_success"
                ],
            }
            write_json(
                output / "STOPPED_CALIBRATION_NAVIGATION_INVALID.json",
                stopped,
            )
            (output / "RUNNING.json").unlink(missing_ok=True)
            return 3
        capacity = float(calibration["calibrated_battery_capacity"])
        validation = run_battery_validation(
            policy,
            args,
            battery_capacity=capacity,
            output=output,
        )
        calibration_digest = file_sha256(output / "battery_calibration.json")
        validation.update(
            {
                "navigation_checkpoint_sha256": checkpoint_digest,
                "navigation_evaluation_sha256": navigation_digest,
                "navigation_artifact_config_sha256": artifact_config_digest,
                "battery_calibration_sha256": calibration_digest,
            }
        )
        write_json(output / "battery_validation.json", validation)
        if not bool(validation["battery_calibration_valid"]):
            stopped = {
                "status": "STOPPED_BATTERY_ENDURANCE_INVALID",
                "battery_capacity": capacity,
                "all_runs_depleted": validation["all_runs_depleted"],
                "relative_endurance_error": validation["relative_endurance_error"],
            }
            write_json(output / "STOPPED_BATTERY_ENDURANCE_INVALID.json", stopped)
            (output / "RUNNING.json").unlink(missing_ok=True)
            return 4
        completed = {
            "status": "COMPLETED",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_digest,
            "navigation_evaluation": str(navigation_path),
            "navigation_evaluation_sha256": navigation_digest,
            "navigation_artifact": str(artifact),
            "navigation_artifact_config_sha256": artifact_config_digest,
            "calibrated_battery_capacity": capacity,
            "calibration_success_rate": calibration["calibration_success_rate"],
            "mean_depletion_time": validation["mean_depletion_time"],
            "relative_endurance_error": validation["relative_endurance_error"],
            "battery_calibration_json": str(output / "battery_calibration.json"),
            "battery_validation_json": str(output / "battery_validation.json"),
            "battery_calibration_sha256": calibration_digest,
            "battery_validation_sha256": file_sha256(
                output / "battery_validation.json"
            ),
        }
        write_json(output / "COMPLETED.json", completed)
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 0
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
