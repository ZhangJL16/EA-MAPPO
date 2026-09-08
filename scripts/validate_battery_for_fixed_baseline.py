#!/usr/bin/env python3
"""Run resumable 100-run endurance validation for an authorized fixed baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.gate_b_prerequisites import (
    CONTINUOUS_ENDURANCE_ESTIMAND,
    FIXED_BASELINE_RECALIBRATION_SCHEMA,
    FIXED_BASELINE_VALIDATION_SCHEMA,
    navigation_artifact_view,
)
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.evaluate_jseb_checkpoints import reconstruct_environment_args
from scripts.train_uav_energy_delivery_sac import (
    freeze_navigation_policy,
    run_battery_validation,
)


FORMAL_VALIDATION_RUNS = 100
FORMAL_NAVIGATION_TRANSITIONS = 500_000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON must contain an object: {path}")
    return value


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def audit_validation_inputs(
    *,
    artifact: Path,
    checkpoint: Path,
    contract_path: Path,
    calibration_path: Path,
    validation_seed_start: int | None = None,
) -> tuple[dict[str, object], dict[str, object], float, list[str]]:
    failures: list[str] = []
    contract = read_object(contract_path)
    calibration = read_object(calibration_path)
    view = navigation_artifact_view(contract)
    checkpoint_sha = file_sha256(checkpoint)
    contract_sha = file_sha256(contract_path)
    config_path = artifact / "config.json"
    config_sha = file_sha256(config_path)
    if not view.fixed_baseline_contract or view.authorization_passed is not True:
        failures.append("navigation contract is not an authorized fixed baseline")
    if view.baseline_id != "R3":
        failures.append("the selected fixed baseline is not R3")
    if view.checkpoint_sha256 != checkpoint_sha:
        failures.append("fixed-baseline contract checkpoint SHA mismatch")
    if contract.get("source_navigation_artifact_config_sha256") != config_sha:
        failures.append("fixed-baseline contract artifact config SHA mismatch")
    if calibration.get("fixed_baseline_calibration_evaluable") is not True:
        failures.append("fixed-baseline calibration is not evaluable")
    if calibration.get("calibration_failure_taxonomy_complete") is not True:
        failures.append("fixed-baseline calibration taxonomy is incomplete")
    if calibration.get("navigation_checkpoint_sha256") != checkpoint_sha:
        failures.append("calibration checkpoint SHA mismatch")
    if calibration.get("navigation_evaluation_sha256") != contract_sha:
        failures.append("calibration fixed-baseline contract SHA mismatch")
    if calibration.get("navigation_artifact_config_sha256") != config_sha:
        failures.append("calibration artifact config SHA mismatch")
    if int(calibration.get("num_tasks", -1)) != 500:
        failures.append("fixed-baseline calibration must contain 500 tasks")
    if calibration.get("policy_unchanged") is not True:
        failures.append("calibration did not preserve the frozen R3 policy")
    if calibration.get("sac_training") is not False:
        failures.append("calibration contains SAC training")
    if calibration.get("td_training") is not False:
        failures.append("calibration contains TD training")
    try:
        capacity = float(calibration.get("calibrated_battery_capacity"))
    except (TypeError, ValueError):
        capacity = float("nan")
    if not math.isfinite(capacity) or capacity <= 0.0:
        failures.append("calibrated battery capacity must be finite and positive")
    if calibration.get("schema") == FIXED_BASELINE_RECALIBRATION_SCHEMA:
        failures.extend(
            audit_refined_calibration_provenance(
                calibration,
                validation_seed_start=validation_seed_start,
            )
        )
    return contract, calibration, capacity, failures


def audit_refined_calibration_provenance(
    calibration: Mapping[str, object],
    *,
    validation_seed_start: int | None,
) -> list[str]:
    """Fail closed on refinement/confirmation leakage or broken provenance."""

    failures: list[str] = []
    if calibration.get("source_validation_role") != (
        "capacity_refinement_only_not_confirmation"
    ):
        failures.append("continuous-endurance source is not calibration-only")
    if calibration.get("confirmatory_validation_required") is not True:
        failures.append("refined calibration does not require confirmation")
    if calibration.get("confirmatory_seeds_disjoint_from_refinement_sample") is not True:
        failures.append("refinement does not attest disjoint confirmatory seeds")

    try:
        source_initial_path = Path(
            str(calibration["source_initial_calibration"])
        ).resolve()
        source_validation_path = Path(
            str(calibration["source_continuous_endurance_validation"])
        ).resolve()
    except (KeyError, TypeError, ValueError):
        failures.append("refined calibration source paths are missing")
        return failures
    for label, path, sha_field in (
        (
            "initial calibration",
            source_initial_path,
            "source_initial_calibration_sha256",
        ),
        (
            "continuous-endurance calibration sample",
            source_validation_path,
            "source_continuous_endurance_validation_sha256",
        ),
    ):
        if not path.is_file():
            failures.append(f"refined calibration {label} is missing")
        elif calibration.get(sha_field) != file_sha256(path):
            failures.append(f"refined calibration {label} SHA mismatch")

    numeric_fields: dict[str, float] = {}
    for field in (
        "source_task_calibrated_battery_capacity",
        "source_validation_mean_depletion_time",
        "calibrated_battery_capacity",
        "target_nominal_endurance_minutes",
    ):
        try:
            value = float(calibration[field])
        except (KeyError, TypeError, ValueError):
            failures.append(f"refined calibration field is invalid: {field}")
            continue
        if not math.isfinite(value) or value <= 0.0:
            failures.append(f"refined calibration field is invalid: {field}")
            continue
        numeric_fields[field] = value
    if len(numeric_fields) == 4:
        target_seconds = numeric_fields["target_nominal_endurance_minutes"] * 60.0
        expected_capacity = (
            numeric_fields["source_task_calibrated_battery_capacity"]
            * target_seconds
            / numeric_fields["source_validation_mean_depletion_time"]
        )
        if not math.isclose(
            numeric_fields["calibrated_battery_capacity"],
            expected_capacity,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            failures.append("refined capacity does not match its declared formula")

    try:
        source_seed_min = int(calibration["source_validation_seed_min"])
        source_seed_max = int(calibration["source_validation_seed_max"])
        required_seed_start = int(calibration["confirmatory_validation_seed_start"])
        required_seed_end = int(
            calibration["confirmatory_validation_seed_end_inclusive"]
        )
        required_runs = int(calibration["confirmatory_validation_runs"])
    except (KeyError, TypeError, ValueError):
        failures.append("refined calibration seed-split contract is invalid")
        return failures
    if required_runs != FORMAL_VALIDATION_RUNS:
        failures.append("refined calibration confirmation must contain 100 runs")
    if required_seed_end != required_seed_start + required_runs - 1:
        failures.append("refined calibration confirmatory seed interval is inconsistent")
    if max(source_seed_min, required_seed_start) <= min(
        source_seed_max, required_seed_end
    ):
        failures.append("refinement and confirmatory seed intervals overlap")
    if validation_seed_start is None:
        failures.append("confirmatory seed start was not supplied for refinement audit")
    elif int(validation_seed_start) != required_seed_start:
        failures.append("runtime confirmatory seed differs from the preregistered split")
    return failures


def aggregate_validation_batches(
    *,
    batch_payloads: list[Mapping[str, object]],
    expected_runs: int,
    battery_capacity: float,
    target_minutes: float,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    batch_files: list[dict[str, object]] = []
    total_transitions = 0
    wall_seconds = 0.0
    for payload in batch_payloads:
        batch_records = payload.get("records")
        if not isinstance(batch_records, list):
            raise ValueError("validation batch is missing records")
        records.extend(dict(record) for record in batch_records)
        total_transitions += int(payload.get("battery_validation_env_transitions", 0))
        execution = payload.get("execution")
        if isinstance(execution, Mapping):
            wall_seconds += float(execution.get("wall_clock_seconds", 0.0))
        batch_files.append(
            {
                "batch_start": payload.get("batch_start"),
                "batch_end_exclusive": payload.get("batch_end_exclusive"),
                "batch_validation_sha256": payload.get("batch_validation_sha256"),
            }
        )
    records.sort(key=lambda row: int(row["run_index"]))
    indices = [int(row["run_index"]) for row in records]
    if indices != list(range(expected_runs)):
        raise ValueError("validation batches do not cover each global run exactly once")
    depletion_times = np.asarray(
        [float(row["actual_depletion_time"]) for row in records],
        dtype=np.float64,
    )
    expected_seconds = float(target_minutes * 60.0)
    observed_mean = float(np.mean(depletion_times))
    relative_error = float((observed_mean - expected_seconds) / expected_seconds)
    all_depleted = all(bool(row.get("energy_exhausted")) for row in records)
    continuous_workload = all(
        bool(row.get("continuous_workload_until_depletion", False))
        for row in records
    )
    censored_run_count = int(
        sum(not bool(row.get("energy_exhausted")) for row in records)
    )
    total_task_step_limit_rollovers = int(
        sum(
            int(row.get("task_step_limit_rollovers_before_depletion", 0))
            for row in records
        )
    )
    calibration_valid = bool(
        all_depleted
        and continuous_workload
        and censored_run_count == 0
        and abs(relative_error) <= 0.20
    )
    return {
        "schema": FIXED_BASELINE_VALIDATION_SCHEMA,
        "stage": "battery_endurance_validation",
        "baseline_id": "R3",
        "endurance_estimand": CONTINUOUS_ENDURANCE_ESTIMAND,
        "engineering_calibration_tolerance_fraction": 0.20,
        "target_nominal_endurance_minutes": target_minutes,
        "target_nominal_endurance_seconds": expected_seconds,
        "battery_capacity": battery_capacity,
        "battery_validation_runs": len(records),
        "battery_validation_env_transitions": total_transitions,
        "mean_depletion_time": observed_mean,
        "median_depletion_time": float(np.median(depletion_times)),
        "std_depletion_time": float(np.std(depletion_times)),
        "p10_depletion_time": float(np.quantile(depletion_times, 0.10)),
        "p25_depletion_time": float(np.quantile(depletion_times, 0.25)),
        "p50_depletion_time": float(np.quantile(depletion_times, 0.50)),
        "p75_depletion_time": float(np.quantile(depletion_times, 0.75)),
        "p90_depletion_time": float(np.quantile(depletion_times, 0.90)),
        "mean_policy_steps_to_depletion": float(
            np.mean([row["actual_policy_steps_to_depletion"] for row in records])
        ),
        "mean_tasks_before_depletion": float(
            np.mean([row["tasks_before_depletion"] for row in records])
        ),
        "mean_distance_before_depletion": float(
            np.mean([row["distance_before_depletion"] for row in records])
        ),
        "relative_endurance_error": relative_error,
        "censored_run_count": censored_run_count,
        "task_step_limit_rollovers": total_task_step_limit_rollovers,
        "runs_with_task_step_limit_rollover": int(
            sum(
                int(
                    row.get(
                        "task_step_limit_rollovers_before_depletion",
                        0,
                    )
                )
                > 0
                for row in records
            )
        ),
        "runs_exceeding_episode_guard": int(
            sum(
                bool(row.get("episode_guard_exceeded_before_depletion", False))
                for row in records
            )
        ),
        "continuous_workload_until_depletion": continuous_workload,
        "all_runs_depleted": all_depleted,
        "battery_calibration_valid": calibration_valid,
        "execution": {
            "resumable_batches": True,
            "num_batches": len(batch_payloads),
            "sum_batch_wall_clock_seconds": wall_seconds,
        },
        "batches": batch_files,
        "records": records,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--navigation-contract", type=Path, required=True)
    parser.add_argument("--battery-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--battery-validation-runs", type=int, default=100)
    parser.add_argument("--validation-batch-size", type=int, default=10)
    parser.add_argument("--battery-validation-seed", type=int, default=90_001)
    parser.add_argument("--evaluation-num-envs", type=int, default=6)
    parser.add_argument("--target-nominal-endurance-minutes", type=float, default=30.0)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args(argv)
    if args.battery_validation_runs != FORMAL_VALIDATION_RUNS:
        parser.error("formal fixed-baseline validation requires exactly 100 runs")
    if not 1 <= args.validation_batch_size <= FORMAL_VALIDATION_RUNS:
        parser.error("validation batch size must lie in [1, 100]")
    if args.evaluation_num_envs <= 0:
        parser.error("evaluation worker count must be positive")
    if args.target_nominal_endurance_minutes != 30.0:
        parser.error("the frozen R3 capacity targets exactly 30 minutes")
    return args


def main(argv: list[str] | None = None) -> int:
    cli = parse_args(argv)
    artifact = cli.artifact.expanduser().resolve()
    checkpoint = cli.checkpoint.expanduser().resolve()
    contract_path = cli.navigation_contract.expanduser().resolve()
    calibration_path = cli.battery_calibration.expanduser().resolve()
    output = cli.output_dir.expanduser().resolve()
    for path in (
        artifact,
        checkpoint,
        contract_path,
        calibration_path,
        artifact / "config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    completed_path = output / "COMPLETED.json"
    if completed_path.is_file():
        completed = read_object(completed_path)
        return 0 if completed.get("status") == "COMPLETED" else 2
    contract, calibration, capacity, failures = audit_validation_inputs(
        artifact=artifact,
        checkpoint=checkpoint,
        contract_path=contract_path,
        calibration_path=calibration_path,
        validation_seed_start=cli.battery_validation_seed,
    )
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "status": "RUNNING",
        "pid": os.getpid(),
        "baseline_id": "R3",
        "artifact": str(artifact),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "navigation_contract": str(contract_path),
        "navigation_contract_sha256": file_sha256(contract_path),
        "battery_calibration": str(calibration_path),
        "battery_calibration_sha256": file_sha256(calibration_path),
        "endurance_estimand": CONTINUOUS_ENDURANCE_ESTIMAND,
        "input_failures": failures,
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
        write_json(
            output / "STOPPED_INPUT_CONTRACT_INVALID.json",
            {**provenance, "status": "STOPPED_INPUT_CONTRACT_INVALID"},
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        return 2

    args, _ = reconstruct_environment_args(
        artifact,
        device=cli.device,
        seed=cli.battery_validation_seed,
    )
    args.device = cli.device
    args.evaluation_num_envs = cli.evaluation_num_envs
    args.evaluation_progress_interval_tasks = cli.validation_batch_size
    args.target_nominal_endurance_minutes = cli.target_nominal_endurance_minutes
    args.smoke = False
    args.pilot = False
    torch.set_num_threads(cli.torch_threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(cli.torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(cli.torch_threads))
    policy = JacobianBridgeSAC.load(checkpoint, device=cli.device)
    if int(policy.num_timesteps) != FORMAL_NAVIGATION_TRANSITIONS:
        raise ValueError("R3 checkpoint payload does not record 500k transitions")
    freeze_navigation_policy(policy)

    batch_payloads: list[dict[str, object]] = []
    try:
        for batch_start in range(
            0,
            cli.battery_validation_runs,
            cli.validation_batch_size,
        ):
            batch_end = min(
                batch_start + cli.validation_batch_size,
                cli.battery_validation_runs,
            )
            batch_dir = output / "batches" / f"runs_{batch_start:03d}_{batch_end - 1:03d}"
            batch_completed_path = batch_dir / "COMPLETED.json"
            batch_validation_path = batch_dir / "battery_validation.json"
            if batch_completed_path.is_file() and batch_validation_path.is_file():
                batch_completed = read_object(batch_completed_path)
                if (
                    batch_completed.get("battery_validation_sha256")
                    != file_sha256(batch_validation_path)
                ):
                    raise ValueError(f"completed validation batch hash mismatch: {batch_dir}")
                batch_payloads.append(read_object(batch_validation_path))
                continue
            if batch_dir.exists() and any(batch_dir.iterdir()):
                raise FileExistsError(
                    f"incomplete validation batch requires inspection: {batch_dir}"
                )
            batch_dir.mkdir(parents=True, exist_ok=True)
            args.battery_validation_runs = batch_end - batch_start
            args.battery_validation_seed = cli.battery_validation_seed + batch_start
            args.evaluation_num_envs = min(
                cli.evaluation_num_envs,
                args.battery_validation_runs,
            )
            payload = run_battery_validation(
                policy,
                args,
                battery_capacity=capacity,
                output=batch_dir,
            )
            local_records = payload.get("records")
            if not isinstance(local_records, list) or len(local_records) != (
                batch_end - batch_start
            ):
                raise ValueError("validation batch returned the wrong record count")
            global_records = []
            for record in local_records:
                if not isinstance(record, Mapping):
                    raise ValueError("validation batch contains a malformed record")
                copied = dict(record)
                copied["local_run_index"] = int(record["run_index"])
                copied["run_index"] = batch_start + int(record["run_index"])
                copied["run_seed"] = cli.battery_validation_seed + int(
                    copied["run_index"]
                )
                global_records.append(copied)
            payload.update(
                {
                    "schema": "fixed_baseline_battery_validation_batch_v2",
                    "baseline_id": "R3",
                    "batch_start": batch_start,
                    "batch_end_exclusive": batch_end,
                    "records": global_records,
                    "navigation_checkpoint_sha256": file_sha256(checkpoint),
                    "navigation_evaluation_sha256": file_sha256(contract_path),
                    "navigation_artifact_config_sha256": contract[
                        "source_navigation_artifact_config_sha256"
                    ],
                    "battery_calibration_sha256": file_sha256(calibration_path),
                }
            )
            write_json(batch_validation_path, payload)
            batch_sha = file_sha256(batch_validation_path)
            payload["batch_validation_sha256"] = batch_sha
            write_json(
                batch_completed_path,
                {
                    "status": "COMPLETED",
                    "batch_start": batch_start,
                    "batch_end_exclusive": batch_end,
                    "battery_validation_sha256": batch_sha,
                },
            )
            batch_payloads.append(payload)

        summary = aggregate_validation_batches(
            batch_payloads=batch_payloads,
            expected_runs=cli.battery_validation_runs,
            battery_capacity=capacity,
            target_minutes=cli.target_nominal_endurance_minutes,
        )
        summary.update(
            {
                "navigation_checkpoint_sha256": file_sha256(checkpoint),
                "navigation_evaluation_sha256": file_sha256(contract_path),
                "navigation_artifact_config_sha256": contract[
                    "source_navigation_artifact_config_sha256"
                ],
                "battery_calibration_sha256": file_sha256(calibration_path),
                "fixed_baseline_contract": str(contract_path),
                "navigation_performance_metrics_descriptive": True,
            }
        )
        validation_path = output / "battery_validation.json"
        write_json(validation_path, summary)
        if not bool(summary["battery_calibration_valid"]):
            write_json(
                output / "STOPPED_BATTERY_ENDURANCE_INVALID.json",
                {
                    "status": "STOPPED_BATTERY_ENDURANCE_INVALID",
                    "all_runs_depleted": summary["all_runs_depleted"],
                    "continuous_workload_until_depletion": summary[
                        "continuous_workload_until_depletion"
                    ],
                    "censored_run_count": summary["censored_run_count"],
                    "relative_endurance_error": summary["relative_endurance_error"],
                    "battery_validation_sha256": file_sha256(validation_path),
                },
            )
            (output / "RUNNING.json").unlink(missing_ok=True)
            return 4
        write_json(
            completed_path,
            {
                "status": "COMPLETED",
                "baseline_id": "R3",
                "battery_capacity": capacity,
                "battery_validation_json": str(validation_path),
                "battery_validation_sha256": file_sha256(validation_path),
                "mean_depletion_time": summary["mean_depletion_time"],
                "relative_endurance_error": summary["relative_endurance_error"],
                "next_stage": "oracle_decision_headroom",
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
            },
        )
        (output / "RUNNING.json").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
