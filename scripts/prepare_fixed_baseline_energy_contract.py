#!/usr/bin/env python3
"""Freeze R3 as the conditional platform for return-to-charge energy research.

The source R3 evaluation and calibration remain immutable historical evidence.
This script writes a compact platform contract and an attested calibration copy
whose extra fields are reproducible from the source files and their hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.gate_b_prerequisites import (
    FIXED_BASELINE_CONTRACT_SCHEMA,
)


FORMAL_NAVIGATION_TRANSITIONS = 500_000
FORMAL_TASKS = 500
R3_CHECKPOINT_NAME = "checkpoint_transition_500000.zip"


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


def _finite(value: object, *, label: str, positive: bool = False) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(scalar) or (positive and scalar <= 0.0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return scalar


def _legacy_navigation_failures(metrics: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    if _finite(metrics.get("overall_success_rate"), label="overall success") < 0.98:
        failures.append("overall_success_rate<0.98")
    buckets = metrics.get("distance_bucket_success")
    if not isinstance(buckets, Mapping) or not buckets:
        failures.append("distance_bucket_success_missing")
    elif min(_finite(value, label="distance bucket") for value in buckets.values()) < 0.95:
        failures.append("minimum_distance_bucket_success<0.95")
    if _finite(metrics.get("mean_path_ratio"), label="mean path ratio") > 1.10:
        failures.append("mean_path_ratio>1.10")
    if int(metrics.get("obstacle_collision_steps", -1)) != 0:
        failures.append("obstacle_collision_steps!=0")
    if _finite(
        metrics.get("boundary_contact_step_rate"),
        label="boundary contact rate",
    ) >= 0.01:
        failures.append("boundary_contact_step_rate>=0.01")
    return failures


def build_fixed_baseline_contract(
    *,
    source_navigation: Mapping[str, object],
    source_navigation_path: Path,
    checkpoint: Path,
    artifact: Path,
    artifact_config_path: Path,
    baseline_id: str = "R3",
) -> dict[str, object]:
    if baseline_id != "R3":
        raise ValueError("the currently authorized fixed baseline is exactly R3")
    if source_navigation.get("variant") != baseline_id:
        raise ValueError("source navigation wrapper is not the selected R3 variant")
    metrics = source_navigation.get("final_navigation")
    if not isinstance(metrics, Mapping):
        raise ValueError("R3 source wrapper is missing final_navigation")
    if int(metrics.get("num_tasks", -1)) != FORMAL_TASKS:
        raise ValueError("R3 source evaluation must contain exactly 500 tasks")
    if int(metrics.get("global_env_transitions", -1)) != FORMAL_NAVIGATION_TRANSITIONS:
        raise ValueError("R3 source evaluation must identify the 500k checkpoint")
    if checkpoint.name != R3_CHECKPOINT_NAME:
        raise ValueError(f"R3 checkpoint must be named {R3_CHECKPOINT_NAME}")
    checkpoint_sha = file_sha256(checkpoint)
    if source_navigation.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("R3 wrapper checkpoint SHA does not match the selected checkpoint")
    if source_navigation.get("navigation_gate_passed") is not False:
        raise ValueError("the derived contract must preserve R3's legacy Gate failure")

    required_metrics = (
        "overall_success_rate",
        "distance_bucket_success",
        "mean_path_ratio",
        "boundary_contact_step_rate",
        "boundary_contact_episode_rate",
        "obstacle_collision_steps",
        "obstacle_collision_step_rate",
        "obstacle_collision_episode_rate",
        "episodes_with_obstacle_collision",
        "num_tasks",
        "global_env_transitions",
    )
    missing = [field for field in required_metrics if field not in metrics]
    if missing:
        raise ValueError(f"R3 source evaluation lacks required telemetry: {missing}")
    compact_metrics = {field: metrics[field] for field in required_metrics}
    legacy_failures = _legacy_navigation_failures(metrics)
    if not legacy_failures:
        raise ValueError("R3 unexpectedly passes the preserved legacy Gate")

    return {
        "schema": FIXED_BASELINE_CONTRACT_SCHEMA,
        "status": "AUTHORIZED",
        "baseline_id": baseline_id,
        "authorization_basis": "user_selected_R3_for_energy_research_2026-08-30",
        "energy_research_authorized": True,
        "frozen_policy_required": True,
        "performance_metrics_are_descriptive": True,
        "legacy_navigation_gate_passed": False,
        "legacy_navigation_gate_failures": legacy_failures,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "source_navigation_artifact": str(artifact.resolve()),
        "source_navigation_evaluation": str(source_navigation_path.resolve()),
        "source_navigation_evaluation_sha256": file_sha256(source_navigation_path),
        "source_navigation_artifact_config": str(artifact_config_path.resolve()),
        "source_navigation_artifact_config_sha256": file_sha256(
            artifact_config_path
        ),
        "navigation_metrics": compact_metrics,
        "estimand": (
            "conditional_stranding_throughput_frontier_difference_given_"
            "frozen_R3_and_HOCBF"
        ),
        "required_outcome_taxonomy": [
            "task_success",
            "task_step_limit",
            "navigation_collision",
            "boundary_contact",
            "energy_exhaustion",
            "return_failure",
            "charger_arrival",
            "premature_return",
        ],
        "non_claim": (
            "Authorization for conditional energy/return research does not convert "
            "R3 into a passing deployment-quality navigation or collision-safety result."
        ),
    }


def attest_fixed_baseline_calibration(
    *,
    source_calibration: Mapping[str, object],
    source_calibration_path: Path,
    contract: Mapping[str, object],
    contract_path: Path,
) -> dict[str, object]:
    tasks = source_calibration.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != FORMAL_TASKS:
        raise ValueError("source R3 calibration must retain exactly 500 task records")
    required = {
        "task_index",
        "distance_bucket",
        "success",
        "end_reason",
        "total_realized_energy",
        "simulation_flight_time",
    }
    if any(not isinstance(row, Mapping) or not required.issubset(row) for row in tasks):
        raise ValueError("source R3 calibration has incomplete task taxonomy")
    indices = sorted(int(row["task_index"]) for row in tasks)
    if indices != list(range(FORMAL_TASKS)):
        raise ValueError("source R3 calibration task indices are not exactly 0..499")
    for index, row in enumerate(tasks):
        _finite(
            row["total_realized_energy"],
            label=f"task {index} energy",
            positive=True,
        )
        _finite(
            row["simulation_flight_time"],
            label=f"task {index} flight time",
            positive=True,
        )
        if not isinstance(row["success"], bool) or not str(row["end_reason"]):
            raise ValueError(f"task {index} has malformed success/end_reason")

    successful = [row for row in tasks if bool(row["success"])]
    success_count = len(successful)
    if success_count == 0:
        raise ValueError("source R3 calibration has no successful trajectories")
    if int(source_calibration.get("num_successful_tasks", -1)) != success_count:
        raise ValueError("source R3 calibration success count is inconsistent")
    observed_rate = _finite(
        source_calibration.get("calibration_success_rate"),
        label="calibration success rate",
    )
    if not math.isclose(observed_rate, success_count / FORMAL_TASKS, abs_tol=1e-12):
        raise ValueError("source R3 calibration success rate is inconsistent")

    successful_energy = sum(float(row["total_realized_energy"]) for row in successful)
    successful_time = sum(float(row["simulation_flight_time"]) for row in successful)
    recomputed_power = successful_energy / successful_time
    recorded_power = _finite(
        source_calibration.get("mean_power"),
        label="calibration mean power",
        positive=True,
    )
    if not math.isclose(recomputed_power, recorded_power, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("source R3 calibration mean power does not reproduce")
    target_minutes = _finite(
        source_calibration.get("target_nominal_endurance_minutes"),
        label="target endurance",
        positive=True,
    )
    capacity = _finite(
        source_calibration.get("calibrated_battery_capacity"),
        label="calibrated battery capacity",
        positive=True,
    )
    if not math.isclose(
        capacity,
        recorded_power * target_minutes * 60.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("source R3 battery capacity does not reproduce")
    if source_calibration.get("policy_unchanged") is not True:
        raise ValueError("source R3 calibration changed the navigation policy")
    if source_calibration.get("sac_training") is not False:
        raise ValueError("source R3 calibration contains SAC training")
    if source_calibration.get("td_training") is not False:
        raise ValueError("source R3 calibration contains TD training")
    if source_calibration.get("energy_source") != "TelemetryCostModel.realized_cost":
        raise ValueError("source R3 calibration used the wrong energy source")

    result = dict(source_calibration)
    result.update(
        {
            "schema": "fixed_baseline_battery_calibration_attestation_v1",
            "baseline_id": contract["baseline_id"],
            "fixed_baseline_calibration_evaluable": True,
            "calibration_failure_taxonomy_complete": True,
            "calibration_end_reason_counts": dict(
                sorted(Counter(str(row["end_reason"]) for row in tasks).items())
            ),
            "legacy_battery_calibration_navigation_valid": source_calibration.get(
                "battery_calibration_navigation_valid"
            ),
            "source_battery_calibration": str(source_calibration_path.resolve()),
            "source_battery_calibration_sha256": file_sha256(
                source_calibration_path
            ),
            "navigation_checkpoint_sha256": contract["checkpoint_sha256"],
            "navigation_evaluation_sha256": file_sha256(contract_path),
            "navigation_artifact_config_sha256": contract[
                "source_navigation_artifact_config_sha256"
            ],
            "fixed_baseline_contract": str(contract_path.resolve()),
            "fixed_baseline_contract_sha256": file_sha256(contract_path),
            "claim_scope": (
                "battery power calibration conditional on successful frozen-R3 "
                "trajectories; all failures retained for stratified sensitivity"
            ),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-navigation-evaluation", type=Path, required=True)
    parser.add_argument("--source-battery-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-id", default="R3")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact = args.artifact.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    source_navigation_path = args.source_navigation_evaluation.expanduser().resolve()
    source_calibration_path = args.source_battery_calibration.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    artifact_config_path = artifact / "config.json"
    for path in (
        artifact,
        checkpoint,
        source_navigation_path,
        source_calibration_path,
        artifact_config_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    source_navigation = read_object(source_navigation_path)
    source_calibration = read_object(source_calibration_path)
    contract = build_fixed_baseline_contract(
        source_navigation=source_navigation,
        source_navigation_path=source_navigation_path,
        checkpoint=checkpoint,
        artifact=artifact,
        artifact_config_path=artifact_config_path,
        baseline_id=args.baseline_id,
    )
    contract_path = output / "navigation_platform_contract.json"
    write_json(contract_path, contract)
    calibration = attest_fixed_baseline_calibration(
        source_calibration=source_calibration,
        source_calibration_path=source_calibration_path,
        contract=contract,
        contract_path=contract_path,
    )
    calibration_path = output / "battery_calibration.json"
    write_json(calibration_path, calibration)
    write_json(
        output / "PREPARED.json",
        {
            "status": "PREPARED_FOR_BATTERY_VALIDATION",
            "baseline_id": args.baseline_id,
            "navigation_platform_contract": str(contract_path),
            "navigation_platform_contract_sha256": file_sha256(contract_path),
            "battery_calibration": str(calibration_path),
            "battery_calibration_sha256": file_sha256(calibration_path),
            "calibrated_battery_capacity": calibration[
                "calibrated_battery_capacity"
            ],
            "legacy_navigation_gate_passed": False,
            "energy_research_authorized": True,
            "next_stage": "100_run_battery_endurance_validation",
        },
    )


if __name__ == "__main__":
    main()
