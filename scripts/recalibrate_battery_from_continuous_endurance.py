#!/usr/bin/env python3
"""Refine frozen-R3 battery capacity from a failed continuous-endurance Gate.

The failed run is treated as calibration data only.  The emitted attestation
requires a disjoint seed interval for the subsequent confirmatory validation.
No policy, dynamics, safety filter, or telemetry-cost parameter is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.gate_b_prerequisites import (
    CONTINUOUS_ENDURANCE_ESTIMAND,
    FIXED_BASELINE_RECALIBRATION_SCHEMA,
    FIXED_BASELINE_VALIDATION_SCHEMA,
)


FORMAL_VALIDATION_RUNS = 100


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


def _finite_float(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def build_refined_calibration(
    *,
    initial_calibration_path: Path,
    failed_validation_path: Path,
    confirmatory_seed_start: int,
    confirmatory_runs: int = FORMAL_VALIDATION_RUNS,
) -> dict[str, object]:
    initial_path = initial_calibration_path.resolve()
    validation_path = failed_validation_path.resolve()
    calibration = read_object(initial_path)
    validation = read_object(validation_path)

    if calibration.get("baseline_id") != "R3":
        raise ValueError("initial calibration is not bound to R3")
    if calibration.get("fixed_baseline_calibration_evaluable") is not True:
        raise ValueError("initial fixed-baseline calibration is not evaluable")
    if calibration.get("calibration_failure_taxonomy_complete") is not True:
        raise ValueError("initial calibration failure taxonomy is incomplete")
    tasks = calibration.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 500:
        raise ValueError("initial calibration must retain all 500 task records")
    if calibration.get("policy_unchanged") is not True:
        raise ValueError("initial calibration changed the frozen policy")
    if calibration.get("sac_training") is not False:
        raise ValueError("initial calibration contains SAC training")
    if calibration.get("td_training") is not False:
        raise ValueError("initial calibration contains TD training")

    if validation.get("schema") != FIXED_BASELINE_VALIDATION_SCHEMA:
        raise ValueError("source validation does not use the formal v2 schema")
    if validation.get("baseline_id") != "R3":
        raise ValueError("source validation is not bound to R3")
    if validation.get("endurance_estimand") != CONTINUOUS_ENDURANCE_ESTIMAND:
        raise ValueError("source validation has the wrong endurance estimand")
    if validation.get("continuous_workload_until_depletion") is not True:
        raise ValueError("source validation is not a continuous workload")
    if validation.get("all_runs_depleted") is not True:
        raise ValueError("source validation contains non-depletion endpoints")
    if int(validation.get("censored_run_count", -1)) != 0:
        raise ValueError("source validation contains censored runs")
    if int(validation.get("battery_validation_runs", -1)) != FORMAL_VALIDATION_RUNS:
        raise ValueError("source validation must contain exactly 100 runs")
    if validation.get("battery_calibration_valid") is not False:
        raise ValueError("only a failed endurance Gate may be repurposed here")
    if validation.get("battery_calibration_sha256") != file_sha256(initial_path):
        raise ValueError("source validation is not linked to the initial calibration")

    source_capacity = _finite_float(
        calibration.get("calibrated_battery_capacity"),
        "initial calibrated battery capacity",
    )
    validation_capacity = _finite_float(
        validation.get("battery_capacity"), "validation battery capacity"
    )
    if not np.isclose(source_capacity, validation_capacity, rtol=1e-12, atol=1e-12):
        raise ValueError("initial calibration and source validation capacities differ")

    records = validation.get("records")
    if not isinstance(records, list) or len(records) != FORMAL_VALIDATION_RUNS:
        raise ValueError("source validation must retain all 100 records")
    if any(not isinstance(record, Mapping) for record in records):
        raise ValueError("source validation contains malformed records")
    if any(record.get("energy_exhausted") is not True for record in records):
        raise ValueError("source validation record is not a true depletion endpoint")
    if any(
        record.get("continuous_workload_until_depletion") is not True
        for record in records
    ):
        raise ValueError("source validation record violates workload continuity")

    depletion_times = np.asarray(
        [
            _finite_float(record.get("actual_depletion_time"), "depletion time")
            for record in records
        ],
        dtype=np.float64,
    )
    if np.any(depletion_times <= 0.0):
        raise ValueError("depletion times must be positive")
    observed_mean = float(np.mean(depletion_times))
    recorded_mean = _finite_float(validation.get("mean_depletion_time"), "mean depletion time")
    if not np.isclose(observed_mean, recorded_mean, rtol=1e-12, atol=1e-9):
        raise ValueError("source validation mean does not match its retained records")
    target_seconds = _finite_float(
        validation.get("target_nominal_endurance_seconds"), "target endurance"
    )
    if not np.isclose(target_seconds, 1800.0, rtol=0.0, atol=1e-12):
        raise ValueError("the frozen R3 target must be exactly 30 minutes")
    tolerance = _finite_float(
        validation.get("engineering_calibration_tolerance_fraction"),
        "engineering tolerance",
    )
    relative_error = (observed_mean - target_seconds) / target_seconds
    if not np.isclose(
        relative_error,
        _finite_float(validation.get("relative_endurance_error"), "relative error"),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("source validation relative error is inconsistent")
    if abs(relative_error) <= tolerance:
        raise ValueError("source validation did not fail the endurance tolerance")

    source_seeds = [int(record["run_seed"]) for record in records]
    if len(set(source_seeds)) != FORMAL_VALIDATION_RUNS:
        raise ValueError("source validation seeds are not unique")
    if confirmatory_runs != FORMAL_VALIDATION_RUNS:
        raise ValueError("confirmatory validation requires exactly 100 runs")
    confirmatory_seeds = set(
        range(confirmatory_seed_start, confirmatory_seed_start + confirmatory_runs)
    )
    if confirmatory_seeds.intersection(source_seeds):
        raise ValueError("confirmatory seed interval overlaps the refinement sample")

    refined_capacity = source_capacity * target_seconds / observed_mean
    effective_power = refined_capacity / target_seconds
    result = dict(calibration)
    result.update(
        {
            "schema": FIXED_BASELINE_RECALIBRATION_SCHEMA,
            "stage": "battery_calibration_refinement",
            "claim_scope": (
                "continuous-workload mean-endurance capacity refinement; the source "
                "100-run failure is calibration-only and cannot serve as confirmation"
            ),
            "calibrated_battery_capacity": refined_capacity,
            "capacity_for_20min": effective_power * 20.0 * 60.0,
            "capacity_for_30min": effective_power * 30.0 * 60.0,
            "capacity_for_40min": effective_power * 40.0 * 60.0,
            "mean_power": effective_power,
            "mean_power_estimator": (
                "source_capacity / sample_mean_time_to_true_depletion"
            ),
            "effective_continuous_workload_power_for_mean_endurance": effective_power,
            "source_task_calibration_mean_power": calibration.get("mean_power"),
            "source_task_calibrated_battery_capacity": source_capacity,
            "source_initial_calibration": str(initial_path),
            "source_initial_calibration_sha256": file_sha256(initial_path),
            "source_initial_calibration_schema": calibration.get("schema"),
            "source_continuous_endurance_validation": str(validation_path),
            "source_continuous_endurance_validation_sha256": file_sha256(validation_path),
            "source_continuous_endurance_validation_schema": validation.get("schema"),
            "source_validation_role": "capacity_refinement_only_not_confirmation",
            "source_validation_runs": FORMAL_VALIDATION_RUNS,
            "source_validation_seed_min": min(source_seeds),
            "source_validation_seed_max": max(source_seeds),
            "source_validation_mean_depletion_time": observed_mean,
            "source_validation_relative_endurance_error": relative_error,
            "capacity_refinement_formula": (
                "C_refined = C_source * target_seconds / source_mean_depletion_seconds"
            ),
            "capacity_scale_factor": target_seconds / observed_mean,
            "confirmatory_validation_required": True,
            "confirmatory_validation_runs": confirmatory_runs,
            "confirmatory_validation_seed_start": confirmatory_seed_start,
            "confirmatory_validation_seed_end_inclusive": (
                confirmatory_seed_start + confirmatory_runs - 1
            ),
            "confirmatory_seeds_disjoint_from_refinement_sample": True,
            "capacity_scaling_contract": {
                "navigation_policy_observation_excludes_remaining_energy": True,
                "telemetry_power_equation_excludes_remaining_energy": True,
                "pre_depletion_navigation_dynamics_exclude_battery_capacity": True,
                "frozen_policy_and_environment_required": True,
            },
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-calibration", type=Path, required=True)
    parser.add_argument("--failed-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirmatory-validation-seed", type=int, required=True)
    parser.add_argument(
        "--confirmatory-validation-runs", type=int, default=FORMAL_VALIDATION_RUNS
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_refined_calibration(
        initial_calibration_path=args.initial_calibration,
        failed_validation_path=args.failed_validation,
        confirmatory_seed_start=args.confirmatory_validation_seed,
        confirmatory_runs=args.confirmatory_validation_runs,
    )
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite calibration attestation: {output}")
    write_json(output, payload)
    print(f"Refined battery capacity: {payload['calibrated_battery_capacity']:.12f}")
    print(f"Confirmatory seeds: {payload['confirmatory_validation_seed_start']}-"
          f"{payload['confirmatory_validation_seed_end_inclusive']}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
