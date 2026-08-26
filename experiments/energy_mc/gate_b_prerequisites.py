from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class GateBPrerequisiteAudit:
    passed: bool
    status: str
    failures: tuple[str, ...]
    calibrated_battery_capacity: float | None
    navigation_success_rate: float | None
    calibration_success_rate: float | None
    observed_endurance_seconds: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def audit_gate_b_prerequisites(
    navigation: Mapping[str, object],
    calibration: Mapping[str, object],
    validation: Mapping[str, object],
    *,
    required_navigation_tasks: int = 500,
    required_calibration_tasks: int = 500,
    required_validation_runs: int = 100,
    navigation_success_threshold: float = 0.98,
    bucket_success_threshold: float = 0.95,
    maximum_path_ratio: float = 1.10,
    maximum_boundary_contact_step_rate: float = 0.01,
    expected_endurance_minutes: float = 30.0,
) -> GateBPrerequisiteAudit:
    failures: list[str] = []
    navigation_success = _optional_float(navigation.get("overall_success_rate"))
    calibration_success = _optional_float(calibration.get("calibration_success_rate"))
    observed_endurance = _optional_float(validation.get("mean_depletion_time"))
    capacity = _optional_float(calibration.get("calibrated_battery_capacity"))

    _require_equal(
        failures,
        navigation.get("num_tasks"),
        required_navigation_tasks,
        "navigation task count",
    )
    if navigation_success is None or navigation_success < navigation_success_threshold:
        failures.append(
            f"navigation success must be >= {navigation_success_threshold:.3f}"
        )
    navigation_buckets = navigation.get("distance_bucket_success")
    _require_bucket_success(
        failures,
        navigation_buckets,
        threshold=bucket_success_threshold,
        label="navigation",
    )
    _require_maximum(
        failures,
        navigation.get("mean_path_ratio"),
        maximum_path_ratio,
        "navigation mean path ratio",
    )
    _require_below(
        failures,
        navigation.get("boundary_contact_step_rate"),
        maximum_boundary_contact_step_rate,
        "navigation boundary-contact step rate",
    )
    if "obstacle_collision_steps" not in navigation:
        failures.append("navigation obstacle-collision audit is missing")
    elif _optional_int(navigation.get("obstacle_collision_steps")) != 0:
        failures.append("navigation evaluation contains obstacle collisions")

    _require_equal(
        failures,
        calibration.get("num_tasks"),
        required_calibration_tasks,
        "calibration task count",
    )
    if calibration.get("battery_calibration_navigation_valid") is not True:
        failures.append("battery calibration navigation validity gate failed")
    if calibration_success is None or calibration_success < navigation_success_threshold:
        failures.append(
            f"calibration success must be >= {navigation_success_threshold:.3f}"
        )
    _require_bucket_success(
        failures,
        calibration.get("calibration_distance_bucket_success"),
        threshold=bucket_success_threshold,
        label="calibration",
    )
    if calibration.get("policy_unchanged") is not True:
        failures.append("calibration changed the frozen navigation policy")
    if calibration.get("sac_training") is not False:
        failures.append("SAC training must be disabled during calibration")
    if _optional_int(calibration.get("td_replay_writes")) != 0:
        failures.append("calibration wrote TD replay")
    if calibration.get("energy_source") != "TelemetryCostModel.realized_cost":
        failures.append("calibration did not use realized TelemetryCostModel energy")
    if capacity is None or capacity <= 0.0:
        failures.append("calibrated battery capacity must be finite and positive")
    _require_close(
        failures,
        calibration.get("target_nominal_endurance_minutes"),
        expected_endurance_minutes,
        "calibration target endurance minutes",
    )

    _require_equal(
        failures,
        validation.get("battery_validation_runs"),
        required_validation_runs,
        "battery validation run count",
    )
    if validation.get("battery_calibration_valid") is not True:
        failures.append("battery endurance sanity gate failed")
    if validation.get("all_runs_depleted") is not True:
        failures.append(
            "battery validation contains censored/task-limit runs instead of depletion"
        )
    tolerance = _optional_float(
        validation.get("engineering_calibration_tolerance_fraction")
    )
    relative_error = _optional_float(validation.get("relative_endurance_error"))
    if (
        tolerance is None
        or relative_error is None
        or abs(relative_error) > tolerance
    ):
        failures.append("observed endurance lies outside the engineering tolerance")
    validation_capacity = _optional_float(validation.get("battery_capacity"))
    if (
        capacity is not None
        and validation_capacity is not None
        and not np.isclose(capacity, validation_capacity, rtol=1e-9, atol=1e-9)
    ):
        failures.append("calibration and validation battery capacities differ")
    _require_close(
        failures,
        validation.get("target_nominal_endurance_minutes"),
        expected_endurance_minutes,
        "validation target endurance minutes",
    )

    passed = not failures
    return GateBPrerequisiteAudit(
        passed=passed,
        status="PASS" if passed else "FAIL",
        failures=tuple(failures),
        calibrated_battery_capacity=capacity,
        navigation_success_rate=navigation_success,
        calibration_success_rate=calibration_success,
        observed_endurance_seconds=observed_endurance,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    return scalar if np.isfinite(scalar) else None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_equal(
    failures: list[str],
    actual: object,
    expected: int,
    label: str,
) -> None:
    try:
        matches = int(actual) == expected
    except (TypeError, ValueError):
        matches = False
    if not matches:
        failures.append(f"{label} must equal {expected}")


def _require_maximum(
    failures: list[str],
    actual: object,
    maximum: float,
    label: str,
) -> None:
    value = _optional_float(actual)
    if value is None or value > maximum:
        failures.append(f"{label} must be <= {maximum:.6g}")


def _require_below(
    failures: list[str],
    actual: object,
    maximum: float,
    label: str,
) -> None:
    value = _optional_float(actual)
    if value is None or value >= maximum:
        failures.append(f"{label} must be < {maximum:.6g}")


def _require_close(
    failures: list[str],
    actual: object,
    expected: float,
    label: str,
) -> None:
    value = _optional_float(actual)
    if value is None or not np.isclose(value, expected, rtol=1e-9, atol=1e-9):
        failures.append(f"{label} must equal {expected:.6g}")


def _require_bucket_success(
    failures: list[str],
    values: object,
    *,
    threshold: float,
    label: str,
) -> None:
    if not isinstance(values, Mapping) or not values:
        failures.append(f"{label} distance-bucket success is missing")
        return
    failed = {
        str(bucket): _optional_float(value)
        for bucket, value in values.items()
        if _optional_float(value) is None or float(value) < threshold
    }
    if failed:
        failures.append(
            f"{label} distance buckets below {threshold:.3f}: {failed}"
        )


__all__ = ["GateBPrerequisiteAudit", "audit_gate_b_prerequisites"]
