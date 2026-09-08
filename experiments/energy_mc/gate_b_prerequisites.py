from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np


FIXED_BASELINE_CONTRACT_SCHEMA = "fixed_navigation_energy_research_contract_v1"
FIXED_BASELINE_VALIDATION_SCHEMA = "fixed_baseline_battery_validation_v2"
FIXED_BASELINE_RECALIBRATION_SCHEMA = (
    "fixed_baseline_continuous_endurance_recalibration_v1"
)
CONTINUOUS_ENDURANCE_ESTIMAND = (
    "time_to_true_energy_exhaustion_under_continuous_task_workload"
)


@dataclass(frozen=True)
class GateBPrerequisiteAudit:
    passed: bool
    status: str
    failures: tuple[str, ...]
    calibrated_battery_capacity: float | None
    navigation_success_rate: float | None
    calibration_success_rate: float | None
    observed_endurance_seconds: float | None
    navigation_artifact_schema: str | None = None
    navigation_checkpoint_sha256: str | None = None
    fixed_baseline_contract: bool = False
    baseline_id: str | None = None
    navigation_performance_metrics_descriptive: bool = False
    battery_validation_schema: str | None = None
    endurance_estimand: str | None = None
    continuous_workload_until_depletion: bool | None = None
    censored_run_count: int | None = None
    battery_validation_runs: int | None = None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class NavigationArtifactView:
    metrics: Mapping[str, object]
    schema: str
    wrapped: bool
    checkpoint_sha256: str | None
    authorization_passed: bool | None
    fixed_baseline_contract: bool = False
    baseline_id: str | None = None
    performance_metrics_descriptive: bool = False


def navigation_artifact_view(
    navigation: Mapping[str, object],
) -> NavigationArtifactView:
    """Normalize supported navigation wrappers without losing authorization.

    Repair runs store held-out metrics under ``final_navigation``. The generic
    checkpoint evaluator stores its formal 500k row under
    ``formal_500k_navigation_gate``. A raw metric table remains usable by local
    metric tests, but it has no explicit downstream authorization or checkpoint
    identity and therefore cannot alone authorize formal Stage B.
    """

    if navigation.get("schema") == FIXED_BASELINE_CONTRACT_SCHEMA:
        metrics = navigation.get("navigation_metrics")
        if not isinstance(metrics, Mapping):
            metrics = {}
        baseline_id = navigation.get("baseline_id")
        authorized = bool(
            navigation.get("status") == "AUTHORIZED"
            and navigation.get("energy_research_authorized") is True
            and navigation.get("frozen_policy_required") is True
            and navigation.get("performance_metrics_are_descriptive") is True
            and isinstance(baseline_id, str)
            and bool(baseline_id)
        )
        return NavigationArtifactView(
            metrics=metrics,
            schema=FIXED_BASELINE_CONTRACT_SCHEMA,
            wrapped=True,
            checkpoint_sha256=_optional_sha256(
                navigation.get("checkpoint_sha256")
            ),
            authorization_passed=authorized,
            fixed_baseline_contract=True,
            baseline_id=str(baseline_id) if isinstance(baseline_id, str) else None,
            performance_metrics_descriptive=True,
        )

    final_navigation = navigation.get("final_navigation")
    if isinstance(final_navigation, Mapping):
        return NavigationArtifactView(
            metrics=final_navigation,
            schema="navigation_repair_completion_wrapper",
            wrapped=True,
            checkpoint_sha256=_optional_sha256(navigation.get("checkpoint_sha256")),
            authorization_passed=bool(
                navigation.get("navigation_gate_passed") is True
                and navigation.get("downstream_navigation_ready") is True
                and navigation.get("downstream_stages_authorized") is True
            ),
        )
    formal_gate = navigation.get("formal_500k_navigation_gate")
    if isinstance(formal_gate, Mapping):
        return NavigationArtifactView(
            metrics=formal_gate,
            schema="checkpoint_evaluator_completion_wrapper",
            wrapped=True,
            checkpoint_sha256=_optional_sha256(formal_gate.get("checkpoint_sha256")),
            authorization_passed=bool(
                formal_gate.get("navigation_gate_passed") is True
                and navigation.get("downstream_navigation_ready") is True
            ),
        )
    return NavigationArtifactView(
        metrics=navigation,
        schema="raw_navigation_metrics",
        wrapped=False,
        checkpoint_sha256=_optional_sha256(navigation.get("checkpoint_sha256")),
        authorization_passed=None,
    )


def audit_gate_b_prerequisites(
    navigation: Mapping[str, object],
    calibration: Mapping[str, object],
    validation: Mapping[str, object],
    *,
    required_navigation_transitions: int = 500_000,
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
    navigation_view = navigation_artifact_view(navigation)
    navigation_metrics = navigation_view.metrics
    navigation_success = _optional_float(
        navigation_metrics.get("overall_success_rate")
    )
    calibration_success = _optional_float(calibration.get("calibration_success_rate"))
    observed_endurance = _optional_float(validation.get("mean_depletion_time"))
    capacity = _optional_float(calibration.get("calibrated_battery_capacity"))
    validation_schema = (
        str(validation["schema"])
        if isinstance(validation.get("schema"), str)
        else None
    )
    endurance_estimand = (
        str(validation["endurance_estimand"])
        if isinstance(validation.get("endurance_estimand"), str)
        else None
    )
    continuous_workload = validation.get("continuous_workload_until_depletion")
    if not isinstance(continuous_workload, bool):
        continuous_workload = None
    censored_run_count = _optional_int(validation.get("censored_run_count"))
    validation_runs = _optional_int(validation.get("battery_validation_runs"))

    _require_equal(
        failures,
        navigation_metrics.get("global_env_transitions"),
        required_navigation_transitions,
        "navigation transition count",
    )
    _require_equal(
        failures,
        navigation_metrics.get("num_tasks"),
        required_navigation_tasks,
        "navigation task count",
    )
    navigation_buckets = navigation_metrics.get("distance_bucket_success")
    if navigation_view.fixed_baseline_contract:
        _require_fixed_baseline_navigation_telemetry(
            failures,
            navigation_metrics,
        )
    else:
        if navigation_success is None or navigation_success < navigation_success_threshold:
            failures.append(
                f"navigation success must be >= {navigation_success_threshold:.3f}"
            )
        _require_bucket_success(
            failures,
            navigation_buckets,
            threshold=bucket_success_threshold,
            label="navigation",
        )
        _require_maximum(
            failures,
            navigation_metrics.get("mean_path_ratio"),
            maximum_path_ratio,
            "navigation mean path ratio",
        )
        _require_below(
            failures,
            navigation_metrics.get("boundary_contact_step_rate"),
            maximum_boundary_contact_step_rate,
            "navigation boundary-contact step rate",
        )
        if "obstacle_collision_steps" not in navigation_metrics:
            failures.append("navigation obstacle-collision audit is missing")
        elif _optional_int(navigation_metrics.get("obstacle_collision_steps")) != 0:
            failures.append("navigation evaluation contains obstacle collisions")
    if (
        navigation_view.wrapped
        and navigation_view.authorization_passed is not True
    ):
        failures.append(
            "navigation completion wrapper does not authorize downstream stages"
        )

    _require_equal(
        failures,
        calibration.get("num_tasks"),
        required_calibration_tasks,
        "calibration task count",
    )
    if navigation_view.fixed_baseline_contract:
        _require_fixed_baseline_calibration_evidence(failures, calibration)
    else:
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
    td_replay_writes = calibration.get("td_replay_writes")
    if td_replay_writes is None:
        if (
            not navigation_view.fixed_baseline_contract
            or calibration.get("td_training") is not False
        ):
            failures.append("calibration TD replay-write audit is missing")
    elif _optional_int(td_replay_writes) != 0:
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
    if navigation_view.fixed_baseline_contract:
        if validation.get("schema") != FIXED_BASELINE_VALIDATION_SCHEMA:
            failures.append(
                "fixed-baseline validation does not use the continuous-workload v2 schema"
            )
        if validation.get("endurance_estimand") != CONTINUOUS_ENDURANCE_ESTIMAND:
            failures.append(
                "fixed-baseline validation has the wrong endurance estimand"
            )
        if validation.get("continuous_workload_until_depletion") is not True:
            failures.append(
                "fixed-baseline validation did not preserve workload continuity until depletion"
            )
        if _optional_int(validation.get("censored_run_count")) != 0:
            failures.append(
                "fixed-baseline validation contains censored endurance runs"
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
        status="PASS" if passed else "FAIL_GATE_B_PREREQUISITES",
        failures=tuple(failures),
        calibrated_battery_capacity=capacity,
        navigation_success_rate=navigation_success,
        calibration_success_rate=calibration_success,
        observed_endurance_seconds=observed_endurance,
        navigation_artifact_schema=navigation_view.schema,
        navigation_checkpoint_sha256=navigation_view.checkpoint_sha256,
        fixed_baseline_contract=navigation_view.fixed_baseline_contract,
        baseline_id=navigation_view.baseline_id,
        navigation_performance_metrics_descriptive=(
            navigation_view.performance_metrics_descriptive
        ),
        battery_validation_schema=validation_schema,
        endurance_estimand=endurance_estimand,
        continuous_workload_until_depletion=continuous_workload,
        censored_run_count=censored_run_count,
        battery_validation_runs=validation_runs,
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


def _optional_sha256(value: object) -> str | None:
    text = str(value).lower() if value is not None else ""
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        return None
    return text


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


def _require_fixed_baseline_navigation_telemetry(
    failures: list[str],
    metrics: Mapping[str, object],
) -> None:
    """Require stratification variables without imposing deployment thresholds."""

    for field in (
        "overall_success_rate",
        "mean_path_ratio",
        "boundary_contact_step_rate",
        "obstacle_collision_steps",
        "episodes_with_obstacle_collision",
    ):
        if _optional_float(metrics.get(field)) is None:
            failures.append(f"fixed-baseline navigation telemetry is missing: {field}")
    buckets = metrics.get("distance_bucket_success")
    if not isinstance(buckets, Mapping) or not buckets:
        failures.append(
            "fixed-baseline navigation distance-bucket telemetry is missing"
        )
    elif any(_optional_float(value) is None for value in buckets.values()):
        failures.append(
            "fixed-baseline navigation distance-bucket telemetry is malformed"
        )


def _require_fixed_baseline_calibration_evidence(
    failures: list[str],
    calibration: Mapping[str, object],
) -> None:
    if calibration.get("fixed_baseline_calibration_evaluable") is not True:
        failures.append("fixed-baseline calibration evidence is not evaluable")
    if calibration.get("calibration_failure_taxonomy_complete") is not True:
        failures.append("fixed-baseline calibration failure taxonomy is incomplete")
    tasks = calibration.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 500:
        failures.append("fixed-baseline calibration must retain all 500 task records")
        return
    required_fields = {
        "task_index",
        "distance_bucket",
        "success",
        "end_reason",
        "total_realized_energy",
        "simulation_flight_time",
    }
    if any(not isinstance(task, Mapping) for task in tasks):
        failures.append("fixed-baseline calibration contains malformed task records")
        return
    if any(not required_fields.issubset(task) for task in tasks):
        failures.append("fixed-baseline calibration task taxonomy is incomplete")
    if not any(bool(task.get("success")) for task in tasks):
        failures.append("fixed-baseline calibration has no successful energy trajectory")


__all__ = [
    "CONTINUOUS_ENDURANCE_ESTIMAND",
    "FIXED_BASELINE_CONTRACT_SCHEMA",
    "FIXED_BASELINE_RECALIBRATION_SCHEMA",
    "FIXED_BASELINE_VALIDATION_SCHEMA",
    "GateBPrerequisiteAudit",
    "NavigationArtifactView",
    "audit_gate_b_prerequisites",
    "navigation_artifact_view",
]
