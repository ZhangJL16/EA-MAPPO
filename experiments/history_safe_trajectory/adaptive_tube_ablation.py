from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from review_bundle.safety.trajectory import (
    HistoryTubeMode,
    SetMembershipBounds,
    adaptive_history_set_membership,
    future_position_error_tube,
    jerk_bounded_set_membership,
)


@dataclass(frozen=True)
class AdaptiveTubeAblationConfig:
    output_dir: str
    seed: int = 20260819
    cases_per_regime: int = 200
    sample_dt: float = 0.1
    history_length: int = 16
    prediction_horizon: float = 1.0
    sensor_error: float = 0.02
    velocity_bound: float = 10.0
    acceleration_bound: float = 6.0
    history_jerk_bound: float = 2.0
    robust_jerk_bound: float = 30.0


def _advance(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    delta_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    next_position = (
        position
        + delta_time * velocity
        + 0.5 * delta_time**2 * acceleration
        + delta_time**3 / 6.0 * jerk
    )
    next_velocity = velocity + delta_time * acceleration + 0.5 * delta_time**2 * jerk
    next_acceleration = acceleration + delta_time * jerk
    return next_position, next_velocity, next_acceleration


def _jerk_schedule(
    regime: str,
    count: int,
    rng: np.random.Generator,
    history_jerk_bound: float,
    robust_jerk_bound: float,
) -> np.ndarray:
    jerk = np.zeros((count - 1, 3), dtype=np.float64)
    if regime == "constant_velocity":
        return jerk
    if regime == "slow_acceleration":
        jerk[:] = rng.uniform(-0.5 * history_jerk_bound, 0.5 * history_jerk_bound, size=(1, 3))
        return jerk
    pulse = rng.uniform(0.65 * robust_jerk_bound, 0.9 * robust_jerk_bound)
    sign = rng.choice((-1.0, 1.0), size=3)
    if regime == "sudden_velocity_change":
        jerk[-3, 0] = sign[0] * pulse
    elif regime == "sudden_direction_change":
        jerk[-4, :2] = sign[:2] * pulse
    elif regime == "stop_and_go":
        jerk[-7, 0] = sign[0] * pulse
        jerk[-4, 0] = -sign[0] * pulse
    else:
        raise ValueError(f"unknown regime: {regime}")
    return jerk


def _simulate_case(
    regime: str,
    config: AdaptiveTubeAblationConfig,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    count = config.history_length
    times = np.arange(count, dtype=np.float64) * config.sample_dt
    position = rng.uniform(-5.0, 5.0, size=3)
    velocity = rng.uniform(-2.0, 2.0, size=3)
    acceleration = (
        rng.uniform(-0.4, 0.4, size=3)
        if regime == "slow_acceleration"
        else np.zeros(3)
    )
    positions = [position.copy()]
    jerks = _jerk_schedule(
        regime,
        count,
        rng,
        config.history_jerk_bound,
        config.robust_jerk_bound,
    )
    for jerk in jerks:
        position, velocity, acceleration = _advance(
            position, velocity, acceleration, jerk, config.sample_dt
        )
        positions.append(position.copy())
    measurements = np.asarray(positions) + rng.uniform(
        -config.sensor_error, config.sensor_error, size=(count, 3)
    )
    future_jerk = np.zeros(3)
    if regime in {"sudden_velocity_change", "sudden_direction_change", "stop_and_go"}:
        future_jerk = rng.uniform(
            -config.robust_jerk_bound,
            config.robust_jerk_bound,
            size=3,
        )
    future_position, _, _ = _advance(
        position,
        velocity,
        acceleration,
        future_jerk,
        config.prediction_horizon,
    )
    return {
        "times": times,
        "measurements": measurements,
        "position": position,
        "velocity": velocity,
        "acceleration": acceleration,
        "future_position": future_position,
        "maximum_history_jerk": np.max(np.abs(jerks), axis=0),
    }


def _no_history_bounds(case: dict[str, np.ndarray], config: AdaptiveTubeAblationConfig) -> SetMembershipBounds:
    current = case["measurements"][-1]
    return SetMembershipBounds(
        feasible=True,
        position=np.stack((current - config.sensor_error, current + config.sensor_error), axis=1),
        velocity=np.tile(np.array([-config.velocity_bound, config.velocity_bound]), (3, 1)),
        acceleration=np.tile(
            np.array([-config.acceleration_bound, config.acceleration_bound]), (3, 1)
        ),
        window_length=1,
    )


def _evaluate_bounds(
    bounds: SetMembershipBounds,
    case: dict[str, np.ndarray],
    config: AdaptiveTubeAblationConfig,
) -> dict[str, float]:
    if not bounds.feasible:
        return {
            "feasible": 0.0,
            "current_state_contained": 0.0,
            "velocity_absolute_error": float("nan"),
            "acceleration_absolute_error": float("nan"),
            "tube_radius": float("nan"),
            "formula_endpoint_contained": float("nan"),
            "contained": 0.0,
        }
    horizon = config.prediction_horizon
    predicted = (
        bounds.center_position
        + horizon * bounds.center_velocity
        + 0.5 * horizon**2 * bounds.center_acceleration
    )
    error = float(np.linalg.norm(case["future_position"] - predicted))
    current_state_contained = bool(
        np.all(case["position"] >= bounds.position[:, 0] - 1e-8)
        and np.all(case["position"] <= bounds.position[:, 1] + 1e-8)
        and np.all(case["velocity"] >= bounds.velocity[:, 0] - 1e-8)
        and np.all(case["velocity"] <= bounds.velocity[:, 1] + 1e-8)
        and np.all(case["acceleration"] >= bounds.acceleration[:, 0] - 1e-8)
        and np.all(case["acceleration"] <= bounds.acceleration[:, 1] + 1e-8)
    )
    component_formula_radius = (
        bounds.radius_position
        + horizon * bounds.radius_velocity
        + 0.5 * horizon**2 * bounds.radius_acceleration
        + horizon**3 / 6.0 * config.robust_jerk_bound
    )
    formula_radius = float(np.linalg.norm(component_formula_radius))
    formula_endpoint_contained = float(error <= formula_radius + 1e-8)
    if current_state_contained:
        _, certified_radius = future_position_error_tube(
            np.array([config.prediction_horizon]),
            bounds,
            future_jerk_bound=config.robust_jerk_bound,
            true_state_containment_verified=True,
        )
        tube_radius = float(certified_radius[0])
        contained = float(error <= certified_radius[0] + 1e-8)
    else:
        tube_radius = float("nan")
        contained = 0.0
    return {
        "feasible": 1.0,
        "current_state_contained": float(current_state_contained),
        "velocity_absolute_error": float(
            np.mean(np.abs(bounds.center_velocity - case["velocity"]))
        ),
        "acceleration_absolute_error": float(
            np.mean(np.abs(bounds.center_acceleration - case["acceleration"]))
        ),
        "tube_radius": tube_radius,
        "formula_endpoint_contained": formula_endpoint_contained,
        "contained": contained,
    }


def _aggregate(records: list[dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in records[0]:
        values = np.asarray([record[key] for record in records], dtype=np.float64)
        valid = np.isfinite(values)
        result[key] = float(np.mean(values[valid])) if np.any(valid) else float("nan")
    return result


def run_adaptive_tube_ablation(config: AdaptiveTubeAblationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(config.seed)
    regimes = (
        "constant_velocity",
        "slow_acceleration",
        "sudden_velocity_change",
        "sudden_direction_change",
        "stop_and_go",
    )
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "future_tube_uses_robust_jerk_bound": True,
        "history_informed_current_set_requires_true_history_jerk_assumption": True,
        "regimes": {},
    }
    for regime in regimes:
        records = {name: [] for name in ("no_history", "fixed_l2", "fixed_l16", "adaptive")}
        adaptive_windows: list[int] = []
        robust_resets = 0
        assumption_satisfied = 0
        robust_model_certified = 0
        latencies: list[float] = []
        for _ in range(config.cases_per_regime):
            case = _simulate_case(regime, config, rng)
            records["no_history"].append(_evaluate_bounds(_no_history_bounds(case, config), case, config))
            for name, length in (("fixed_l2", 2), ("fixed_l16", config.history_length)):
                bounds = jerk_bounded_set_membership(
                    case["times"][-length:],
                    case["measurements"][-length:],
                    sensor_error=config.sensor_error,
                    velocity_bound=config.velocity_bound,
                    acceleration_bound=config.acceleration_bound,
                    jerk_bound=config.history_jerk_bound,
                )
                records[name].append(_evaluate_bounds(bounds, case, config))
            started = time.perf_counter()
            adaptive = adaptive_history_set_membership(
                case["times"],
                case["measurements"],
                sensor_error=config.sensor_error,
                velocity_bound=config.velocity_bound,
                acceleration_bound=config.acceleration_bound,
                history_jerk_bound=config.history_jerk_bound,
                robust_jerk_bound=config.robust_jerk_bound,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            adaptive_windows.append(adaptive.bounds.window_length)
            robust_resets += int(adaptive.mode is HistoryTubeMode.ROBUST_BASE)
            history_assumption = bool(
                np.all(case["maximum_history_jerk"] <= config.history_jerk_bound + 1e-12)
            )
            assumption_satisfied += int(history_assumption)
            robust_model_certified += int(
                adaptive.certified_under_declared_robust_model
            )
            adaptive_record = _evaluate_bounds(adaptive.bounds, case, config)
            adaptive_record["history_motion_bound_assumption_satisfied"] = float(
                history_assumption
            )
            adaptive_record["certified_under_declared_robust_model"] = float(
                adaptive.certified_under_declared_robust_model
            )
            records["adaptive"].append(adaptive_record)
        summary["regimes"][regime] = {
            "methods": {name: _aggregate(method_records) for name, method_records in records.items()},
            "adaptive": {
                "mean_window": float(np.mean(adaptive_windows)),
                "robust_reset_rate": float(robust_resets / config.cases_per_regime),
                "history_motion_bound_assumption_rate": float(
                    assumption_satisfied / config.cases_per_regime
                ),
                "certified_under_declared_robust_model_rate": float(
                    robust_model_certified / config.cases_per_regime
                ),
                "latency_ms_mean": float(np.mean(latencies)),
                "latency_ms_p95": float(np.percentile(latencies, 95)),
                "latency_ms_p99": float(np.percentile(latencies, 99)),
                "deadline_miss_rate_50ms": float(np.mean(np.asarray(latencies) > 50.0)),
            },
        }
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
