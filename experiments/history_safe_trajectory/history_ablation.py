from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class HistoryAblationConfig:
    seed: int = 20260820
    cases: int = 10_000
    lidar_dt: float = 0.1
    range_noise_std: float = 0.10
    dropout_probability: float = 0.05
    contact_distance: float = 2.0
    hazard_horizon_seconds: float = 3.0
    output_dir: str = "artifacts/history_motion_ablation_10k"


def _linear_slope(times: np.ndarray, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    count = np.sum(valid, axis=1)
    mean_time = np.sum(valid * times[None, :], axis=1) / np.maximum(count, 1)
    mean_value = np.sum(valid * values, axis=1) / np.maximum(count, 1)
    centered_time = times[None, :] - mean_time[:, None]
    centered_value = values - mean_value[:, None]
    numerator = np.sum(valid * centered_time * centered_value, axis=1)
    denominator = np.sum(valid * centered_time**2, axis=1)
    slope = numerator / np.maximum(denominator, 1e-15)
    slope[count < 2] = np.nan
    return slope


def _metrics(
    estimate: np.ndarray,
    truth: np.ndarray,
    range_now: np.ndarray,
    ego_speed: np.ndarray,
    config: HistoryAblationConfig,
) -> dict[str, float]:
    valid = np.isfinite(estimate)
    error = estimate[valid] - truth[valid]
    true_closing = np.maximum(0.0, ego_speed - truth)
    estimated_closing = np.maximum(0.0, ego_speed - estimate)
    true_ttc = (range_now - config.contact_distance) / np.maximum(true_closing, 1e-12)
    estimated_ttc = (range_now - config.contact_distance) / np.maximum(
        estimated_closing, 1e-12
    )
    true_hazard = true_ttc <= config.hazard_horizon_seconds
    estimated_hazard = estimated_ttc <= config.hazard_horizon_seconds
    true_positive = np.sum(valid & true_hazard & estimated_hazard)
    false_negative = np.sum(valid & true_hazard & ~estimated_hazard)
    false_positive = np.sum(valid & ~true_hazard & estimated_hazard)
    return {
        "valid_fraction": float(np.mean(valid)),
        "velocity_mae": float(np.mean(np.abs(error))) if error.size else float("nan"),
        "velocity_rmse": float(np.sqrt(np.mean(error**2))) if error.size else float("nan"),
        "velocity_bias": float(np.mean(error)) if error.size else float("nan"),
        "hazard_recall": float(true_positive / max(true_positive + false_negative, 1)),
        "hazard_precision": float(true_positive / max(true_positive + false_positive, 1)),
    }


def _generate_ranges(
    rng: np.random.Generator,
    *,
    cases: int,
    length: int,
    dt: float,
    noise_std: float,
    dropout_probability: float,
    abrupt: bool,
) -> tuple[np.ndarray, ...]:
    times = np.arange(-(length - 1), 1, dtype=np.float64) * dt
    ego_speed = rng.uniform(-5.0, 5.0, size=cases)
    current_obstacle_speed = rng.uniform(-3.0, 3.0, size=cases)
    range_now = rng.uniform(5.0, 50.0, size=cases)
    if abrupt:
        old_speed = rng.uniform(-3.0, 3.0, size=cases)
        change_time = -2.0 * dt
        obstacle_displacement = np.empty((cases, length), dtype=np.float64)
        recent = times >= change_time
        obstacle_displacement[:, recent] = current_obstacle_speed[:, None] * times[recent]
        obstacle_displacement[:, ~recent] = (
            current_obstacle_speed[:, None] * change_time
            + old_speed[:, None] * (times[~recent] - change_time)
        )
    else:
        obstacle_displacement = current_obstacle_speed[:, None] * times[None, :]
    ego_displacement = ego_speed[:, None] * times[None, :]
    ranges = (
        range_now[:, None]
        + obstacle_displacement
        - ego_displacement
        + rng.normal(0.0, noise_std, size=(cases, length))
    )
    valid = rng.random(size=(cases, length)) >= dropout_probability
    valid[:, -1] = True
    return times, ranges, valid, ego_displacement, ego_speed, current_obstacle_speed, range_now


def run_history_ablation(config: HistoryAblationConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(config.seed)
    summary: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "causal_frames_only": True,
        "future_information_used": False,
        "regimes": {},
    }
    for regime, abrupt in (("constant_velocity", False), ("abrupt_velocity_change", True)):
        regime_result: dict[str, object] = {}
        for length in (2, 4, 8, 16):
            (
                times,
                ranges,
                valid,
                ego_displacement,
                ego_speed,
                truth,
                range_now,
            ) = _generate_ranges(
                rng,
                cases=config.cases,
                length=length,
                dt=config.lidar_dt,
                noise_std=config.range_noise_std,
                dropout_probability=config.dropout_probability,
                abrupt=abrupt,
            )
            current_only = np.zeros(config.cases, dtype=np.float64)
            raw_history = _linear_slope(times, ranges, valid)
            compensated_history = _linear_slope(
                times, ranges + ego_displacement, valid
            )
            regime_result[str(length)] = {
                "current_state_zero_velocity_prior": _metrics(
                    current_only, truth, range_now, ego_speed, config
                ),
                "raw_lidar_history": _metrics(
                    raw_history, truth, range_now, ego_speed, config
                ),
                "history_plus_physical_ego_compensation": _metrics(
                    compensated_history, truth, range_now, ego_speed, config
                ),
            }
        summary["regimes"][regime] = regime_result
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
