from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class CounterexampleConfig:
    seed: int = 20260819
    random_states: int = 100_000
    abrupt_states: int = 100_000
    dt: float = 0.1
    sensor_bound: float = 0.10
    jerk_residual_bound: float = 12.0
    base_velocity_bound: float = 12.0
    base_acceleration_bound: float = 8.0
    k1: float = 1.0
    k2: float = 1.0
    tracking_trials_per_count: int = 100
    tracking_frames: int = 20
    output_dir: str = "artifacts/memory_counterexamples_200k"


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _observer_matrices(dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    identity = np.eye(3)
    transition = np.block(
        [
            [identity, dt * identity, 0.5 * dt**2 * identity],
            [np.zeros((3, 3)), identity, dt * identity],
            [np.zeros((3, 3)), np.zeros((3, 3)), identity],
        ]
    )
    jerk_map = np.concatenate(
        (
            (dt**3 / 6.0) * identity,
            (dt**2 / 2.0) * identity,
            dt * identity,
        ),
        axis=0,
    )
    observation = np.concatenate((identity, np.zeros((3, 6))), axis=1)
    gain = np.concatenate((0.85 * identity, 2.5 * identity, 5.0 * identity), axis=0)
    return transition, jerk_map, observation, gain


def _valid_observer_trials(
    rng: np.random.Generator,
    config: CounterexampleConfig,
) -> dict[str, float | int]:
    count = config.random_states
    transition, jerk_map, observation, gain = _observer_matrices(config.dt)
    center = rng.normal(scale=10.0, size=(count, 9))
    radius = rng.uniform(0.01, 2.0, size=(count, 9))
    error = rng.uniform(-1.0, 1.0, size=(count, 9)) * radius
    nominal_jerk = rng.uniform(-5.0, 5.0, size=(count, 3))
    residual = rng.uniform(
        -config.jerk_residual_bound,
        config.jerk_residual_bound,
        size=(count, 3),
    )
    truth = center + error
    true_next = truth @ transition.T + (nominal_jerk + residual) @ jerk_map.T
    predicted = center @ transition.T + nominal_jerk @ jerk_map.T
    predicted_radius = radius @ np.abs(transition).T
    predicted_radius += config.jerk_residual_bound * np.sum(np.abs(jerk_map), axis=1)
    prediction_contained = np.all(
        np.abs(true_next - predicted) <= predicted_radius + 1e-12,
        axis=1,
    )
    noise = rng.uniform(-config.sensor_bound, config.sensor_bound, size=(count, 3))
    measurement = true_next[:, :3] + noise
    innovation = measurement - predicted[:, :3]
    corrected = predicted + innovation @ gain.T
    corrected_radius = predicted_radius @ np.abs(np.eye(9) - gain @ observation).T
    corrected_radius += config.sensor_bound * np.sum(np.abs(gain), axis=1)
    correction_contained = np.all(
        np.abs(true_next - corrected) <= corrected_radius + 1e-12,
        axis=1,
    )
    return {
        "states": count,
        "prediction_containment_rate": float(np.mean(prediction_contained)),
        "correction_containment_rate": float(np.mean(correction_contained)),
        "prediction_underbound_count": int(np.sum(~prediction_contained)),
        "correction_underbound_count": int(np.sum(~correction_contained)),
    }


def _abrupt_trials(
    rng: np.random.Generator,
    config: CounterexampleConfig,
) -> dict[str, float | int]:
    count = config.abrupt_states
    transition, jerk_map, _, _ = _observer_matrices(config.dt)
    center = rng.normal(scale=10.0, size=(count, 9))
    radius = rng.uniform(0.01, 0.25, size=(count, 9))
    truth = center + rng.uniform(-1.0, 1.0, size=(count, 9)) * radius
    nominal_jerk = rng.uniform(-5.0, 5.0, size=(count, 3))
    magnitude = rng.uniform(1.05, 4.0, size=(count, 3)) * config.jerk_residual_bound
    signs = rng.choice((-1.0, 1.0), size=(count, 3))
    residual = magnitude * signs
    true_next = truth @ transition.T + (nominal_jerk + residual) @ jerk_map.T
    predicted = center @ transition.T + nominal_jerk @ jerk_map.T
    predicted_radius = radius @ np.abs(transition).T
    predicted_radius += config.jerk_residual_bound * np.sum(np.abs(jerk_map), axis=1)
    contained_before_reset = np.all(
        np.abs(true_next - predicted) <= predicted_radius + 1e-12,
        axis=1,
    )
    noise = rng.uniform(-config.sensor_bound, config.sensor_bound, size=(count, 3))
    measurement = true_next[:, :3] + noise
    reset_center = np.concatenate(
        (measurement, np.zeros((count, 6))),
        axis=1,
    )
    reset_radius = np.broadcast_to(
        np.array(
            [config.sensor_bound] * 3
            + [config.base_velocity_bound] * 3
            + [config.base_acceleration_bound] * 3
        ),
        (count, 9),
    )
    base_assumption = np.all(
        np.abs(true_next[:, 3:6]) <= config.base_velocity_bound,
        axis=1,
    ) & np.all(
        np.abs(true_next[:, 6:9]) <= config.base_acceleration_bound,
        axis=1,
    )
    reset_contained = np.all(
        np.abs(true_next - reset_center) <= reset_radius + 1e-12,
        axis=1,
    )
    return {
        "states": count,
        "residual_bound_violated_rate": 1.0,
        "pre_reset_containment_rate": float(np.mean(contained_before_reset)),
        "pre_reset_underbound_count": int(np.sum(~contained_before_reset)),
        "base_assumption_satisfied_rate": float(np.mean(base_assumption)),
        "reset_containment_rate_unconditional": float(np.mean(reset_contained)),
        "reset_containment_rate_given_base_assumption": float(
            np.mean(reset_contained[base_assumption]) if np.any(base_assumption) else np.nan
        ),
    }


def _robust_hocbf_trials(
    rng: np.random.Generator,
    config: CounterexampleConfig,
) -> dict[str, float | int]:
    count = config.random_states
    direction = rng.normal(size=(count, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    position_radius = rng.uniform(0.01, 2.0, size=(count, 3))
    velocity_radius = rng.uniform(0.01, 1.0, size=(count, 3))
    acceleration_radius = rng.uniform(0.01, 2.0, size=(count, 3))
    nominal_h = rng.uniform(-2.0, 20.0, size=count)
    nominal_h_dot = rng.uniform(-10.0, 10.0, size=count)
    nominal_obstacle_acceleration = rng.uniform(-4.0, 4.0, size=(count, 3))
    support = (
        np.sum(np.abs(direction) * acceleration_radius, axis=1)
        + (config.k1 + config.k2)
        * np.sum(np.abs(direction) * velocity_radius, axis=1)
        + config.k1
        * config.k2
        * np.sum(np.abs(direction) * position_radius, axis=1)
    )
    nominal_drift = (
        -np.sum(direction * nominal_obstacle_acceleration, axis=1)
        + (config.k1 + config.k2) * nominal_h_dot
        + config.k1 * config.k2 * nominal_h
    )
    required_projection = -nominal_drift + support
    action = required_projection[:, None] * direction
    position_error = rng.uniform(-1.0, 1.0, size=(count, 3)) * position_radius
    velocity_error = rng.uniform(-1.0, 1.0, size=(count, 3)) * velocity_radius
    acceleration_error = rng.uniform(-1.0, 1.0, size=(count, 3)) * acceleration_radius
    true_h = nominal_h - np.sum(direction * position_error, axis=1)
    true_h_dot = nominal_h_dot - np.sum(direction * velocity_error, axis=1)
    true_obstacle_acceleration = nominal_obstacle_acceleration + acceleration_error
    psi2 = (
        np.sum(direction * (action - true_obstacle_acceleration), axis=1)
        + (config.k1 + config.k2) * true_h_dot
        + config.k1 * config.k2 * true_h
    )
    return {
        "states": count,
        "minimum_true_psi2": float(np.min(psi2)),
        "violations": int(np.sum(psi2 < -1e-10)),
        "violation_rate": float(np.mean(psi2 < -1e-10)),
    }


def _dropout_growth(config: CounterexampleConfig) -> dict[str, list[float]]:
    transition, jerk_map, _, _ = _observer_matrices(config.dt)
    radius = np.array([config.sensor_bound] * 3 + [0.5] * 3 + [0.5] * 3)
    results: dict[str, list[float]] = {"0": radius[:3].tolist()}
    for frame in range(1, 6):
        radius = np.abs(transition) @ radius
        radius += config.jerk_residual_bound * np.sum(np.abs(jerk_map), axis=1)
        if frame in {1, 2, 3, 5}:
            results[str(frame)] = radius[:3].tolist()
    return results


def _association_counterexample(config: CounterexampleConfig) -> dict[str, object]:
    predicted_a = np.array([-0.04, 0.0, 0.0])
    predicted_b = np.array([0.04, 0.0, 0.0])
    radius = np.array([0.10, 0.10, 0.10])
    swapped_measurement_for_a = predicted_b.copy()
    innovation = swapped_measurement_for_a - predicted_a
    passes_gate = bool(
        np.all(np.abs(innovation) <= radius + config.sensor_bound)
    )
    return {
        "crossing_track_separation": float(np.linalg.norm(predicted_a - predicted_b)),
        "swapped_measurement_innovation": innovation.tolist(),
        "swap_passes_single_track_innovation_gate": passes_gate,
        "implication": (
            "innovation alone cannot certify identity; ambiguous association must inflate, union, or reset"
        ),
    }


def _sensor_delay_counterexample(config: CounterexampleConfig) -> dict[str, object]:
    velocity = np.array([25.0, -8.0, 3.0])
    delay_seconds = 3.0 * config.dt
    stale_position_error = np.abs(velocity) * delay_seconds
    naive_radius = np.full(3, config.sensor_bound)
    contained_without_delay_inflation = bool(
        np.all(stale_position_error <= naive_radius)
    )
    required_radius = naive_radius + stale_position_error
    return {
        "velocity": velocity.tolist(),
        "delay_seconds": delay_seconds,
        "stale_position_error": stale_position_error.tolist(),
        "naive_sensor_radius": naive_radius.tolist(),
        "contained_without_delay_inflation": contained_without_delay_inflation,
        "minimum_delay_aware_radius": required_radius.tolist(),
        "implication": (
            "a delayed measurement is not a current-state interval; certification "
            "must timestamp-propagate it or inflate by a valid motion envelope"
        ),
    }


def _tracking_scaling_trials(
    rng: np.random.Generator,
    config: CounterexampleConfig,
) -> dict[str, object]:
    """Audit explicit per-track memory and Hungarian association scaling."""

    results: dict[str, object] = {}
    numerical_payload_bytes_per_track = 22 * np.dtype(np.float64).itemsize
    for obstacle_count in (2, 4, 8, 16, 32):
        latencies = []
        correct = 0
        assignments = 0
        ambiguous = 0
        for _ in range(config.tracking_trials_per_count):
            positions = rng.uniform(-20.0, 20.0, size=(obstacle_count, 3))
            velocities = rng.uniform(-2.0, 2.0, size=(obstacle_count, 3))
            for _ in range(config.tracking_frames):
                predicted = positions + velocities * config.dt
                truth = predicted.copy()
                measurements = truth + rng.uniform(
                    -config.sensor_bound,
                    config.sensor_bound,
                    size=truth.shape,
                )
                permutation = rng.permutation(obstacle_count)
                shuffled = measurements[permutation]
                cost = np.linalg.norm(
                    predicted[:, None, :] - shuffled[None, :, :],
                    axis=2,
                )
                started = perf_counter()
                rows, columns = linear_sum_assignment(cost)
                latencies.append(perf_counter() - started)
                assigned_identity = permutation[columns]
                correct += int(np.sum(assigned_identity == rows))
                assignments += obstacle_count
                if obstacle_count > 1:
                    nearest_two = np.partition(cost, 1, axis=1)[:, :2]
                    ambiguity_gap = nearest_two[:, 1] - nearest_two[:, 0]
                    ambiguous += int(
                        np.sum(ambiguity_gap <= 2.0 * config.sensor_bound)
                    )
                positions = truth
        latency_array = np.asarray(latencies, dtype=np.float64)
        results[str(obstacle_count)] = {
            "trials": config.tracking_trials_per_count,
            "frames_per_trial": config.tracking_frames,
            "assignments": assignments,
            "identity_accuracy": correct / max(assignments, 1),
            "ambiguous_track_fraction": ambiguous / max(assignments, 1),
            "hungarian_latency_p50_ms": float(
                np.quantile(latency_array, 0.50) * 1000.0
            ),
            "hungarian_latency_p99_ms": float(
                np.quantile(latency_array, 0.99) * 1000.0
            ),
            "explicit_numeric_memory_bytes": (
                obstacle_count * numerical_payload_bytes_per_track
            ),
        }
    return {
        "association_method": "Hungarian assignment on predicted-position distance",
        "memory_semantics": (
            "22 float64 values per track: p/v/a centers, p/v/a radii, "
            "innovation vector, and one confidence score; excludes Python-object overhead"
        ),
        "counts": results,
        "nonclaim": (
            "High random-scene identity accuracy is not an association certificate; "
            "the explicit crossing counterexample remains decisive."
        ),
    }


def run(config: CounterexampleConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    rng = np.random.default_rng(config.seed)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "git_sha": _git_sha(),
        "source_sha256": _source_hash(),
        "valid_assumption_trials": _valid_observer_trials(rng, config),
        "abrupt_assumption_violation_trials": _abrupt_trials(rng, config),
        "robust_hocbf_trials": _robust_hocbf_trials(rng, config),
        "dropout_position_radius": _dropout_growth(config),
        "association_counterexample": _association_counterexample(config),
        "sensor_delay_counterexample": _sensor_delay_counterexample(config),
        "multi_obstacle_tracking": _tracking_scaling_trials(rng, config),
        "formal_500k": False,
    }
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
