from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.memory_safety.closed_loop import _apply_velocity_limits
from experiments.uav_safety_filter.benchmark import (
    GOAL_RADIUS,
    HORIZONTAL_A_MAX,
    HORIZONTAL_V_MAX,
    SAFETY_DT,
    UAV_RADIUS,
    VERTICAL_A_MAX,
    VERTICAL_V_MAX,
    WORLD_SIZE,
    normalized_action_to_acceleration,
    sac_observation,
)
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel
from review_bundle.safety.collision.filter import (
    SafetyFilterConfig,
    SafetyFilterMethod,
    UAVSafetyActionFilter,
)
from review_bundle.safety.collision.hocbf import HOCBFConfig, SphericalObstacle
from scripts.train_uav_energy_mc import load_frozen_sac


SCENARIO_FAMILIES = (
    "static",
    "constant_velocity",
    "accelerating",
    "turning",
    "abrupt_change",
    "dropout",
    "dense_multi_obstacle",
    "boundary_interaction",
)
ROUTE_CLASSES = (
    "return_nominal",
    "continue_left",
    "continue_right",
    "climb",
    "descend",
    "brake",
)
FEATURE_DIM = 23
ENERGY_STATE_DIM = 7
CONTRACTION_HISTORY = 16
CERTIFIED_CLEARANCE_MARGIN = 2.0


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GroundedDatasetConfig:
    seed: int = 20260820
    minimum_controlled_transitions: int = 100_000
    max_scenarios: int = 400
    max_steps: int = 800
    sensor_period_steps: int = 2
    sensor_error_bound: float = 0.10
    dropout_probability: float = 0.05
    history_seconds: float = 2.0
    contraction_stride: int = 10
    obstacle_radius: float = 20.0
    sac_checkpoint: str = (
        "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
        "phase1_navigation/checkpoint_transition_500000.zip"
    )
    output_dir: str = "artifacts/grounded_memory_real_dataset"
    device: str = "cpu"
    require_hard_filter_valid: bool = True

    def __post_init__(self) -> None:
        if self.minimum_controlled_transitions < 100_000:
            raise ValueError("real diagnostic dataset must contain at least 100k transitions")
        if self.max_scenarios <= 0 or self.max_steps <= 0:
            raise ValueError("scenario and step budgets must be positive")
        if self.sensor_period_steps <= 0 or self.contraction_stride <= 0:
            raise ValueError("sensor and contraction periods must be positive")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must lie in [0, 1)")


@dataclass(frozen=True)
class DynamicObstacleSpec:
    identifier: str
    center: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    radius: float
    family: str
    phase: float


@dataclass(frozen=True)
class GroundedScenario:
    identifier: str
    family: str
    start: np.ndarray
    goal: np.ndarray
    obstacles: tuple[DynamicObstacleSpec, ...]


def _orthogonal(direction: np.ndarray) -> np.ndarray:
    lateral = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
    return lateral / max(float(np.linalg.norm(lateral)), 1e-12)


def make_scenario(index: int, seed: int, obstacle_radius: float) -> GroundedScenario:
    rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
    family = SCENARIO_FAMILIES[index % len(SCENARIO_FAMILIES)]
    angle = rng.uniform(-np.pi, np.pi)
    direction = np.array([np.cos(angle), np.sin(angle), rng.uniform(-0.08, 0.08)])
    direction /= np.linalg.norm(direction)
    lateral = _orthogonal(direction)
    distance = float(rng.uniform(320.0, 620.0))
    if family == "boundary_interaction":
        boundary_side = rng.choice((-1.0, 1.0))
        start_x = 80.0 if boundary_side > 0.0 else WORLD_SIZE[0] - 80.0
        inward = boundary_side * 0.20
        along = rng.choice((-1.0, 1.0)) * np.sqrt(1.0 - inward**2)
        direction = np.array([inward, along, rng.uniform(-0.03, 0.03)])
        direction /= np.linalg.norm(direction)
        start_y = (
            rng.uniform(300.0, 900.0)
            if along > 0.0
            else rng.uniform(3100.0, 3700.0)
        )
        start = np.array([start_x, start_y, rng.uniform(80.0, 320.0)])
        goal = start + direction * distance
    else:
        midpoint = np.array(
            [
                rng.uniform(800.0, 3200.0),
                rng.uniform(800.0, 3200.0),
                rng.uniform(90.0, 310.0),
            ]
        )
        start = midpoint - 0.5 * distance * direction
        goal = midpoint + 0.5 * distance * direction
    midpoint = 0.5 * (start + goal)
    if family in {"static", "dense_multi_obstacle", "boundary_interaction"}:
        main_velocity = np.zeros(3)
        crossing_offset = np.zeros(3)
    else:
        lateral_speed = rng.uniform(3.0, 5.0) * rng.choice((-1.0, 1.0))
        main_velocity = lateral_speed * lateral
        crossing_offset = (
            rng.choice((-1.0, 1.0))
            * (obstacle_radius + CERTIFIED_CLEARANCE_MARGIN + rng.uniform(8.0, 16.0))
            * lateral
        )
    main_center = midpoint + crossing_offset - main_velocity * rng.uniform(8.0, 13.0)
    obstacles = [
        DynamicObstacleSpec(
            identifier=f"{family}_{index}_main",
            center=main_center.astype(np.float64),
            velocity=main_velocity.astype(np.float64),
            acceleration=np.zeros(3),
            radius=obstacle_radius,
            family=family,
            phase=float(rng.uniform(0.0, 2.0 * np.pi)),
        )
    ]
    if family == "dense_multi_obstacle":
        for obstacle_index, (forward, side, vertical) in enumerate(
            ((-75.0, 18.0, 0.0), (35.0, -20.0, 8.0), (95.0, 16.0, -6.0)),
            start=1,
        ):
            obstacles.append(
                DynamicObstacleSpec(
                    identifier=f"{family}_{index}_{obstacle_index}",
                    center=(midpoint + forward * direction + side * lateral + np.array([0.0, 0.0, vertical])),
                    velocity=np.zeros(3),
                    acceleration=np.zeros(3),
                    radius=0.85 * obstacle_radius,
                    family="static",
                    phase=0.0,
                )
            )
    return GroundedScenario(
        identifier=f"{family}_{index:04d}",
        family=family,
        start=start.astype(np.float64),
        goal=goal.astype(np.float64),
        obstacles=tuple(obstacles),
    )


def _advance_obstacle(
    spec: DynamicObstacleSpec,
    center: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jerk = np.zeros(3, dtype=np.float64)
    if spec.family == "accelerating":
        axis = _orthogonal(velocity if np.linalg.norm(velocity[:2]) > 1e-9 else np.array([1.0, 0.0, 0.0]))
        jerk = 0.4 * np.sin(0.035 * step + spec.phase) * axis
    elif spec.family == "turning":
        speed = max(float(np.linalg.norm(velocity[:2])), 1e-9)
        tangent = velocity / speed
        desired_acceleration = 0.2 * _orthogonal(tangent)
        jerk = np.clip((desired_acceleration - acceleration) / SAFETY_DT, -2.0, 2.0)
    elif spec.family == "abrupt_change":
        if 180 <= step < 200:
            jerk = 2.0 * _orthogonal(velocity)
        elif 200 <= step < 220:
            jerk = -2.0 * _orthogonal(velocity)
    next_acceleration = np.clip(acceleration + jerk * SAFETY_DT, -6.0, 6.0)
    realized_jerk = (next_acceleration - acceleration) / SAFETY_DT
    next_center = (
        center
        + velocity * SAFETY_DT
        + 0.5 * acceleration * SAFETY_DT**2
        + realized_jerk * SAFETY_DT**3 / 6.0
    )
    next_velocity = velocity + acceleration * SAFETY_DT + 0.5 * realized_jerk * SAFETY_DT**2
    return next_center, next_velocity, next_acceleration, realized_jerk


def _energy_state(position: np.ndarray, velocity: np.ndarray, goal: np.ndarray) -> np.ndarray:
    delta = goal - position
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-12)
    return np.asarray(
        [
            velocity[0] / HORIZONTAL_V_MAX,
            velocity[1] / HORIZONTAL_V_MAX,
            velocity[2] / VERTICAL_V_MAX,
            *direction,
            distance / float(np.linalg.norm(WORLD_SIZE)),
        ],
        dtype=np.float32,
    )


def _transition_feature(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    obstacle_center: np.ndarray,
    obstacle_velocity: np.ndarray,
    nominal: np.ndarray,
    executed: np.ndarray,
    intervention: float,
    progress: float,
    measurement_valid: bool,
    boundary_contact: bool,
) -> np.ndarray:
    observation = sac_observation(position, velocity, goal).astype(np.float64)
    relative = (obstacle_center - position) / 250.0
    obstacle_velocity_feature = obstacle_velocity / 30.0
    nominal_feature = nominal / np.array([HORIZONTAL_A_MAX, HORIZONTAL_A_MAX, VERTICAL_A_MAX])
    executed_feature = executed / np.array([HORIZONTAL_A_MAX, HORIZONTAL_A_MAX, VERTICAL_A_MAX])
    feature = np.concatenate(
        (
            observation,
            relative,
            obstacle_velocity_feature,
            nominal_feature,
            executed_feature,
            np.asarray(
                [
                    intervention / max(float(np.hypot(HORIZONTAL_A_MAX, VERTICAL_A_MAX)), 1e-12),
                    progress / max(HORIZONTAL_V_MAX * SAFETY_DT, 1e-12),
                    float(measurement_valid),
                    float(boundary_contact),
                ]
            ),
        )
    )
    if feature.shape != (FEATURE_DIM,):
        raise AssertionError(f"unexpected feature shape {feature.shape}")
    return feature.astype(np.float32)


def _route_labels(
    positions: np.ndarray,
    velocities: np.ndarray,
    interventions: np.ndarray,
    goal: np.ndarray,
    obstacle_positions: np.ndarray,
    horizon: int = 40,
) -> dict[str, np.ndarray]:
    count = positions.shape[0]
    route_class = np.zeros(count, dtype=np.int64)
    future_path = np.zeros(count, dtype=np.float32)
    detour_ratio = np.ones(count, dtype=np.float32)
    future_progress = np.zeros(count, dtype=np.float32)
    future_interventions = np.zeros(count, dtype=np.float32)
    freeze_risk = np.zeros(count, dtype=np.float32)
    curvature = np.zeros(count, dtype=np.float32)
    time_to_clear = np.full(count, horizon * SAFETY_DT, dtype=np.float32)
    previous_side = np.zeros(count, dtype=np.int8)
    future_side = np.zeros(count, dtype=np.int8)
    for index in range(count):
        end = min(count - 1, index + horizon)
        segment = positions[index : end + 1]
        if segment.shape[0] <= 1:
            continue
        displacement = segment[-1] - segment[0]
        goal_delta = goal - segment[0]
        goal_direction = goal_delta / max(float(np.linalg.norm(goal_delta)), 1e-12)
        lateral = _orthogonal(goal_direction)
        straight_progress = float(displacement @ goal_direction)
        lateral_offset = float(displacement @ lateral)
        vertical_offset = float(displacement[2] - straight_progress * goal_direction[2])
        speed = np.linalg.norm(velocities[index : end + 1], axis=1)
        path = float(np.sum(np.linalg.norm(np.diff(segment, axis=0), axis=1)))
        future_path[index] = path
        future_progress[index] = straight_progress
        detour_ratio[index] = path / max(straight_progress, 1e-3)
        future_interventions[index] = float(np.sum(interventions[index : end + 1] > 1e-6))
        freeze_risk[index] = float(np.any(np.convolve((speed < 0.2).astype(int), np.ones(5, dtype=int), mode="valid") >= 5)) if speed.size >= 5 else float(np.all(speed < 0.2))
        if segment.shape[0] >= 3:
            directions = np.diff(segment, axis=0)
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            unit = directions / np.maximum(norms, 1e-12)
            curvature[index] = float(np.mean(np.linalg.norm(np.diff(unit, axis=0), axis=1)))
        clearances = np.linalg.norm(segment - obstacle_positions[index : end + 1], axis=1)
        cleared = np.flatnonzero(clearances > 2.5 * 20.0)
        if cleared.size:
            time_to_clear[index] = float(cleared[0] * SAFETY_DT)
        side_threshold = 1.0
        future_side[index] = 1 if lateral_offset > side_threshold else (-1 if lateral_offset < -side_threshold else 0)
        if index > 0:
            prior_displacement = positions[index] - positions[max(0, index - min(horizon, index))]
            prior_lateral = float(prior_displacement @ lateral)
            previous_side[index] = 1 if prior_lateral > side_threshold else (-1 if prior_lateral < -side_threshold else 0)
        if abs(vertical_offset) > max(abs(lateral_offset), 1.5):
            route_class[index] = 3 if vertical_offset > 0.0 else 4
        elif abs(lateral_offset) > 1.5:
            route_class[index] = 1 if lateral_offset > 0.0 else 2
        elif straight_progress < 0.25 * max(path, 1e-6) and np.any(interventions[index : end + 1] > 1e-6):
            route_class[index] = 5
    return {
        "route_class": route_class,
        "previous_side": previous_side,
        "future_side": future_side,
        "future_safe_path_length": future_path,
        "detour_ratio": detour_ratio,
        "future_goal_progress": future_progress,
        "future_intervention_count": future_interventions,
        "freeze_risk": freeze_risk,
        "trajectory_curvature": curvature,
        "time_to_clear_obstacle": time_to_clear,
    }


def _make_filter(telemetry: TelemetryCostModel) -> UAVSafetyActionFilter:
    return UAVSafetyActionFilter(
        SafetyFilterConfig(
            method=SafetyFilterMethod.SAMPLED_DATA_HOCBF,
            safety_dt=SAFETY_DT,
            deadline_seconds=SAFETY_DT,
            horizontal_acceleration_limit=HORIZONTAL_A_MAX,
            vertical_acceleration_limit=VERTICAL_A_MAX,
            horizontal_velocity_limit=HORIZONTAL_V_MAX,
            vertical_velocity_limit=VERTICAL_V_MAX,
            sampled_data_robust=True,
            energy_matrix=np.diag(telemetry.config.acceleration_coefficients),
        ),
        HOCBFConfig(
            k1=1.0,
            k2=1.0,
            uav_radius=UAV_RADIUS,
            uncertainty_margin=CERTIFIED_CLEARANCE_MARGIN,
        ),
    )


def _boundary_project(position: np.ndarray, velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    projected_position = np.clip(position, 0.0, WORLD_SIZE)
    contact = bool(np.any(np.abs(projected_position - position) > 1e-12))
    projected_velocity = velocity.copy()
    if contact:
        for axis in range(3):
            if projected_position[axis] != position[axis]:
                projected_velocity[axis] = 0.0
    return projected_position, projected_velocity, contact


def collect_trajectory(
    policy,
    scenario: GroundedScenario,
    config: GroundedDatasetConfig,
    trajectory_index: int,
) -> tuple[dict[str, np.ndarray], dict[str, object], list[dict[str, object]]]:
    telemetry = TelemetryCostModel()
    action_filter = _make_filter(telemetry)
    rng = np.random.default_rng(np.random.SeedSequence([config.seed, trajectory_index, 911]))
    position = scenario.start.copy()
    velocity = np.zeros(3, dtype=np.float64)
    obstacle_centers = [item.center.copy() for item in scenario.obstacles]
    obstacle_velocities = [item.velocity.copy() for item in scenario.obstacles]
    obstacle_accelerations = [item.acceleration.copy() for item in scenario.obstacles]
    measurement_history: list[tuple[np.ndarray, int, str, bool]] = []
    arrays: dict[str, list[np.ndarray | float | int | bool]] = {
        "feature": [],
        "energy_state": [],
        "position": [],
        "velocity": [],
        "obstacle_position": [],
        "nominal_action": [],
        "executed_action": [],
        "safe_energy": [],
        "nominal_energy": [],
        "intervention": [],
        "boundary_contact": [],
        "clearance": [],
        "filter_feasible": [],
        "fallback_unsafe": [],
        "minimum_executed_slack": [],
    }
    contraction_rows: list[dict[str, object]] = []
    nominal = np.zeros(3, dtype=np.float64)
    success = False
    collision_steps = 0
    path_length = 0.0
    started = perf_counter()
    for step in range(config.max_steps):
        if step % 4 == 0:
            action, _ = policy.predict(sac_observation(position, velocity, scenario.goal), deterministic=True)
            nominal = normalized_action_to_acceleration(action)
        exact_obstacles = tuple(
            SphericalObstacle(
                obstacle_centers[index],
                item.radius,
                obstacle_velocities[index],
                obstacle_accelerations[index],
                item.identifier,
            )
            for index, item in enumerate(scenario.obstacles)
        )
        filtered = action_filter.filter(
            position,
            velocity,
            nominal,
            exact_obstacles,
            progress_direction=scenario.goal - position,
        )
        executed = filtered.acceleration
        next_velocity, realized_acceleration, _ = _apply_velocity_limits(
            velocity,
            executed,
            SAFETY_DT,
        )
        nominal_velocity, nominal_realized_acceleration, _ = _apply_velocity_limits(
            velocity,
            nominal,
            SAFETY_DT,
        )
        next_position = position + velocity * SAFETY_DT + 0.5 * realized_acceleration * SAFETY_DT**2
        next_position, projected_velocity, boundary_contact = _boundary_project(next_position, next_velocity)
        safe_energy = telemetry.realized_cost(
            NavigationState(next_position.copy(), next_velocity.copy(), 0.0, (step + 1) * SAFETY_DT),
            realized_acceleration,
            SAFETY_DT,
        )
        nominal_position = position + velocity * SAFETY_DT + 0.5 * nominal_realized_acceleration * SAFETY_DT**2
        nominal_energy = telemetry.realized_cost(
            NavigationState(nominal_position, nominal_velocity, 0.0, (step + 1) * SAFETY_DT),
            nominal_realized_acceleration,
            SAFETY_DT,
        )
        previous_distance = float(np.linalg.norm(scenario.goal - position))
        next_distance = float(np.linalg.norm(scenario.goal - next_position))
        progress = previous_distance - next_distance
        measurement_valid = False
        if step % config.sensor_period_steps == 0:
            dropout = config.dropout_probability
            if scenario.family == "dropout" and 120 <= step < 260:
                dropout = 0.65
            measurement_valid = bool(rng.random() >= dropout)
            if measurement_valid:
                measurement = obstacle_centers[0] + rng.uniform(
                    -config.sensor_error_bound,
                    config.sensor_error_bound,
                    size=3,
                )
                measurement_history.append((measurement, step, scenario.obstacles[0].identifier, True))
        arrays["feature"].append(
            _transition_feature(
                position,
                velocity,
                scenario.goal,
                obstacle_centers[0],
                obstacle_velocities[0],
                nominal,
                realized_acceleration,
                filtered.diagnostics.intervention_norm,
                progress,
                measurement_valid,
                boundary_contact,
            )
        )
        arrays["energy_state"].append(_energy_state(position, velocity, scenario.goal))
        arrays["position"].append(position.copy())
        arrays["velocity"].append(velocity.copy())
        arrays["obstacle_position"].append(obstacle_centers[0].copy())
        arrays["nominal_action"].append(nominal.copy())
        arrays["executed_action"].append(realized_acceleration.copy())
        arrays["safe_energy"].append(safe_energy)
        arrays["nominal_energy"].append(nominal_energy)
        arrays["intervention"].append(filtered.diagnostics.intervention_norm)
        arrays["boundary_contact"].append(boundary_contact)
        clearance = min(
            float(np.linalg.norm(next_position - obstacle.center) - obstacle.radius - UAV_RADIUS)
            for obstacle in exact_obstacles
        )
        arrays["clearance"].append(clearance)
        arrays["filter_feasible"].append(filtered.diagnostics.feasible)
        arrays["fallback_unsafe"].append(
            filtered.diagnostics.fallback_used
            and not filtered.diagnostics.fallback_satisfies_constraints
        )
        arrays["minimum_executed_slack"].append(
            np.nan
            if filtered.diagnostics.minimum_executed_slack is None
            else filtered.diagnostics.minimum_executed_slack
        )
        collision_steps += int(clearance <= 0.0)
        if step % config.contraction_stride == 0 and measurement_history:
            current_time = step * SAFETY_DT
            recent = [
                item
                for item in measurement_history
                if current_time - item[1] * SAFETY_DT <= config.history_seconds + 1e-12
            ][-CONTRACTION_HISTORY:]
            base_measurement, base_step, track_id, association = recent[-1]
            candidates = list(recent[:-1])
            if step % (2 * config.contraction_stride) == 0:
                candidates.append(
                    (
                        base_measurement + rng.uniform(8.0, 16.0, size=3),
                        max(0, base_step - config.sensor_period_steps),
                        f"decoy_{scenario.identifier}",
                        False,
                    )
                )
            contraction_rows.append(
                {
                    "trajectory_id": trajectory_index,
                    "step": step,
                    "track_id": track_id,
                    "base_measurement": base_measurement.tolist(),
                    "base_age_seconds": current_time - base_step * SAFETY_DT,
                    "base_epoch": base_step,
                    "candidate_measurements": [item[0].tolist() for item in candidates],
                    "candidate_ages_seconds": [current_time - item[1] * SAFETY_DT for item in candidates],
                    "candidate_epochs": [item[1] for item in candidates],
                    "candidate_track_ids": [item[2] for item in candidates],
                    "candidate_association_verified": [item[3] for item in candidates],
                    "true_state": np.concatenate(
                        (obstacle_centers[0], obstacle_velocities[0], obstacle_accelerations[0])
                    ).tolist(),
                }
            )
        previous_position = position.copy()
        position = next_position
        velocity = projected_velocity
        path_length += float(np.linalg.norm(position - previous_position))
        for obstacle_index, item in enumerate(scenario.obstacles):
            (
                obstacle_centers[obstacle_index],
                obstacle_velocities[obstacle_index],
                obstacle_accelerations[obstacle_index],
                _,
            ) = _advance_obstacle(
                item,
                obstacle_centers[obstacle_index],
                obstacle_velocities[obstacle_index],
                obstacle_accelerations[obstacle_index],
                step,
            )
        if next_distance <= GOAL_RADIUS:
            success = True
            break
    dense = {name: np.asarray(values) for name, values in arrays.items()}
    route = _route_labels(
        dense["position"],
        dense["velocity"],
        dense["intervention"],
        scenario.goal,
        dense["obstacle_position"],
    )
    dense.update(route)
    dense["safe_energy_to_go"] = np.cumsum(dense["safe_energy"][::-1])[::-1]
    dense["nominal_energy_to_go"] = np.cumsum(dense["nominal_energy"][::-1])[::-1]
    dense["energy_overhead_to_go"] = dense["safe_energy_to_go"] - dense["nominal_energy_to_go"]
    dense["future_energy_overhead_h40"] = np.asarray(
        [
            np.sum(
                dense["safe_energy"][index : index + 40]
                - dense["nominal_energy"][index : index + 40]
            )
            for index in range(dense["safe_energy"].shape[0])
        ],
        dtype=np.float32,
    )
    metadata = {
        "trajectory_id": trajectory_index,
        "scenario_id": scenario.identifier,
        "family": scenario.family,
        "success": success,
        "steps": int(dense["feature"].shape[0]),
        "collision_steps": collision_steps,
        "path_length": path_length,
        "straight_distance": float(np.linalg.norm(scenario.goal - scenario.start)),
        "path_ratio": path_length / max(float(np.linalg.norm(scenario.goal - scenario.start)), 1e-12),
        "realized_energy": float(np.sum(dense["safe_energy"])),
        "nominal_energy": float(np.sum(dense["nominal_energy"])),
        "energy_overhead": float(np.sum(dense["safe_energy"] - dense["nominal_energy"])),
        "mean_intervention": float(np.mean(dense["intervention"])),
        "intervention_fraction": float(np.mean(dense["intervention"] > 1e-6)),
        "boundary_contact_steps": int(np.sum(dense["boundary_contact"])),
        "minimum_clearance": float(np.min(dense["clearance"])),
        "infeasible_steps": int(np.sum(~dense["filter_feasible"].astype(bool))),
        "fallback_unsafe_steps": int(np.sum(dense["fallback_unsafe"].astype(bool))),
        "collection_seconds": perf_counter() - started,
    }
    return dense, metadata, contraction_rows


def _concatenate_trajectories(
    trajectories: list[dict[str, np.ndarray]],
    metadata: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    keys = trajectories[0].keys()
    packed = {key: np.concatenate([trajectory[key] for trajectory in trajectories]) for key in keys}
    packed["trajectory_id"] = np.concatenate(
        [
            np.full(int(row["steps"]), int(row["trajectory_id"]), dtype=np.int64)
            for row in metadata
        ]
    )
    packed["step_in_trajectory"] = np.concatenate(
        [np.arange(int(row["steps"]), dtype=np.int32) for row in metadata]
    )
    return packed


def _split_ids(trajectory_ids: list[int], seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    ids = rng.permutation(np.asarray(trajectory_ids, dtype=np.int64))
    train_end = int(0.60 * ids.size)
    validation_end = int(0.80 * ids.size)
    return {
        "train": sorted(int(value) for value in ids[:train_end]),
        "validation": sorted(int(value) for value in ids[train_end:validation_end]),
        "test": sorted(int(value) for value in ids[validation_end:]),
    }


def collect_grounded_real_dataset(config: GroundedDatasetConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    policy_path = Path(config.sac_checkpoint)
    policy = load_frozen_sac(policy_path, config.device)
    trajectories: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, object]] = []
    rejected_metadata: list[dict[str, object]] = []
    contraction_rows: list[dict[str, object]] = []
    transitions = 0
    index = 0
    while transitions < config.minimum_controlled_transitions and index < config.max_scenarios:
        scenario = make_scenario(index, config.seed, config.obstacle_radius)
        dense, row, snapshots = collect_trajectory(policy, scenario, config, index)
        hard_filter_valid = bool(
            int(row["collision_steps"]) == 0
            and int(row["infeasible_steps"]) == 0
            and int(row["fallback_unsafe_steps"]) == 0
        )
        if config.require_hard_filter_valid and not hard_filter_valid:
            rejected_metadata.append(
                {
                    **row,
                    "accepted_for_diagnostic": False,
                    "rejection_reason": "hard_filter_precondition_or_feasibility_failure",
                }
            )
            index += 1
            continue
        trajectories.append(dense)
        metadata.append({**row, "accepted_for_diagnostic": True})
        contraction_rows.extend(snapshots)
        transitions += int(row["steps"])
        index += 1
        if index % 10 == 0:
            print(f"[grounded-data] trajectories={index} transitions={transitions}", flush=True)
    if transitions < config.minimum_controlled_transitions:
        raise RuntimeError(
            f"collected {transitions} transitions before max_scenarios={config.max_scenarios}"
        )
    packed = _concatenate_trajectories(trajectories, metadata)
    np.savez_compressed(output / "transitions.npz", **packed)
    with (output / "trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "rejected_trajectories.jsonl").open("w", encoding="utf-8") as handle:
        for row in rejected_metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with (output / "contraction_snapshots.jsonl").open("w", encoding="utf-8") as handle:
        for row in contraction_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    splits = _split_ids(
        [int(row["trajectory_id"]) for row in metadata],
        config.seed + 40_001,
    )
    (output / "splits.json").write_text(
        json.dumps(
            {
                "split_unit": "whole_trajectory",
                "seed": config.seed + 40_001,
                **splits,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "config": asdict(config),
        "git_sha": _git_sha(),
        "source_sha256": _sha256(Path(__file__)),
        "sac_checkpoint_sha256": _sha256(policy_path),
        "controlled_transitions": transitions,
        "trajectory_count": len(metadata),
        "rejected_trajectory_count": len(rejected_metadata),
        "rejected_trajectory_ids": [int(row["trajectory_id"]) for row in rejected_metadata],
        "contraction_snapshot_count": len(contraction_rows),
        "trajectory_split": {name: len(values) for name, values in splits.items()},
        "families": {
            family: sum(row["family"] == family for row in metadata)
            for family in SCENARIO_FAMILIES
        },
        "success_rate": float(np.mean([row["success"] for row in metadata])),
        "collision_step_count": int(sum(row["collision_steps"] for row in metadata)),
        "infeasible_step_count": int(sum(row["infeasible_steps"] for row in metadata)),
        "fallback_unsafe_step_count": int(sum(row["fallback_unsafe_steps"] for row in metadata)),
        "hard_safety_certificate": (
            "identical exact-state sampled-data HOCBF with fixed 2m certified clearance margin "
            "for all collected rows"
        ),
        "labels": {
            "route": [
                "previous_avoidance_side",
                "future_avoidance_side",
                "future_safe_path_length",
                "detour_ratio",
                "future_goal_progress",
                "future_intervention_count",
                "freeze_risk",
                "corridor_identity",
                "trajectory_curvature",
                "time_to_clear_obstacle",
            ],
            "energy": [
                "E_nominal",
                "E_safe",
                "E_overhead",
                "future_H40_energy_overhead",
                "terminal_MC_energy_to_go",
            ],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary
