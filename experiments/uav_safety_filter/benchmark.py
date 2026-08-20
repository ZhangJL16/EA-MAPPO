from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel
from review_bundle.safety.energy.gradients import mc_energy_physical_gradient
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from review_bundle.safety.collision.filter import (
    SafetyFilterConfig,
    SafetyFilterMethod,
    UAVSafetyActionFilter,
)
from review_bundle.safety.collision.hocbf import (
    HOCBFConfig,
    SphericalObstacle,
    actuator_polygon_constraints,
    aggregate_hocbf_constraint,
    energy_to_go_action_gradient,
    project_polyhedral_qp,
    project_single_halfspace,
    select_top_k_constraints,
    sphere_hocbf_constraint,
)
from review_bundle.safety.collision.geometry3d import (
    Lidar3DConfig,
    Lidar3DModel,
    SphereGeometry,
)


WORLD_SIZE = np.array([4000.0, 4000.0, 400.0], dtype=np.float64)
HORIZONTAL_V_MAX = 20.0
VERTICAL_V_MAX = 5.0
HORIZONTAL_A_MAX = 5.0
VERTICAL_A_MAX = 3.0
SAFETY_DT = 0.05
POLICY_PERIOD_STEPS = 4
LIDAR_PERIOD_STEPS = 2
LIDAR_RANGE = 100.0
GOAL_RADIUS = 5.0
UAV_RADIUS = 0.5


def sac_observation(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
) -> np.ndarray:
    p = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    target = np.asarray(goal, dtype=np.float64)
    delta = target - p
    distance = float(np.linalg.norm(delta))
    direction = delta / max(distance, 1e-9)
    feature = np.array(
        [
            v[0] / HORIZONTAL_V_MAX,
            v[1] / HORIZONTAL_V_MAX,
            v[2] / VERTICAL_V_MAX,
            direction[0],
            direction[1],
            direction[2],
            np.log1p(distance) / np.log1p(np.linalg.norm(WORLD_SIZE)),
        ],
        dtype=np.float32,
    )
    return np.clip(feature, np.array([-1.0] * 6 + [0.0]), 1.0)


def normalized_action_to_acceleration(action: np.ndarray) -> np.ndarray:
    normalized = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
    horizontal = normalized[:2]
    norm = float(np.linalg.norm(horizontal))
    if norm > 1.0:
        horizontal = horizontal / norm
    return np.array(
        [
            horizontal[0] * HORIZONTAL_A_MAX,
            horizontal[1] * HORIZONTAL_A_MAX,
            normalized[2] * VERTICAL_A_MAX,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class BenchmarkScenario:
    identifier: str
    family: str
    start: np.ndarray
    goal: np.ndarray
    initial_velocity: np.ndarray
    obstacles: tuple[SphericalObstacle, ...]
    max_steps: int = 1600


@dataclass(frozen=True)
class RolloutResult:
    method: str
    scenario_id: str
    family: str
    success: bool
    collision_steps: int
    near_collision_steps: int
    minimum_clearance: float
    hocbf_violation_steps: int
    infeasible_steps: int
    fallback_unsafe_steps: int
    deadline_misses: int
    minimum_h: float
    minimum_psi1: float
    minimum_executed_slack: float
    minimum_feasibility_margin: float | None
    feasibility_margin_negative_steps: int
    maximum_feasibility_margin_duality_gap: float
    steps: int
    flight_time: float
    path_length: float
    straight_distance: float
    path_ratio: float
    realized_energy: float
    acceleration_energy: float
    energy_per_meter: float
    intervention_steps: int
    intervention_fraction: float
    mean_intervention: float
    max_consecutive_intervention: int
    mean_solver_seconds: float
    median_solver_seconds: float
    p90_solver_seconds: float
    p95_solver_seconds: float
    p99_solver_seconds: float
    mean_total_filter_seconds: float
    median_total_filter_seconds: float
    p90_total_filter_seconds: float
    p95_total_filter_seconds: float
    p99_total_filter_seconds: float
    max_total_filter_seconds: float
    mean_lidar_seconds: float
    median_lidar_seconds: float
    p90_lidar_seconds: float
    p95_lidar_seconds: float
    p99_lidar_seconds: float
    max_lidar_seconds: float
    mean_extraction_seconds: float
    mean_constraint_build_seconds: float
    mean_raw_lidar_points: float
    mean_candidate_obstacles: float
    mean_active_constraints: float
    mean_energy_gradient_norm: float
    mean_energy_gradient_seconds: float
    max_candidate_obstacles: int
    mean_speed: float
    max_speed: float
    goal_progress: float
    final_goal_distance: float


def _segment_sphere_clearance(
    start: np.ndarray,
    end: np.ndarray,
    obstacle: SphericalObstacle,
) -> float:
    segment = end - start
    denominator = float(segment @ segment)
    if denominator <= 1e-15:
        closest = start
    else:
        fraction = float(
            np.clip((obstacle.center - start) @ segment / denominator, 0.0, 1.0)
        )
        closest = start + fraction * segment
    return float(
        np.linalg.norm(closest - obstacle.center) - obstacle.radius - UAV_RADIUS
    )


def perturb_perceived_obstacles(
    position: np.ndarray,
    detected_indices: np.ndarray,
    obstacles: tuple[SphericalObstacle, ...],
    *,
    range_noise_std: float,
    range_noise_clip_sigma: float,
    obstacle_dropout_probability: float,
    rng: np.random.Generator,
) -> tuple[SphericalObstacle, ...]:
    if range_noise_std < 0.0 or range_noise_clip_sigma <= 0.0:
        raise ValueError("range noise scale must be nonnegative and clip positive")
    if not 0.0 <= obstacle_dropout_probability < 1.0:
        raise ValueError("obstacle dropout probability must lie in [0, 1)")
    point = np.asarray(position, dtype=np.float64)
    perceived = []
    for raw_index in np.asarray(detected_indices, dtype=np.int64):
        if rng.random() < obstacle_dropout_probability:
            continue
        obstacle = obstacles[int(raw_index)]
        center = obstacle.center.copy()
        if range_noise_std > 0.0:
            radial = obstacle.center - point
            radial /= max(float(np.linalg.norm(radial)), 1e-12)
            error = float(
                np.clip(
                    rng.normal(0.0, range_noise_std),
                    -range_noise_clip_sigma * range_noise_std,
                    range_noise_clip_sigma * range_noise_std,
                )
            )
            center += error * radial
        perceived.append(
            SphericalObstacle(
                center,
                obstacle.radius,
                obstacle.velocity,
                obstacle.acceleration,
                obstacle.identifier,
            )
        )
    return tuple(perceived)


def make_scenarios(count: int, seed: int) -> list[BenchmarkScenario]:
    if count <= 0:
        raise ValueError("scenario count must be positive")
    rng = np.random.default_rng(seed)
    families = ("sparse", "medium", "dense", "narrow", "vertical", "long")
    scenarios = []
    for index in range(count):
        family = families[index % len(families)]
        angle = rng.uniform(-np.pi, np.pi)
        direction = np.array([np.cos(angle), np.sin(angle), rng.uniform(-0.08, 0.08)])
        direction /= np.linalg.norm(direction)
        distance = 250.0 if family != "long" else 450.0
        center = np.array([2000.0, 2000.0, 200.0]) + rng.uniform(-150.0, 150.0, size=3)
        center[2] = np.clip(center[2], 80.0, 320.0)
        start = center - 0.5 * distance * direction
        goal = center + 0.5 * distance * direction
        lateral = np.array([-direction[1], direction[0], 0.0])
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        vertical = np.array([0.0, 0.0, 1.0])
        obstacles: list[SphericalObstacle] = []
        if family == "sparse":
            offsets = [(0.0, 0.0, 0.0, 35.0)]
        elif family == "medium":
            offsets = [(-40.0, 4.0, 0.0, 30.0), (40.0, -4.0, 0.0, 30.0)]
        elif family == "dense":
            offsets = [
                (-60.0, 8.0, 0.0, 25.0),
                (-20.0, -7.0, 5.0, 24.0),
                (25.0, 8.0, -4.0, 24.0),
                (70.0, -6.0, 2.0, 23.0),
            ]
        elif family == "narrow":
            offsets = [(0.0, 30.0, 0.0, 27.0), (0.0, -30.0, 0.0, 27.0)]
        elif family == "vertical":
            offsets = [(0.0, 0.0, -30.0, 27.0), (0.0, 0.0, 30.0, 27.0)]
        else:
            offsets = [
                (-120.0, 4.0, 0.0, 32.0),
                (0.0, -4.0, 0.0, 32.0),
                (120.0, 4.0, 0.0, 32.0),
            ]
        for obstacle_index, (forward, side, up, radius) in enumerate(offsets):
            obstacle_center = center + forward * direction + side * lateral + up * vertical
            obstacles.append(
                SphericalObstacle(
                    obstacle_center,
                    radius,
                    identifier=f"{family}_{index}_{obstacle_index}",
                )
            )
        scenarios.append(
            BenchmarkScenario(
                identifier=f"{family}_{index:04d}",
                family=family,
                start=start,
                goal=goal,
                initial_velocity=np.zeros(3),
                obstacles=tuple(obstacles),
                max_steps=2400 if family == "long" else 1600,
            )
        )
    return scenarios


def run_rollout(
    policy_action: Callable[[np.ndarray], np.ndarray],
    scenario: BenchmarkScenario,
    method: str,
    *,
    energy_weight: float = 1.0,
    energy_gradient_weight: float = 1.0,
    energy_gradient_characteristic: float = 1.0,
    energy_gradient_normalization: str = "fixed",
    energy_estimator: EnergyToGoRegressor | None = None,
    range_noise_std: float = 0.0,
    range_noise_clip_sigma: float = 3.0,
    obstacle_dropout_probability: float = 0.0,
    lidar_period_steps: int = LIDAR_PERIOD_STEPS,
    perception_uncertainty_margin: float = 0.0,
    perception_seed: int = 0,
) -> RolloutResult:
    if lidar_period_steps <= 0 or perception_uncertainty_margin < 0.0:
        raise ValueError("LiDAR period must be positive and margin nonnegative")
    perception_rng = np.random.default_rng(perception_seed)
    position = scenario.start.astype(np.float64).copy()
    velocity = scenario.initial_velocity.astype(np.float64).copy()
    nominal_acceleration = np.zeros(3)
    telemetry = TelemetryCostModel()
    lidar = Lidar3DModel(
        Lidar3DConfig(
            horizontal_sectors=128,
            vertical_sectors=8,
            max_range=LIDAR_RANGE,
        )
    )
    lidar_geometries = tuple(
        SphereGeometry(obstacle.center, obstacle.radius, obstacle.identifier)
        for obstacle in scenario.obstacles
    )
    action_filter = None
    if method != "frozen_sac_no_filter":
        filter_method = SafetyFilterMethod(method)
        energy_matrix = np.diag(telemetry.config.acceleration_coefficients)
        max_action_energy = SAFETY_DT * (
            telemetry.config.acceleration_coefficients[0] * HORIZONTAL_A_MAX**2
            + telemetry.config.acceleration_coefficients[2] * VERTICAL_A_MAX**2
        )
        action_filter = UAVSafetyActionFilter(
            SafetyFilterConfig(
                method=filter_method,
                safety_dt=SAFETY_DT,
                deadline_seconds=SAFETY_DT,
                energy_matrix=energy_matrix,
                energy_characteristic=max(max_action_energy, 1e-12),
                energy_weight=energy_weight,
                energy_gradient_weight=energy_gradient_weight,
                energy_gradient_characteristic=energy_gradient_characteristic,
                energy_gradient_normalization=energy_gradient_normalization,
                horizontal_velocity_limit=HORIZONTAL_V_MAX,
                vertical_velocity_limit=VERTICAL_V_MAX,
                sampled_data_robust=(
                    filter_method
                    in (
                        SafetyFilterMethod.SAMPLED_DATA_HOCBF,
                        SafetyFilterMethod.SAMPLED_DATA_ENERGY_AWARE_HOCBF,
                        SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF,
                        SafetyFilterMethod.SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF,
                    )
                ),
            ),
            HOCBFConfig(
                k1=1.0,
                k2=1.0,
                uav_radius=UAV_RADIUS,
                uncertainty_margin=perception_uncertainty_margin,
            ),
        )
    path_length = 0.0
    realized_energy = 0.0
    collision_steps = 0
    near_steps = 0
    min_clearance = np.inf
    minimum_h = np.inf
    minimum_psi1 = np.inf
    minimum_executed_slack = np.inf
    minimum_feasibility_margin = np.inf
    feasibility_margin_negative_steps = 0
    maximum_feasibility_margin_duality_gap = 0.0
    violation_steps = 0
    infeasible_steps = 0
    fallback_unsafe_steps = 0
    deadline_misses = 0
    interventions = []
    solver_times = []
    total_times = []
    lidar_times = []
    extraction_times = []
    constraint_build_times = []
    raw_lidar_points = []
    candidate_obstacle_counts = []
    active_constraint_counts = []
    energy_gradient_norms = []
    energy_gradient_times = []
    speeds = []
    acceleration_energy = 0.0
    consecutive_intervention = 0
    max_consecutive_intervention = 0
    perceived_obstacles: tuple[SphericalObstacle, ...] = ()
    success = False
    steps = 0

    for step in range(scenario.max_steps):
        if step % POLICY_PERIOD_STEPS == 0:
            observation = sac_observation(position, velocity, scenario.goal)
            nominal_acceleration = normalized_action_to_acceleration(policy_action(observation))
        if action_filter is not None and step % lidar_period_steps == 0:
            lidar_started = perf_counter()
            packet = lidar.measure(position, lidar_geometries, timestamp=step * SAFETY_DT)
            lidar_times.append(perf_counter() - lidar_started)
            extraction_started = perf_counter()
            detected_indices = np.unique(packet.obstacle_indices[packet.hit])
            perceived_obstacles = perturb_perceived_obstacles(
                position,
                detected_indices,
                scenario.obstacles,
                range_noise_std=range_noise_std,
                range_noise_clip_sigma=range_noise_clip_sigma,
                obstacle_dropout_probability=obstacle_dropout_probability,
                rng=perception_rng,
            )
            extraction_times.append(perf_counter() - extraction_started)
            raw_lidar_points.append(int(np.sum(packet.hit)))
        executed = nominal_acceleration
        if action_filter is not None:
            energy_gradient = None
            gradient_seconds = 0.0
            if method in (
                SafetyFilterMethod.ENERGY_GRADIENT_HOCBF.value,
                SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF.value,
            ):
                if energy_estimator is None:
                    raise ValueError("energy-gradient benchmark requires an energy estimator")
                gradient_started = perf_counter()
                predicted_next_position = (
                    position
                    + velocity * SAFETY_DT
                    + 0.5 * nominal_acceleration * SAFETY_DT**2
                )
                predicted_next_velocity = velocity + nominal_acceleration * SAFETY_DT
                physical_gradient = mc_energy_physical_gradient(
                    energy_estimator,
                    predicted_next_position,
                    predicted_next_velocity,
                    scenario.goal,
                    horizontal_velocity_limit=HORIZONTAL_V_MAX,
                    vertical_velocity_limit=VERTICAL_V_MAX,
                    distance_scale=float(np.linalg.norm(WORLD_SIZE)),
                )
                energy_gradient = energy_to_go_action_gradient(
                    physical_gradient.grad_position,
                    physical_gradient.grad_velocity,
                    SAFETY_DT,
                )
                gradient_seconds = perf_counter() - gradient_started
                energy_gradient_norms.append(float(np.linalg.norm(energy_gradient)))
                energy_gradient_times.append(gradient_seconds)
            filtered = action_filter.filter(
                position,
                velocity,
                nominal_acceleration,
                perceived_obstacles,
                energy_gradient=energy_gradient,
                progress_direction=scenario.goal - position,
            )
            executed = filtered.acceleration
            diagnostics = filtered.diagnostics
            solver_times.append(diagnostics.solver_seconds)
            total_layer_seconds = diagnostics.total_seconds + gradient_seconds
            total_times.append(total_layer_seconds)
            constraint_build_times.append(diagnostics.constraint_build_seconds)
            candidate_obstacle_counts.append(diagnostics.candidate_constraints)
            active_constraint_counts.append(diagnostics.active_constraints)
            minimum_h = min(
                minimum_h,
                diagnostics.minimum_h if diagnostics.minimum_h is not None else np.inf,
            )
            minimum_psi1 = min(
                minimum_psi1,
                diagnostics.minimum_psi1
                if diagnostics.minimum_psi1 is not None
                else np.inf,
            )
            minimum_executed_slack = min(
                minimum_executed_slack,
                diagnostics.minimum_executed_slack
                if diagnostics.minimum_executed_slack is not None
                else np.inf,
            )
            violation_steps += int(
                diagnostics.minimum_executed_slack is not None
                and diagnostics.minimum_executed_slack < -1e-7
            )
            infeasible_steps += int(not diagnostics.feasible)
            if diagnostics.feasibility_margin is not None:
                minimum_feasibility_margin = min(
                    minimum_feasibility_margin,
                    diagnostics.feasibility_margin,
                )
                feasibility_margin_negative_steps += int(
                    diagnostics.feasibility_margin < -1e-7
                )
            if diagnostics.feasibility_margin_duality_gap is not None:
                maximum_feasibility_margin_duality_gap = max(
                    maximum_feasibility_margin_duality_gap,
                    diagnostics.feasibility_margin_duality_gap,
                )
            fallback_unsafe_steps += int(
                diagnostics.fallback_used
                and not diagnostics.fallback_satisfies_constraints
            )
            deadline_misses += int(total_layer_seconds > SAFETY_DT)
            interventions.append(diagnostics.intervention_norm)
            if diagnostics.intervention_norm > 1e-6:
                consecutive_intervention += 1
                max_consecutive_intervention = max(
                    max_consecutive_intervention,
                    consecutive_intervention,
                )
            else:
                consecutive_intervention = 0
        else:
            interventions.append(0.0)

        velocity_before = velocity.copy()
        velocity += executed * SAFETY_DT
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed > HORIZONTAL_V_MAX:
            velocity[:2] *= HORIZONTAL_V_MAX / horizontal_speed
        velocity[2] = np.clip(velocity[2], -VERTICAL_V_MAX, VERTICAL_V_MAX)
        realized_acceleration = (velocity - velocity_before) / SAFETY_DT
        previous_position = position.copy()
        position += velocity * SAFETY_DT
        path_length += float(np.linalg.norm(position - previous_position))
        realized_energy += telemetry.realized_cost(
            NavigationState(position.copy(), velocity.copy(), 0.0, (step + 1) * SAFETY_DT),
            realized_acceleration,
            SAFETY_DT,
        )
        acceleration_energy += float(
            SAFETY_DT
            * telemetry.config.flight_energy_multiplier
            * np.sum(
                telemetry.config.acceleration_coefficients
                * realized_acceleration
                * realized_acceleration
            )
        )
        speeds.append(float(np.linalg.norm(velocity)))
        step_clearances = [
            _segment_sphere_clearance(previous_position, position, obstacle)
            for obstacle in scenario.obstacles
        ]
        current_clearance = min(step_clearances, default=np.inf)
        min_clearance = min(min_clearance, current_clearance)
        analysis_constraints = [
            sphere_hocbf_constraint(
                position,
                velocity,
                obstacle,
                HOCBFConfig(k1=1.0, k2=1.0, uav_radius=UAV_RADIUS),
            )
            for obstacle in scenario.obstacles
        ]
        minimum_h = min(
            minimum_h,
            min((constraint.h for constraint in analysis_constraints), default=np.inf),
        )
        minimum_psi1 = min(
            minimum_psi1,
            min((constraint.psi1 for constraint in analysis_constraints), default=np.inf),
        )
        minimum_executed_slack = min(
            minimum_executed_slack,
            min(
                (constraint.slack(realized_acceleration) for constraint in analysis_constraints),
                default=np.inf,
            ),
        )
        collision_steps += int(current_clearance <= 0.0)
        near_steps += int(current_clearance <= 2.0)
        steps = step + 1
        if np.linalg.norm(position - scenario.goal) <= GOAL_RADIUS:
            success = True
            break

    straight = float(np.linalg.norm(scenario.goal - scenario.start))
    solver_array = np.asarray(solver_times or [0.0])
    total_array = np.asarray(total_times or [0.0])
    lidar_array = np.asarray(lidar_times or [0.0])
    extraction_array = np.asarray(extraction_times or [0.0])
    build_array = np.asarray(constraint_build_times or [0.0])
    raw_points_array = np.asarray(raw_lidar_points or [0.0])
    candidate_array = np.asarray(candidate_obstacle_counts or [0.0])
    active_array = np.asarray(active_constraint_counts or [0.0])
    energy_gradient_array = np.asarray(energy_gradient_norms or [0.0])
    energy_gradient_time_array = np.asarray(energy_gradient_times or [0.0])
    speed_array = np.asarray(speeds or [0.0])
    interventions_array = np.asarray(interventions)
    return RolloutResult(
        method=method,
        scenario_id=scenario.identifier,
        family=scenario.family,
        success=success,
        collision_steps=collision_steps,
        near_collision_steps=near_steps,
        minimum_clearance=float(min_clearance),
        hocbf_violation_steps=violation_steps,
        infeasible_steps=infeasible_steps,
        fallback_unsafe_steps=fallback_unsafe_steps,
        deadline_misses=deadline_misses,
        minimum_h=float(minimum_h),
        minimum_psi1=float(minimum_psi1),
        minimum_executed_slack=float(minimum_executed_slack),
        minimum_feasibility_margin=(
            float(minimum_feasibility_margin)
            if np.isfinite(minimum_feasibility_margin)
            else None
        ),
        feasibility_margin_negative_steps=feasibility_margin_negative_steps,
        maximum_feasibility_margin_duality_gap=float(
            maximum_feasibility_margin_duality_gap
        ),
        steps=steps,
        flight_time=steps * SAFETY_DT,
        path_length=path_length,
        straight_distance=straight,
        path_ratio=path_length / max(straight, 1e-12),
        realized_energy=realized_energy,
        acceleration_energy=acceleration_energy,
        energy_per_meter=realized_energy / max(path_length, 1e-12),
        intervention_steps=int(np.sum(interventions_array > 1e-6)),
        intervention_fraction=float(np.mean(interventions_array > 1e-6)),
        mean_intervention=float(np.mean(interventions_array)),
        max_consecutive_intervention=max_consecutive_intervention,
        mean_solver_seconds=float(np.mean(solver_array)),
        median_solver_seconds=float(np.median(solver_array)),
        p90_solver_seconds=float(np.quantile(solver_array, 0.90)),
        p95_solver_seconds=float(np.quantile(solver_array, 0.95)),
        p99_solver_seconds=float(np.quantile(solver_array, 0.99)),
        mean_total_filter_seconds=float(np.mean(total_array)),
        median_total_filter_seconds=float(np.median(total_array)),
        p90_total_filter_seconds=float(np.quantile(total_array, 0.90)),
        p95_total_filter_seconds=float(np.quantile(total_array, 0.95)),
        p99_total_filter_seconds=float(np.quantile(total_array, 0.99)),
        max_total_filter_seconds=float(np.max(total_array)),
        mean_lidar_seconds=float(np.mean(lidar_array)),
        median_lidar_seconds=float(np.median(lidar_array)),
        p90_lidar_seconds=float(np.quantile(lidar_array, 0.90)),
        p95_lidar_seconds=float(np.quantile(lidar_array, 0.95)),
        p99_lidar_seconds=float(np.quantile(lidar_array, 0.99)),
        max_lidar_seconds=float(np.max(lidar_array)),
        mean_extraction_seconds=float(np.mean(extraction_array)),
        mean_constraint_build_seconds=float(np.mean(build_array)),
        mean_raw_lidar_points=float(np.mean(raw_points_array)),
        mean_candidate_obstacles=float(np.mean(candidate_array)),
        mean_active_constraints=float(np.mean(active_array)),
        mean_energy_gradient_norm=float(np.mean(energy_gradient_array)),
        mean_energy_gradient_seconds=float(np.mean(energy_gradient_time_array)),
        max_candidate_obstacles=int(np.max(candidate_array)),
        mean_speed=float(np.mean(speed_array)),
        max_speed=float(np.max(speed_array)),
        goal_progress=float(straight - np.linalg.norm(position - scenario.goal)),
        final_goal_distance=float(np.linalg.norm(position - scenario.goal)),
    )


def summarize(results: list[RolloutResult]) -> dict[str, object]:
    grouped: dict[str, list[RolloutResult]] = {}
    for row in results:
        grouped.setdefault(row.method, []).append(row)
    summary: dict[str, object] = {}
    baseline_by_scenario = {
        row.scenario_id: row
        for row in results
        if row.method == "frozen_sac_no_filter"
    }
    hocbf_by_scenario = {
        row.scenario_id: row
        for row in results
        if row.method == "second_order_hocbf"
    }
    if not hocbf_by_scenario:
        hocbf_by_scenario = {
            row.scenario_id: row
            for row in results
            if row.method == "sampled_data_hocbf"
        }
    for method, rows in grouped.items():
        total_steps = sum(row.steps for row in rows)
        successful_rows = [row for row in rows if row.success]
        has_baseline_reference = all(
            row.scenario_id in baseline_by_scenario for row in rows
        )
        has_hocbf_reference = all(
            row.scenario_id in hocbf_by_scenario for row in rows
        )
        baseline_overheads = (
            [
                row.realized_energy
                - baseline_by_scenario[row.scenario_id].realized_energy
                for row in rows
            ]
            if has_baseline_reference
            else None
        )
        hocbf_overheads = (
            [
                row.realized_energy
                - hocbf_by_scenario[row.scenario_id].realized_energy
                for row in rows
            ]
            if has_hocbf_reference
            else None
        )
        feasibility_margins = [
            row.minimum_feasibility_margin
            for row in rows
            if row.minimum_feasibility_margin is not None
        ]
        summary[method] = {
            "rollouts": len(rows),
            "total_method_transitions": total_steps,
            "success_rate": float(np.mean([row.success for row in rows])),
            "collision_rollout_rate": float(np.mean([row.collision_steps > 0 for row in rows])),
            "collision_steps": int(sum(row.collision_steps for row in rows)),
            "collision_step_rate": float(
                sum(row.collision_steps for row in rows) / max(total_steps, 1)
            ),
            "near_collision_steps": int(sum(row.near_collision_steps for row in rows)),
            "near_collision_step_rate": float(
                sum(row.near_collision_steps for row in rows) / max(total_steps, 1)
            ),
            "minimum_clearance": float(min(row.minimum_clearance for row in rows)),
            "minimum_h": float(min(row.minimum_h for row in rows)),
            "minimum_psi1": float(min(row.minimum_psi1 for row in rows)),
            "minimum_executed_slack": float(
                min(row.minimum_executed_slack for row in rows)
            ),
            "minimum_feasibility_margin": (
                float(min(feasibility_margins)) if feasibility_margins else None
            ),
            "feasibility_margin_negative_steps": int(
                sum(row.feasibility_margin_negative_steps for row in rows)
            ),
            "maximum_feasibility_margin_duality_gap": float(
                max(row.maximum_feasibility_margin_duality_gap for row in rows)
            ),
            "hocbf_violation_steps": int(sum(row.hocbf_violation_steps for row in rows)),
            "infeasible_steps": int(sum(row.infeasible_steps for row in rows)),
            "fallback_unsafe_steps": int(sum(row.fallback_unsafe_steps for row in rows)),
            "mean_steps": float(np.mean([row.steps for row in rows])),
            "mean_flight_time": float(np.mean([row.flight_time for row in rows])),
            "mean_path_ratio": float(np.mean([row.path_ratio for row in rows])),
            "mean_successful_path_ratio": (
                float(np.mean([row.path_ratio for row in successful_rows]))
                if successful_rows
                else None
            ),
            "mean_realized_energy": float(np.mean([row.realized_energy for row in rows])),
            "mean_energy_per_successful_task": (
                float(np.mean([row.realized_energy for row in successful_rows]))
                if successful_rows
                else None
            ),
            "mean_acceleration_energy": float(
                np.mean([row.acceleration_energy for row in rows])
            ),
            "mean_energy_per_meter": float(
                np.mean([row.energy_per_meter for row in rows])
            ),
            "mean_energy_overhead_vs_no_safety": (
                float(np.mean(baseline_overheads))
                if baseline_overheads is not None
                else None
            ),
            "mean_energy_overhead_vs_standard_hocbf": (
                float(np.mean(hocbf_overheads))
                if hocbf_overheads is not None
                else None
            ),
            "mean_energy_overhead_fraction_vs_no_safety": (
                float(np.mean(
                    [
                        overhead
                        / max(
                            baseline_by_scenario[row.scenario_id].realized_energy,
                            1e-12,
                        )
                        for row, overhead in zip(rows, baseline_overheads)
                    ]
                ))
                if baseline_overheads is not None
                else None
            ),
            "mean_energy_overhead_fraction_vs_standard_hocbf": (
                float(np.mean(
                    [
                        overhead
                        / max(
                            hocbf_by_scenario[row.scenario_id].realized_energy,
                            1e-12,
                        )
                        for row, overhead in zip(rows, hocbf_overheads)
                    ]
                ))
                if hocbf_overheads is not None
                else None
            ),
            "mean_intervention_fraction": float(
                np.mean([row.intervention_fraction for row in rows])
            ),
            "mean_intervention_norm": float(
                np.mean([row.mean_intervention for row in rows])
            ),
            "max_consecutive_intervention": int(
                max(row.max_consecutive_intervention for row in rows)
            ),
            "mean_solver_ms": 1000.0 * float(np.mean([row.mean_solver_seconds for row in rows])),
            "median_solver_ms": 1000.0
            * float(np.mean([row.median_solver_seconds for row in rows])),
            "worst_rollout_p90_solver_ms": 1000.0
            * float(max(row.p90_solver_seconds for row in rows)),
            "worst_rollout_p95_solver_ms": 1000.0
            * float(max(row.p95_solver_seconds for row in rows)),
            "worst_rollout_p99_solver_ms": 1000.0
            * float(max(row.p99_solver_seconds for row in rows)),
            "mean_total_filter_ms": 1000.0
            * float(np.mean([row.mean_total_filter_seconds for row in rows])),
            "median_total_filter_ms": 1000.0
            * float(np.mean([row.median_total_filter_seconds for row in rows])),
            "worst_rollout_p90_total_filter_ms": 1000.0
            * float(max(row.p90_total_filter_seconds for row in rows)),
            "worst_rollout_p95_total_filter_ms": 1000.0
            * float(max(row.p95_total_filter_seconds for row in rows)),
            "worst_rollout_p99_total_filter_ms": 1000.0
            * float(max(row.p99_total_filter_seconds for row in rows)),
            "max_total_filter_ms": 1000.0
            * float(max(row.max_total_filter_seconds for row in rows)),
            "mean_lidar_ms": 1000.0
            * float(np.mean([row.mean_lidar_seconds for row in rows])),
            "median_lidar_ms": 1000.0
            * float(np.mean([row.median_lidar_seconds for row in rows])),
            "worst_rollout_p90_lidar_ms": 1000.0
            * float(max(row.p90_lidar_seconds for row in rows)),
            "worst_rollout_p95_lidar_ms": 1000.0
            * float(max(row.p95_lidar_seconds for row in rows)),
            "worst_rollout_p99_lidar_ms": 1000.0
            * float(max(row.p99_lidar_seconds for row in rows)),
            "max_lidar_ms": 1000.0
            * float(max(row.max_lidar_seconds for row in rows)),
            "mean_extraction_ms": 1000.0
            * float(np.mean([row.mean_extraction_seconds for row in rows])),
            "mean_constraint_build_ms": 1000.0
            * float(np.mean([row.mean_constraint_build_seconds for row in rows])),
            "mean_raw_lidar_points": float(
                np.mean([row.mean_raw_lidar_points for row in rows])
            ),
            "mean_candidate_obstacles": float(
                np.mean([row.mean_candidate_obstacles for row in rows])
            ),
            "mean_active_constraints": float(
                np.mean([row.mean_active_constraints for row in rows])
            ),
            "mean_energy_gradient_norm": float(
                np.mean([row.mean_energy_gradient_norm for row in rows])
            ),
            "mean_energy_gradient_ms": 1000.0
            * float(np.mean([row.mean_energy_gradient_seconds for row in rows])),
            "max_candidate_obstacles": int(
                max(row.max_candidate_obstacles for row in rows)
            ),
            "mean_speed": float(np.mean([row.mean_speed for row in rows])),
            "max_speed": float(max(row.max_speed for row in rows)),
            "mean_goal_progress": float(np.mean([row.goal_progress for row in rows])),
            "deadline_miss_rate": float(
                sum(row.deadline_misses for row in rows) / max(total_steps, 1)
            ),
            "deadline_misses": int(sum(row.deadline_misses for row in rows)),
            "families": {
                family: {
                    "rollouts": len(family_rows),
                    "success_rate": float(np.mean([row.success for row in family_rows])),
                    "collision_rollout_rate": float(
                        np.mean([row.collision_steps > 0 for row in family_rows])
                    ),
                    "minimum_clearance": float(
                        min(row.minimum_clearance for row in family_rows)
                    ),
                }
                for family in sorted({row.family for row in rows})
                if (family_rows := [row for row in rows if row.family == family])
            },
        }
    return summary


def result_dict(row: RolloutResult) -> dict[str, object]:
    return asdict(row)


def compute_scaling(
    constraint_counts: list[int],
    *,
    repeats: int,
    seed: int,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    output = []
    hocbf_config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=UAV_RADIUS)
    methods = {
        "full_qp": SafetyFilterConfig(method=SafetyFilterMethod.HOCBF),
        "top_k_16_qp": SafetyFilterConfig(
            method=SafetyFilterMethod.HOCBF,
            top_k=16,
        ),
        "aggregate_qp": SafetyFilterConfig(
            method=SafetyFilterMethod.AGGREGATE_HOCBF,
        ),
    }
    for count in constraint_counts:
        method_samples: dict[str, dict[str, list[float]]] = {
            method: {"total": [], "solver": [], "feasible": []}
            for method in (*methods, "aggregate_closed_form")
        }
        for repeat in range(repeats):
            angles = rng.uniform(-np.pi, np.pi, size=count)
            elevations = rng.uniform(-0.4, 0.4, size=count)
            distances = rng.uniform(20.0, 100.0, size=count)
            centers = np.column_stack(
                (
                    distances * np.cos(elevations) * np.cos(angles),
                    distances * np.cos(elevations) * np.sin(angles),
                    distances * np.sin(elevations),
                )
            )
            obstacles = [
                SphericalObstacle(center, 1.0, identifier=f"{count}_{index}")
                for index, center in enumerate(centers)
            ]
            nominal = rng.uniform(-1.0, 1.0, size=3)
            for method, config in methods.items():
                action_filter = UAVSafetyActionFilter(config, hocbf_config)
                started = perf_counter()
                result = action_filter.filter(
                    np.zeros(3),
                    np.zeros(3),
                    nominal,
                    obstacles,
                )
                method_samples[method]["total"].append(perf_counter() - started)
                method_samples[method]["solver"].append(
                    result.diagnostics.solver_seconds
                )
                method_samples[method]["feasible"].append(
                    float(result.diagnostics.feasible)
                )

            started = perf_counter()
            constraints = [
                sphere_hocbf_constraint(
                    np.zeros(3),
                    np.zeros(3),
                    obstacle,
                    hocbf_config,
                )
                for obstacle in obstacles
            ]
            aggregate = aggregate_hocbf_constraint(
                constraints,
                np.zeros(3),
                np.zeros(3),
                obstacles,
                hocbf_config,
                rho=1.0,
            )
            solve_started = perf_counter()
            projected = project_single_halfspace(
                nominal,
                aggregate.row,
                aggregate.lower_bound,
            )
            solve_elapsed = perf_counter() - solve_started
            total_elapsed = perf_counter() - started
            method_samples["aggregate_closed_form"]["total"].append(total_elapsed)
            method_samples["aggregate_closed_form"]["solver"].append(solve_elapsed)
            method_samples["aggregate_closed_form"]["feasible"].append(
                float(aggregate.slack(projected) >= -1e-8)
            )

        for method, samples in method_samples.items():
            total = np.asarray(samples["total"])
            solve = np.asarray(samples["solver"])
            output.append(
                {
                    "method": method,
                    "constraints": count,
                    "solved_constraints": (
                        min(count, 16) if method == "top_k_16_qp" else count
                    ),
                    "repeats": repeats,
                    "feasible_rate": float(np.mean(samples["feasible"])),
                    "mean_total_ms": 1000.0 * float(np.mean(total)),
                    "median_total_ms": 1000.0 * float(np.median(total)),
                    "p90_total_ms": 1000.0 * float(np.quantile(total, 0.90)),
                    "p95_total_ms": 1000.0 * float(np.quantile(total, 0.95)),
                    "p99_total_ms": 1000.0 * float(np.quantile(total, 0.99)),
                    "max_total_ms": 1000.0 * float(np.max(total)),
                    "mean_solver_ms": 1000.0 * float(np.mean(solve)),
                    "median_solver_ms": 1000.0 * float(np.median(solve)),
                    "p90_solver_ms": 1000.0 * float(np.quantile(solve, 0.90)),
                    "p95_solver_ms": 1000.0 * float(np.quantile(solve, 0.95)),
                    "p99_solver_ms": 1000.0 * float(np.quantile(solve, 0.99)),
                    "max_solver_ms": 1000.0 * float(np.max(solve)),
                    "deadline_miss_rate": float(np.mean(total > SAFETY_DT)),
                    "actuator_limits_included": method != "aggregate_closed_form",
                }
            )
    return output


def compare_top_k_priorities(
    constraint_counts: list[int],
    *,
    top_k: int,
    trials: int,
    seed: int,
) -> list[dict[str, object]]:
    if top_k <= 0 or trials <= 0:
        raise ValueError("top_k and trials must be positive")
    priorities = (
        "distance",
        "closing_speed",
        "ttc",
        "barrier_value",
        "hocbf_slack",
        "braking_margin",
    )
    rng = np.random.default_rng(seed)
    hocbf_config = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=UAV_RADIUS)
    actuator_rows, actuator_bounds = actuator_polygon_constraints(
        HORIZONTAL_A_MAX,
        VERTICAL_A_MAX,
        facets=32,
    )
    braking_acceleration = float(np.hypot(HORIZONTAL_A_MAX, VERTICAL_A_MAX))
    output: list[dict[str, object]] = []
    for count in constraint_counts:
        samples = {
            priority: {
                "selection": [],
                "solver": [],
                "total": [],
                "omitted_violations": [],
                "max_omitted_violation": [],
                "action_deviation": [],
                "full_action_difference": [],
                "feasible": [],
            }
            for priority in priorities
        }
        full_feasible = []
        for trial in range(trials):
            velocity_direction = rng.normal(size=3)
            velocity_direction[2] *= 0.25
            velocity_direction /= max(
                float(np.linalg.norm(velocity_direction)),
                1e-12,
            )
            velocity = velocity_direction * rng.uniform(5.0, 12.0)
            velocity[2] = np.clip(
                velocity[2],
                -VERTICAL_V_MAX,
                VERTICAL_V_MAX,
            )
            danger_count = min(count, top_k + 4)
            decoy_count = min(max(count - danger_count, 0), top_k)
            centers = []
            for index in range(count):
                if index < danger_count:
                    direction = velocity_direction + rng.normal(scale=0.18, size=3)
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    distance = rng.uniform(25.0, 80.0)
                elif index < danger_count + decoy_count:
                    direction = -velocity_direction + rng.normal(scale=0.18, size=3)
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    distance = rng.uniform(10.0, 30.0)
                else:
                    direction = rng.normal(size=3)
                    direction[2] *= 0.45
                    direction /= max(float(np.linalg.norm(direction)), 1e-12)
                    distance = rng.uniform(30.0, 100.0)
                centers.append(direction * distance)
            obstacles = [
                SphericalObstacle(
                    center,
                    rng.uniform(1.0, 4.0),
                    identifier=f"{count}_{trial}_{index}",
                )
                for index, center in enumerate(centers)
            ]
            nominal_horizontal = rng.normal(size=2)
            nominal_horizontal *= rng.uniform(0.0, HORIZONTAL_A_MAX) / max(
                float(np.linalg.norm(nominal_horizontal)),
                1e-12,
            )
            nominal = np.array(
                [
                    nominal_horizontal[0],
                    nominal_horizontal[1],
                    rng.uniform(-VERTICAL_A_MAX, VERTICAL_A_MAX),
                ]
            )
            constraints = [
                sphere_hocbf_constraint(np.zeros(3), velocity, obstacle, hocbf_config)
                for obstacle in obstacles
            ]
            full_rows = np.vstack(
                (np.stack([item.row for item in constraints]), actuator_rows)
            )
            full_bounds = np.concatenate(
                (
                    np.asarray([item.lower_bound for item in constraints]),
                    actuator_bounds,
                )
            )
            full = project_polyhedral_qp(
                nominal,
                np.eye(3),
                full_rows,
                full_bounds,
            )
            full_feasible.append(float(full.feasible))
            for priority in priorities:
                selection_started = perf_counter()
                selected = select_top_k_constraints(
                    constraints,
                    nominal,
                    top_k,
                    priority=priority,
                    braking_acceleration=braking_acceleration,
                )
                selection_seconds = perf_counter() - selection_started
                rows = np.vstack(
                    (np.stack([item.row for item in selected]), actuator_rows)
                )
                bounds = np.concatenate(
                    (
                        np.asarray([item.lower_bound for item in selected]),
                        actuator_bounds,
                    )
                )
                solve_started = perf_counter()
                projected = project_polyhedral_qp(
                    nominal,
                    np.eye(3),
                    rows,
                    bounds,
                )
                solver_seconds = perf_counter() - solve_started
                selected_ids = {item.identifier for item in selected}
                omitted = [
                    item for item in constraints if item.identifier not in selected_ids
                ]
                omitted_slacks = np.asarray(
                    [item.slack(projected.acceleration) for item in omitted],
                    dtype=np.float64,
                )
                violations = int(np.sum(omitted_slacks < -1e-7))
                record = samples[priority]
                record["selection"].append(selection_seconds)
                record["solver"].append(solver_seconds)
                record["total"].append(selection_seconds + solver_seconds)
                record["omitted_violations"].append(violations)
                record["max_omitted_violation"].append(
                    float(max(0.0, -np.min(omitted_slacks)))
                    if omitted_slacks.size
                    else 0.0
                )
                record["action_deviation"].append(
                    float(np.linalg.norm(projected.acceleration - nominal))
                )
                record["feasible"].append(float(projected.feasible))
                if full.feasible and projected.feasible:
                    record["full_action_difference"].append(
                        float(
                            np.linalg.norm(
                                projected.acceleration - full.acceleration
                            )
                        )
                    )
        for priority in priorities:
            record = samples[priority]
            omitted = np.asarray(record["omitted_violations"], dtype=np.float64)
            total = np.asarray(record["total"], dtype=np.float64)
            selection = np.asarray(record["selection"], dtype=np.float64)
            solver = np.asarray(record["solver"], dtype=np.float64)
            action_difference = np.asarray(
                record["full_action_difference"],
                dtype=np.float64,
            )
            output.append(
                {
                    "constraints": count,
                    "top_k": min(top_k, count),
                    "priority": priority,
                    "trials": trials,
                    "full_feasible_rate": float(np.mean(full_feasible)),
                    "selected_feasible_rate": float(np.mean(record["feasible"])),
                    "omitted_constraint_violation_trial_rate": float(
                        np.mean(omitted > 0)
                    ),
                    "mean_omitted_constraint_violations": float(np.mean(omitted)),
                    "worst_omitted_constraint_violation": float(
                        np.max(record["max_omitted_violation"])
                    ),
                    "mean_action_deviation": float(
                        np.mean(record["action_deviation"])
                    ),
                    "mean_action_difference_from_full": (
                        float(np.mean(action_difference))
                        if action_difference.size
                        else None
                    ),
                    "paired_full_feasible_trials": int(action_difference.size),
                    "mean_selection_ms": 1000.0 * float(np.mean(selection)),
                    "mean_solver_ms": 1000.0 * float(np.mean(solver)),
                    "p95_total_ms": 1000.0 * float(np.quantile(total, 0.95)),
                    "p99_total_ms": 1000.0 * float(np.quantile(total, 0.99)),
                }
            )
    return output
