from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.memory_safety.estimation_benchmark import (
    DiagonalSSMEstimator,
    FlattenMLP,
    PhysicsResidualEstimator,
    RecurrentEstimator,
    TemporalConvEstimator,
    _features,
    _imm_estimate,
    _kalman_estimate,
    _mhe_estimate,
)
from experiments.uav_safety_filter.benchmark import (
    GOAL_RADIUS,
    HORIZONTAL_A_MAX,
    HORIZONTAL_V_MAX,
    SAFETY_DT,
    UAV_RADIUS,
    VERTICAL_A_MAX,
    VERTICAL_V_MAX,
    normalized_action_to_acceleration,
    sac_observation,
)
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostModel
from review_bundle.safety.collision.hocbf import (
    HOCBFConfig,
    actuator_polygon_constraints,
    emergency_braking_acceleration,
    energy_aware_quadratic,
    project_polyhedral_qp,
    strengthen_constraint_for_sample_hold,
    velocity_polygon_constraints,
)
from review_bundle.safety.collision.feasibility import (
    exact_constant_jerk_sphere_clearance,
)
from review_bundle.safety.memory import (
    IntervalObserverConfig,
    IntervalObserverState,
    PhysicsMemoryObserver,
    TrustedBaseIntervalCertificate,
    directional_hocbf_preconditions_hold,
    directional_hocbf_sample_hold_margin,
    directional_interval_hocbf_constraint,
)
from scripts.train_uav_energy_mc import load_frozen_sac


METHODS = (
    "A_exact_candidate_a",
    "B_current_only",
    "C_CA_Kalman",
    "D_IMM",
    "E_ego_L16_MLP",
    "F_GRU",
    "G_Physics_GRU",
    "H_Contractive_Physics_Memory",
    "I_analytic_interval_observer",
)

NUMERICAL_CERTIFICATION_RESERVE = 1e-6
SCENARIO_OBSTACLE_ACCELERATION_LIMIT = 6.0


@dataclass(frozen=True)
class ClosedLoopConfig:
    seed: int = 20260819
    scenarios: int = 12
    max_steps: int = 800
    sensor_period_steps: int = 2
    sensor_error_bound: float = 0.10
    sensor_range: float = 200.0
    dropout_probability: float = 0.05
    obstacle_jerk_bound: float = 12.0
    obstacle_acceleration_bound: float = SCENARIO_OBSTACLE_ACCELERATION_LIMIT
    obstacle_radius: float = 4.0
    history_length: int = 16
    sac_checkpoint: str = (
        "artifacts/uav_energy_delivery_v3_formal_20260816_004619/"
        "phase1_navigation/checkpoint_transition_500000.zip"
    )
    estimation_artifact: str = "artifacts/memory_estimation_5k_20260819_v2"
    output_dir: str = "artifacts/memory_closed_loop_controlled"

    def __post_init__(self) -> None:
        if self.scenarios * len(METHODS) * self.max_steps > 100_000:
            raise ValueError("closed-loop protocol exceeds the 100k transition ceiling")
        if self.sensor_period_steps <= 0 or self.max_steps <= 0:
            raise ValueError("step counts must be positive")
        if self.sensor_range <= 0.0:
            raise ValueError("sensor_range must be positive")
        if self.obstacle_jerk_bound < 12.0:
            raise ValueError(
                "the pre-registered maneuver families require obstacle_jerk_bound >= 12"
            )
        if (
            not np.isfinite(self.obstacle_acceleration_bound)
            or self.obstacle_acceleration_bound
            < SCENARIO_OBSTACLE_ACCELERATION_LIMIT
        ):
            raise ValueError(
                "obstacle_acceleration_bound must cover the 6 m/s^2 scenario limit"
            )


@dataclass(frozen=True)
class DynamicScenario:
    identifier: str
    family: str
    start: np.ndarray
    goal: np.ndarray
    obstacle_position: np.ndarray
    obstacle_velocity: np.ndarray
    obstacle_acceleration: np.ndarray
    crossing_fraction: float
    crossing_step: int
    maneuver_step: int
    maneuver_axis: np.ndarray


def _git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_hash() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _executed_source_hashes() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path("review_bundle/safety/memory/observer.py"),
        Path("review_bundle/safety/memory/networks.py"),
        Path("review_bundle/safety/memory/robust_hocbf.py"),
        Path("review_bundle/safety/memory/tube.py"),
        Path("review_bundle/safety/collision/feasibility.py"),
        Path("review_bundle/safety/collision/hocbf.py"),
        Path("review_bundle/safety/collision/filter.py"),
        Path("experiments/memory_safety/estimation_benchmark.py"),
        Path("experiments/uav_safety_filter/benchmark.py"),
        Path("scripts/train_uav_energy_mc.py"),
        Path("review_bundle/envs/navigation/state.py"),
        Path("review_bundle/envs/navigation/telemetry_cost.py"),
    )
    return {str(path): _sha256(path) for path in paths}


def make_scenarios(config: ClosedLoopConfig) -> list[DynamicScenario]:
    rng = np.random.default_rng(config.seed)
    families = (
        "constant_velocity",
        "bounded_acceleration",
        "sudden_maneuver",
        "stop_go",
    )
    scenarios = []
    for index in range(config.scenarios):
        family = families[index % len(families)]
        start = np.array([1000.0, 1000.0, 100.0])
        goal = np.array([1200.0, 1000.0, 100.0])
        crossing_x = rng.uniform(1060.0, 1140.0)
        crossing_time = rng.uniform(6.0, 11.0)
        lateral_speed = rng.uniform(5.0, 9.0) * rng.choice((-1.0, 1.0))
        obstacle_position = np.array(
            [crossing_x, 1000.0 - lateral_speed * crossing_time, 100.0]
        )
        obstacle_velocity = np.array([rng.uniform(-0.5, 0.5), lateral_speed, 0.0])
        obstacle_acceleration = np.zeros(3)
        maneuver_axis = rng.normal(size=3)
        maneuver_axis[2] *= 0.2
        maneuver_axis /= np.linalg.norm(maneuver_axis)
        scenarios.append(
            DynamicScenario(
                identifier=f"{family}_{index}",
                family=family,
                start=start,
                goal=goal,
                obstacle_position=obstacle_position,
                obstacle_velocity=obstacle_velocity,
                obstacle_acceleration=obstacle_acceleration,
                crossing_fraction=float(rng.uniform(0.35, 0.70)),
                crossing_step=-1,
                maneuver_step=int(rng.integers(80, 180)),
                maneuver_axis=maneuver_axis,
            )
        )
    return scenarios


def _nominal_reference(policy, scenario: DynamicScenario, max_steps: int) -> np.ndarray:
    position = scenario.start.copy()
    velocity = np.zeros(3)
    nominal = np.zeros(3)
    positions = [position.copy()]
    for step in range(max_steps):
        if step % 4 == 0:
            observation = sac_observation(position, velocity, scenario.goal)
            action, _ = policy.predict(observation, deterministic=True)
            nominal = normalized_action_to_acceleration(action)
        proposed_velocity = velocity + nominal * SAFETY_DT
        horizontal_speed = float(np.linalg.norm(proposed_velocity[:2]))
        if horizontal_speed > HORIZONTAL_V_MAX:
            proposed_velocity[:2] *= HORIZONTAL_V_MAX / horizontal_speed
        proposed_velocity[2] = np.clip(
            proposed_velocity[2], -VERTICAL_V_MAX, VERTICAL_V_MAX
        )
        realized_acceleration = (proposed_velocity - velocity) / SAFETY_DT
        position = (
            position
            + velocity * SAFETY_DT
            + 0.5 * realized_acceleration * SAFETY_DT**2
        )
        velocity = proposed_velocity
        positions.append(position.copy())
        if np.linalg.norm(position - scenario.goal) <= GOAL_RADIUS:
            break
    return np.asarray(positions)


def _obstacle_displacement(scenario: DynamicScenario, steps: int) -> np.ndarray:
    position = np.zeros(3)
    velocity = scenario.obstacle_velocity.copy()
    acceleration = scenario.obstacle_acceleration.copy()
    for step in range(steps):
        jerk = _jerk(scenario, step, acceleration)
        position, velocity, acceleration, _ = _advance_obstacle_state(
            position,
            velocity,
            acceleration,
            jerk,
            SAFETY_DT,
        )
    return position


def _advance_obstacle_state(
    position: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    commanded_jerk: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance the obstacle with constant realized jerk over one hold."""

    next_acceleration = np.clip(
        acceleration + commanded_jerk * dt,
        -SCENARIO_OBSTACLE_ACCELERATION_LIMIT,
        SCENARIO_OBSTACLE_ACCELERATION_LIMIT,
    )
    realized_jerk = (next_acceleration - acceleration) / dt
    next_position = (
        position
        + velocity * dt
        + 0.5 * acceleration * dt**2
        + realized_jerk * dt**3 / 6.0
    )
    next_velocity = (
        velocity
        + acceleration * dt
        + 0.5 * realized_jerk * dt**2
    )
    return next_position, next_velocity, next_acceleration, realized_jerk


def _finite_horizon_reset_velocity_bound(
    scenario: DynamicScenario,
    config: ClosedLoopConfig,
) -> np.ndarray:
    """Componentwise velocity envelope from initial speed and bounded acceleration."""

    horizon = config.max_steps * SAFETY_DT
    return (
        np.abs(scenario.obstacle_velocity)
        + config.obstacle_acceleration_bound * horizon
    )


def align_scenarios_to_nominal_path(
    policy,
    scenarios: list[DynamicScenario],
    max_steps: int,
) -> list[DynamicScenario]:
    """Make each obstacle cross the no-safety frozen-policy trajectory."""

    if not scenarios:
        return []
    reference = _nominal_reference(policy, scenarios[0], max_steps)
    final_index = len(reference) - 1
    if final_index < 20:
        raise ValueError("nominal reference is too short for a crossing scenario")
    aligned = []
    for scenario in scenarios:
        crossing_step = int(np.clip(
            round(scenario.crossing_fraction * final_index),
            10,
            final_index - 1,
        ))
        displacement = _obstacle_displacement(scenario, crossing_step)
        aligned.append(
            replace(
                scenario,
                obstacle_position=reference[crossing_step] - displacement,
                crossing_step=crossing_step,
            )
        )
    return aligned


def _jerk(scenario: DynamicScenario, step: int, acceleration: np.ndarray) -> np.ndarray:
    if scenario.family == "constant_velocity":
        return -acceleration / SAFETY_DT
    if scenario.family == "bounded_acceleration":
        return 4.0 * np.sin(0.03 * step) * scenario.maneuver_axis
    if scenario.family == "sudden_maneuver":
        if scenario.maneuver_step <= step < scenario.maneuver_step + 20:
            return 12.0 * scenario.maneuver_axis
        if scenario.maneuver_step + 20 <= step < scenario.maneuver_step + 40:
            return -12.0 * scenario.maneuver_axis
        return np.zeros(3)
    if scenario.maneuver_step <= step < scenario.maneuver_step + 25:
        return -np.clip(acceleration + scenario.maneuver_axis * 4.0, -6.0, 6.0) / 1.25
    if scenario.maneuver_step + 25 <= step < scenario.maneuver_step + 50:
        return 6.0 * scenario.maneuver_axis
    return np.zeros(3)


def _scenario_index(scenario: DynamicScenario) -> int:
    return int(scenario.identifier.rsplit("_", 1)[1])


def _perception_draw(
    config: ClosedLoopConfig,
    scenario: DynamicScenario,
    sensor_update_index: int,
) -> tuple[np.ndarray, float]:
    """Return a method-independent matched sensor draw."""

    seed = np.random.SeedSequence(
        [config.seed, _scenario_index(scenario), sensor_update_index]
    )
    rng = np.random.default_rng(seed)
    noise = rng.uniform(
        -config.sensor_error_bound,
        config.sensor_error_bound,
        size=3,
    )
    return noise, float(rng.random())


def _probabilistic_state_update(
    previous: IntervalObserverState,
    measurement: np.ndarray | None,
    estimate_velocity: np.ndarray,
    estimate_acceleration: np.ndarray,
    velocity_radius: np.ndarray,
    acceleration_radius: np.ndarray,
    config: ClosedLoopConfig,
    *,
    dt: float,
) -> IntervalObserverState:
    """Update a marginally calibrated baseline without unobserved truth."""

    velocity_radius = np.asarray(velocity_radius, dtype=np.float64)
    acceleration_radius = np.asarray(acceleration_radius, dtype=np.float64)
    if measurement is not None:
        return IntervalObserverState(
            position=np.asarray(measurement, dtype=np.float64),
            velocity=estimate_velocity,
            acceleration=estimate_acceleration,
            position_radius=np.full(3, config.sensor_error_bound),
            velocity_radius=velocity_radius,
            acceleration_radius=acceleration_radius,
            track_id="probabilistic_baseline",
            certification_valid=False,
            invalid_reason="empirical_max_residual_box",
        )

    jerk_radius = np.full(3, config.obstacle_jerk_bound)
    return IntervalObserverState(
        position=(
            previous.position
            + previous.velocity * dt
            + 0.5 * previous.acceleration * dt**2
        ),
        velocity=estimate_velocity,
        acceleration=estimate_acceleration,
        position_radius=(
            previous.position_radius
            + dt * previous.velocity_radius
            + 0.5 * dt**2 * previous.acceleration_radius
            + (dt**3 / 6.0) * jerk_radius
        ),
        velocity_radius=np.maximum(
            velocity_radius,
            previous.velocity_radius
            + dt * previous.acceleration_radius
            + 0.5 * dt**2 * jerk_radius,
        ),
        acceleration_radius=np.maximum(
            acceleration_radius,
            previous.acceleration_radius + dt * jerk_radius,
        ),
        track_id="probabilistic_baseline",
        certification_valid=False,
        invalid_reason="empirical_max_residual_box",
        missed_frames=previous.missed_frames + 1,
    )


def _calibrated_state_radii(artifact: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    calibration = np.load(artifact / "validation.npz")
    prediction_file = np.load(artifact / "validation_predictions.npz")
    calibration_count = len(calibration["obstacle_velocity"])
    if calibration_count < 100:
        raise ValueError("dedicated state-radius calibration partition is too small")
    target = np.concatenate(
        (
            calibration["obstacle_velocity"][:calibration_count, -1],
            calibration["obstacle_acceleration"][:calibration_count, -1],
        ),
        axis=1,
    )
    mapping = {
        "B_current_only": "A_current_only",
        "C_CA_Kalman": "CA_Kalman",
        "D_IMM": "IMM",
        "E_ego_L16_MLP": "C_ego_L16_MLP",
        "F_GRU": "E_GRU",
        "G_Physics_GRU": "I_Physics_GRU",
        "H_Contractive_Physics_Memory": "M_Contractive_Physics_Memory",
    }
    result = {}
    for method, source in mapping.items():
        error = target - prediction_file[source][:calibration_count]
        radius = np.max(np.abs(error), axis=0)
        result[method] = (radius[:3], radius[3:])
    return result


def _load_models(artifact: Path, device: torch.device) -> dict[str, tuple[torch.nn.Module, dict]]:
    specs = {
        "E_ego_L16_MLP": ("C_ego_L16_MLP", FlattenMLP(16, 32)),
        "F_GRU": ("E_GRU", RecurrentEstimator("gru", 32)),
        "G_Physics_GRU": ("I_Physics_GRU", PhysicsResidualEstimator(32, contractive=False)),
        "H_Contractive_Physics_Memory": (
            "M_Contractive_Physics_Memory",
            PhysicsResidualEstimator(32, contractive=True),
        ),
    }
    loaded = {}
    for method, (source, model) in specs.items():
        checkpoint = torch.load(artifact / "models" / f"{source}.pt", map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device).eval()
        loaded[method] = (model, checkpoint["training"])
    return loaded


def _history_split(
    relative_history: list[np.ndarray],
    ego_history: list[np.ndarray],
    valid_history: list[bool],
    length: int,
) -> dict[str, np.ndarray]:
    start = max(0, len(relative_history) - length)
    relative = [item.copy() for item in relative_history[start:]]
    ego = [item.copy() for item in ego_history[start:]]
    valid = list(valid_history[start:])
    carry = None
    for index in range(start - 1, -1, -1):
        if valid_history[index]:
            carry = relative_history[index].copy()
            break
    if carry is None:
        carry = next(
            (value.copy() for value, is_valid in zip(relative, valid) if is_valid),
            None,
        )
    if carry is None:
        raise ValueError("history requires at least one valid measurement")
    for index, is_valid in enumerate(valid):
        if is_valid:
            carry = relative[index].copy()
        else:
            relative[index] = carry.copy()
    while len(relative) < length:
        relative.insert(0, relative[0].copy())
        ego.insert(0, ego[0].copy())
        valid.insert(0, False)
    relative_array = np.asarray(relative, dtype=np.float32)[None, ...]
    valid_array = np.asarray(valid, dtype=bool)[None, ...]
    return {
        "relative_measurement": relative_array,
        "ego_position": np.asarray(ego, dtype=np.float32)[None, ...],
        "valid": valid_array,
    }


def _model_prediction(
    method: str,
    split: dict[str, np.ndarray],
    models: dict[str, tuple[torch.nn.Module, dict]],
    device: torch.device,
    fallback_velocity: np.ndarray,
    fallback_acceleration: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "B_current_only":
        return np.zeros(3), np.zeros(3)
    if not np.any(split["valid"]):
        return fallback_velocity.copy(), fallback_acceleration.copy()
    if method == "C_CA_Kalman":
        prediction = _kalman_estimate(split, SAFETY_DT * 2, constant_acceleration=True)[0]
    elif method == "D_IMM":
        prediction = _imm_estimate(split, SAFETY_DT * 2)[0]
    else:
        model, training = models[method]
        features = _features(split, ego_compensated=True)
        mean = np.asarray(training["target_mean"], dtype=np.float32)
        scale = np.asarray(training["target_scale"], dtype=np.float32)
        if method in {"G_Physics_GRU", "H_Contractive_Physics_Memory"}:
            base = _mhe_estimate(split, SAFETY_DT * 2)[0]
        else:
            base = np.zeros(6, dtype=np.float32)
        with torch.no_grad():
            output = model(
                torch.from_numpy(features).to(device),
                torch.from_numpy(((base - mean) / scale)[None, :]).to(device),
            )[0]
        prediction = output.cpu().numpy() * scale + mean
    return prediction[:3].astype(np.float64), prediction[3:].astype(np.float64)


def _robust_filter(
    position: np.ndarray,
    velocity: np.ndarray,
    nominal: np.ndarray,
    obstacle: IntervalObserverState,
    obstacle_radius: float,
    jerk_bound: float,
    telemetry: TelemetryCostModel,
    *,
    allow_uncertified_interval: bool,
) -> tuple[np.ndarray, bool, bool, float, bool, float, float, float]:
    started = perf_counter()
    if not obstacle.certification_valid and not allow_uncertified_interval:
        action = emergency_braking_acceleration(
            velocity,
            HORIZONTAL_A_MAX,
            VERTICAL_A_MAX,
        )
        return (
            action,
            False,
            False,
            float(np.linalg.norm(action - nominal)),
            True,
            perf_counter() - started,
            0.0,
            0.0,
        )
    tube_started = perf_counter()
    hocbf = HOCBFConfig(k1=1.0, k2=1.0, uav_radius=UAV_RADIUS)
    constraint = directional_interval_hocbf_constraint(
        position,
        velocity,
        obstacle,
        obstacle_radius,
        hocbf,
        allow_uncertified_interval=allow_uncertified_interval,
    )
    margin = directional_hocbf_sample_hold_margin(
        velocity,
        obstacle,
        constraint.row,
        hocbf,
        hold_dt=SAFETY_DT,
        horizontal_acceleration_limit=HORIZONTAL_A_MAX,
        vertical_acceleration_limit=VERTICAL_A_MAX,
        true_obstacle_jerk_bound=jerk_bound,
        allow_uncertified_interval=allow_uncertified_interval,
    )
    constraint = strengthen_constraint_for_sample_hold(constraint, margin)
    actuator_rows, actuator_bounds = actuator_polygon_constraints(
        HORIZONTAL_A_MAX,
        VERTICAL_A_MAX,
    )
    velocity_rows, velocity_bounds = velocity_polygon_constraints(
        velocity,
        dt=SAFETY_DT,
        horizontal_limit=HORIZONTAL_V_MAX,
        vertical_limit=VERTICAL_V_MAX,
    )
    rows = np.vstack((constraint.row[None, :], actuator_rows, velocity_rows))
    bounds = np.concatenate(([constraint.lower_bound], actuator_bounds, velocity_bounds))
    safety_intervention = bool(constraint.row @ nominal < constraint.lower_bound - 1e-8)
    action_scale = float(np.hypot(HORIZONTAL_A_MAX, VERTICAL_A_MAX))
    energy_matrix = np.diag(telemetry.config.acceleration_coefficients)
    energy_characteristic = SAFETY_DT * (
        telemetry.config.acceleration_coefficients[0] * HORIZONTAL_A_MAX**2
        + telemetry.config.acceleration_coefficients[2] * VERTICAL_A_MAX**2
    )
    hessian, center = energy_aware_quadratic(
        nominal,
        np.eye(3) / action_scale**2,
        energy_matrix / max(energy_characteristic, 1e-12),
        energy_weight=1.0,
        dt=SAFETY_DT,
    )
    tube_seconds = perf_counter() - tube_started
    solver_bounds = bounds + NUMERICAL_CERTIFICATION_RESERVE
    qp_started = perf_counter()
    result = project_polyhedral_qp(
        center,
        hessian,
        rows,
        solver_bounds,
        tolerance=1e-9,
    )
    qp_seconds = perf_counter() - qp_started
    action = result.acceleration
    numerical_feasible, minimum_original_slack = _projection_is_certifiable(
        result,
        rows,
        bounds,
    )
    certified = bool(
        obstacle.certification_valid
        and not allow_uncertified_interval
        and directional_hocbf_preconditions_hold(constraint)
        and numerical_feasible
    )
    if not numerical_feasible:
        action = emergency_braking_acceleration(
            velocity,
            HORIZONTAL_A_MAX,
            VERTICAL_A_MAX,
        )
        certified = False
    return (
        action,
        numerical_feasible,
        certified,
        float(np.linalg.norm(action - nominal)),
        safety_intervention,
        perf_counter() - started,
        tube_seconds,
        qp_seconds,
    )


def _projection_is_certifiable(
    result,
    rows: np.ndarray,
    original_bounds: np.ndarray,
) -> tuple[bool, float]:
    """Require convergence and a positive post-solve slack in original constraints."""

    minimum_original_slack = float(
        np.min(np.asarray(rows) @ result.acceleration - np.asarray(original_bounds))
    )
    return (
        bool(
            result.feasible
            and result.converged
            and minimum_original_slack
            >= 0.5 * NUMERICAL_CERTIFICATION_RESERVE
        ),
        minimum_original_slack,
    )


def _apply_velocity_limits(
    velocity: np.ndarray,
    commanded_acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Apply simulator speed limits and expose any actuation-model mismatch."""

    velocity = np.asarray(velocity, dtype=np.float64)
    commanded_acceleration = np.asarray(commanded_acceleration, dtype=np.float64)
    proposed_velocity = velocity + commanded_acceleration * dt
    horizontal_speed = float(np.linalg.norm(proposed_velocity[:2]))
    if horizontal_speed > HORIZONTAL_V_MAX:
        proposed_velocity[:2] *= HORIZONTAL_V_MAX / horizontal_speed
    proposed_velocity[2] = np.clip(
        proposed_velocity[2],
        -VERTICAL_V_MAX,
        VERTICAL_V_MAX,
    )
    realized_acceleration = (proposed_velocity - velocity) / dt
    actuation_matches_model = bool(
        np.all(np.isfinite(realized_acceleration))
        and np.allclose(
            realized_acceleration,
            commanded_acceleration,
            atol=1e-8,
            rtol=0.0,
        )
    )
    return proposed_velocity, realized_acceleration, actuation_matches_model


def run_rollout(
    policy,
    scenario: DynamicScenario,
    method: str,
    config: ClosedLoopConfig,
    models: dict[str, tuple[torch.nn.Module, dict]],
    radii: dict[str, tuple[np.ndarray, np.ndarray]],
    device: torch.device,
) -> dict[str, object]:
    position = scenario.start.copy()
    velocity = np.zeros(3)
    obstacle_position = scenario.obstacle_position.copy()
    obstacle_velocity = scenario.obstacle_velocity.copy()
    obstacle_acceleration = scenario.obstacle_acceleration.copy()
    maximum_obstacle_velocity_component = float(np.max(np.abs(obstacle_velocity)))
    reset_bound_violations = 0
    telemetry = TelemetryCostModel()
    reset_velocity_bound = _finite_horizon_reset_velocity_bound(scenario, config)
    analytic = PhysicsMemoryObserver(
        IntervalObserverConfig(
            dt=SAFETY_DT,
            sensor_error_bound=config.sensor_error_bound,
            jerk_residual_bound=config.obstacle_jerk_bound,
            reset_velocity_bound=reset_velocity_bound,
            reset_acceleration_bound=config.obstacle_acceleration_bound,
        )
    )
    analytic_base_certificate = TrustedBaseIntervalCertificate.from_config(
        analytic.config,
        track_id=scenario.identifier,
        provenance="controlled-simulation declared motion and sensor bounds",
        measurement_epoch=0,
    )
    relative_history: list[np.ndarray] = []
    ego_history: list[np.ndarray] = []
    valid_history: list[bool] = []
    initial_noise, _ = _perception_draw(config, scenario, 0)
    first_measurement = obstacle_position + initial_noise
    relative_history.append(first_measurement - position)
    ego_history.append(position.copy())
    valid_history.append(True)
    analytic.initialize(
        first_measurement,
        trusted_base_certificate=analytic_base_certificate,
    )
    nominal = np.zeros(3)
    estimate_velocity = np.zeros(3)
    estimate_acceleration = np.zeros(3)
    velocity_radius, acceleration_radius = radii.get(
        method,
        (np.full(3, 12.0), np.full(3, 8.0)),
    )
    estimate_state = IntervalObserverState(
        position=first_measurement,
        velocity=np.zeros(3),
        acceleration=np.zeros(3),
        position_radius=np.full(3, config.sensor_error_bound),
        velocity_radius=np.full(3, 12.0),
        acceleration_radius=np.full(3, 8.0),
        track_id="probabilistic_baseline",
        certification_valid=False,
        invalid_reason="empirical_max_residual_box",
    )
    if method == "I_analytic_interval_observer":
        estimate_state = analytic.state
    collision_steps = 0
    minimum_clearance = float("inf")
    interventions = []
    safety_interventions = []
    filter_times = []
    memory_update_times = []
    tube_times = []
    qp_times = []
    uncertified_steps = 0
    actuation_mismatch_steps = 0
    infeasible_steps = 0
    resets = 0
    dropout_steps = 0
    realized_energy = 0.0
    path_length = 0.0
    freeze_steps = 0
    interval_underbound_steps = 0
    success = False
    for step in range(config.max_steps):
        step_compute_started = perf_counter()
        if step % 4 == 0:
            observation = sac_observation(position, velocity, scenario.goal)
            action, _ = policy.predict(observation, deterministic=True)
            nominal = normalized_action_to_acceleration(action)
        memory_update_started = perf_counter()
        measurement = None
        sensor_update = step > 0 and step % config.sensor_period_steps == 0
        if sensor_update:
            distance = float(np.linalg.norm(obstacle_position - position))
            noise, dropout_draw = _perception_draw(
                config,
                scenario,
                step // config.sensor_period_steps + 1,
            )
            observed = (
                distance <= config.sensor_range
                and dropout_draw >= config.dropout_probability
            )
            if observed:
                measurement = obstacle_position + noise
                relative_history.append(measurement - position)
                valid_history.append(True)
            else:
                measurement = None
                relative_history.append(np.full(3, np.nan))
                valid_history.append(False)
                dropout_steps += 1
            ego_history.append(position.copy())
            split = _history_split(
                relative_history,
                ego_history,
                valid_history,
                config.history_length,
            )
            if method not in {"A_exact_candidate_a", "I_analytic_interval_observer"}:
                estimate_velocity, estimate_acceleration = _model_prediction(
                    method,
                    split,
                    models,
                    device,
                    estimate_state.velocity,
                    estimate_state.acceleration,
                )
                velocity_radius, acceleration_radius = radii[method]
        if step > 0 and method == "I_analytic_interval_observer":
            fresh_certificate = (
                None
                if measurement is None
                else TrustedBaseIntervalCertificate.from_config(
                    analytic.config,
                    track_id=scenario.identifier,
                    provenance=(
                        "controlled-simulation fresh measurement and declared bounds"
                    ),
                    measurement_epoch=step,
                )
            )
            observer_step = analytic.step(
                measurement,
                track_id=scenario.identifier if measurement is not None else None,
                trusted_base_certificate=fresh_certificate,
            )
            estimate_state = observer_step.state
            resets += int(observer_step.reset)
        elif step > 0 and method != "A_exact_candidate_a":
            estimate_state = _probabilistic_state_update(
                estimate_state,
                measurement,
                estimate_velocity,
                estimate_acceleration,
                velocity_radius,
                acceleration_radius,
                config,
                dt=SAFETY_DT,
            )
        memory_update_times.append(perf_counter() - memory_update_started)
        if method != "A_exact_candidate_a":
            interval_contains_truth = bool(
                np.all(
                    np.abs(obstacle_position - estimate_state.position)
                    <= estimate_state.position_radius + 1e-12
                )
                and np.all(
                    np.abs(obstacle_velocity - estimate_state.velocity)
                    <= estimate_state.velocity_radius + 1e-12
                )
                and np.all(
                    np.abs(obstacle_acceleration - estimate_state.acceleration)
                    <= estimate_state.acceleration_radius + 1e-12
                )
            )
            interval_underbound_steps += int(not interval_contains_truth)
        if method == "A_exact_candidate_a":
            exact_state = IntervalObserverState(
                position=obstacle_position,
                velocity=obstacle_velocity,
                acceleration=obstacle_acceleration,
                position_radius=np.zeros(3),
                velocity_radius=np.zeros(3),
                acceleration_radius=np.zeros(3),
                track_id=scenario.identifier,
                certification_valid=True,
                invalid_reason=None,
                certificate_epoch=0,
            )
            (
                executed,
                feasible,
                certified,
                intervention,
                safety_intervention,
                filter_seconds,
                tube_seconds,
                qp_seconds,
            ) = _robust_filter(
                position,
                velocity,
                nominal,
                exact_state,
                config.obstacle_radius,
                config.obstacle_jerk_bound,
                telemetry,
                allow_uncertified_interval=False,
            )
        else:
            (
                executed,
                feasible,
                certified,
                intervention,
                safety_intervention,
                filter_seconds,
                tube_seconds,
                qp_seconds,
            ) = _robust_filter(
                position,
                velocity,
                nominal,
                estimate_state,
                config.obstacle_radius,
                config.obstacle_jerk_bound,
                telemetry,
                allow_uncertified_interval=(
                    method != "I_analytic_interval_observer"
                ),
            )
        interventions.append(intervention)
        safety_interventions.append(safety_intervention)
        filter_times.append(perf_counter() - step_compute_started)
        tube_times.append(tube_seconds)
        qp_times.append(qp_seconds)
        infeasible_steps += int(not feasible)
        jerk = _jerk(scenario, step, obstacle_acceleration)
        obstacle_acceleration_during_hold = obstacle_acceleration.copy()
        (
            next_obstacle_position,
            next_obstacle_velocity,
            next_obstacle_acceleration,
            realized_obstacle_jerk,
        ) = _advance_obstacle_state(
            obstacle_position,
            obstacle_velocity,
            obstacle_acceleration_during_hold,
            jerk,
            SAFETY_DT,
        )
        (
            proposed_velocity,
            realized_acceleration,
            actuation_matches_model,
        ) = _apply_velocity_limits(
            velocity,
            executed,
            SAFETY_DT,
        )
        actuation_mismatch_steps += int(not actuation_matches_model)
        certified = bool(certified and actuation_matches_model)
        uncertified_steps += int(not certified)
        clearance = exact_constant_jerk_sphere_clearance(
            position - obstacle_position,
            velocity - obstacle_velocity,
            realized_acceleration - obstacle_acceleration_during_hold,
            -realized_obstacle_jerk,
            config.obstacle_radius + UAV_RADIUS,
            SAFETY_DT,
        )
        minimum_clearance = min(minimum_clearance, clearance.minimum_clearance)
        collision_steps += int(clearance.minimum_barrier <= 0.0)
        previous_position = position.copy()
        position = (
            position
            + velocity * SAFETY_DT
            + 0.5 * realized_acceleration * SAFETY_DT**2
        )
        velocity = proposed_velocity
        freeze_steps += int(
            np.linalg.norm(velocity) < 0.2
            and np.linalg.norm(position - scenario.goal) > GOAL_RADIUS
        )
        obstacle_position = next_obstacle_position
        obstacle_velocity = next_obstacle_velocity
        obstacle_acceleration = next_obstacle_acceleration
        maximum_obstacle_velocity_component = max(
            maximum_obstacle_velocity_component,
            float(np.max(np.abs(obstacle_velocity))),
        )
        reset_bound_violations += int(
            np.any(
                np.abs(obstacle_velocity)
                > analytic.config.reset_velocity_bound + 1e-10
            )
            or np.any(
                np.abs(obstacle_acceleration)
                > analytic.config.reset_acceleration_bound + 1e-10
            )
        )
        path_length += float(np.linalg.norm(position - previous_position))
        realized_energy += telemetry.realized_cost(
            NavigationState(position.copy(), velocity.copy(), 0.0, (step + 1) * SAFETY_DT),
            realized_acceleration,
            SAFETY_DT,
        )
        if np.linalg.norm(position - scenario.goal) <= GOAL_RADIUS:
            success = True
            steps = step + 1
            break
    else:
        steps = config.max_steps
    straight = float(np.linalg.norm(scenario.goal - scenario.start))
    return {
        "method": method,
        "scenario": scenario.identifier,
        "family": scenario.family,
        "success": success,
        "steps": steps,
        "collision_steps": collision_steps,
        "minimum_clearance": minimum_clearance,
        "infeasible_steps": infeasible_steps,
        "uncertified_steps": uncertified_steps,
        "actuation_mismatch_steps": actuation_mismatch_steps,
        "action_modification_fraction": float(
            np.mean(np.asarray(interventions) > 1e-6)
        ),
        "safety_intervention_fraction": float(np.mean(safety_interventions)),
        "mean_intervention": float(np.mean(interventions)),
        "freeze_fraction": freeze_steps / max(steps, 1),
        "realized_energy": realized_energy,
        "path_length": path_length,
        "path_ratio_at_end": path_length / max(straight, 1e-12),
        "final_goal_distance": float(np.linalg.norm(position - scenario.goal)),
        "p99_total_compute_ms": float(np.quantile(filter_times, 0.99) * 1000.0),
        "p99_memory_update_ms": float(
            np.quantile(memory_update_times, 0.99) * 1000.0
        ),
        "p99_tube_propagation_ms": float(np.quantile(tube_times, 0.99) * 1000.0),
        "p99_qp_ms": float(np.quantile(qp_times, 0.99) * 1000.0),
        "dropout_updates": dropout_steps,
        "observer_resets": resets,
        "interval_underbound_steps": interval_underbound_steps,
        "maximum_obstacle_velocity_component": maximum_obstacle_velocity_component,
        "maximum_declared_reset_velocity_bound": float(
            np.max(analytic.config.reset_velocity_bound)
        ),
        "reset_bound_violations": reset_bound_violations,
    }


def _aggregate_rows(selected: list[dict[str, object]]) -> dict[str, object]:
    total_steps = sum(int(row["steps"]) for row in selected)
    successful = [row for row in selected if row["success"]]
    return {
            "rollouts": len(selected),
            "success_rate": float(np.mean([row["success"] for row in selected])),
            "collision_rollout_rate": float(
                np.mean([int(row["collision_steps"]) > 0 for row in selected])
            ),
            "collision_step_rate": float(
                sum(int(row["collision_steps"]) for row in selected) / max(total_steps, 1)
            ),
            "minimum_clearance": float(min(row["minimum_clearance"] for row in selected)),
            "uncertified_step_rate": float(
                sum(int(row["uncertified_steps"]) for row in selected) / max(total_steps, 1)
            ),
            "actuation_mismatch_step_rate": float(
                sum(int(row["actuation_mismatch_steps"]) for row in selected)
                / max(total_steps, 1)
            ),
            "infeasible_step_rate": float(
                sum(int(row["infeasible_steps"]) for row in selected) / max(total_steps, 1)
            ),
            "interval_underbound_step_rate": float(
                sum(int(row["interval_underbound_steps"]) for row in selected)
                / max(total_steps, 1)
            ),
            "reset_bound_violation_rate": float(
                sum(int(row["reset_bound_violations"]) for row in selected)
                / max(total_steps, 1)
            ),
            "maximum_obstacle_velocity_component": float(
                max(row["maximum_obstacle_velocity_component"] for row in selected)
            ),
            "mean_action_modification_fraction": float(
                np.mean([row["action_modification_fraction"] for row in selected])
            ),
            "mean_safety_intervention_fraction": float(
                np.mean([row["safety_intervention_fraction"] for row in selected])
            ),
            "mean_intervention": float(
                np.mean([row["mean_intervention"] for row in selected])
            ),
            "mean_energy": float(np.mean([row["realized_energy"] for row in selected])),
            "mean_path_ratio_at_end_all_rollouts": float(
                np.mean([row["path_ratio_at_end"] for row in selected])
            ),
            "mean_success_path_ratio": (
                None
                if not successful
                else float(np.mean([row["path_ratio_at_end"] for row in successful]))
            ),
            "mean_success_energy": (
                None
                if not successful
                else float(np.mean([row["realized_energy"] for row in successful]))
            ),
            "mean_success_steps": (
                None
                if not successful
                else float(np.mean([row["steps"] for row in successful]))
            ),
            "mean_final_goal_distance": float(
                np.mean([row["final_goal_distance"] for row in selected])
            ),
            "mean_freeze_fraction": float(
                np.mean([row["freeze_fraction"] for row in selected])
            ),
            "worst_rollout_p99_compute_ms": float(
                max(row["p99_total_compute_ms"] for row in selected)
            ),
            "worst_rollout_p99_memory_update_ms": float(
                max(row["p99_memory_update_ms"] for row in selected)
            ),
            "worst_rollout_p99_tube_propagation_ms": float(
                max(row["p99_tube_propagation_ms"] for row in selected)
            ),
            "worst_rollout_p99_qp_ms": float(
                max(row["p99_qp_ms"] for row in selected)
            ),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary = {}
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        aggregate = _aggregate_rows(selected)
        aggregate["by_family"] = {
            family: _aggregate_rows(
                [row for row in selected if row["family"] == family]
            )
            for family in sorted({str(row["family"]) for row in selected})
        }
        summary[method] = aggregate
    return summary


def run(config: ClosedLoopConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_frozen_sac(Path(config.sac_checkpoint), str(device))
    artifact = Path(config.estimation_artifact)
    models = _load_models(artifact, device)
    radii = _calibrated_state_radii(artifact)
    scenarios = align_scenarios_to_nominal_path(
        policy,
        make_scenarios(config),
        config.max_steps,
    )
    (output / "scenarios.json").write_text(
        json.dumps(
            [
                {
                    **asdict(scenario),
                    "start": scenario.start.tolist(),
                    "goal": scenario.goal.tolist(),
                    "obstacle_position": scenario.obstacle_position.tolist(),
                    "obstacle_velocity": scenario.obstacle_velocity.tolist(),
                    "obstacle_acceleration": scenario.obstacle_acceleration.tolist(),
                    "maneuver_axis": scenario.maneuver_axis.tolist(),
                    "nominal_crossing_error": float(
                        np.linalg.norm(
                            scenario.obstacle_position
                            + _obstacle_displacement(
                                scenario, scenario.crossing_step
                            )
                            - _nominal_reference(
                                policy, scenario, config.max_steps
                            )[scenario.crossing_step]
                        )
                    ),
                }
                for scenario in scenarios
            ],
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    rows = [
        run_rollout(policy, scenario, method, config, models, radii, device)
        for scenario in scenarios
        for method in METHODS
    ]
    estimation_summary = json.loads((artifact / "summary.json").read_text())
    current_estimator_source_sha256 = _sha256(
        Path("experiments/memory_safety/estimation_benchmark.py")
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "git_sha": _git_sha(),
        "source_sha256": _source_hash(),
        "executed_source_sha256": _executed_source_hashes(),
        "input_sha256": {
            "sac_checkpoint": _sha256(Path(config.sac_checkpoint)),
            "estimation_summary": _sha256(artifact / "summary.json"),
            "development_calibration_data": _sha256(artifact / "validation.npz"),
            "development_calibration_predictions": _sha256(
                artifact / "validation_predictions.npz"
            ),
            "heldout_test_data": _sha256(artifact / "test.npz"),
            "heldout_test_predictions": _sha256(
                artifact / "test_predictions.npz"
            ),
            **{
                f"estimator:{path.name}": _sha256(path)
                for path in sorted((artifact / "models").glob("*.pt"))
            },
        },
        "device": str(device),
        "matched_scenarios": config.scenarios,
        "methods": list(METHODS),
        "maximum_possible_controlled_transitions": (
            config.scenarios * len(METHODS) * config.max_steps
        ),
        "actual_controlled_transitions": sum(int(row["steps"]) for row in rows),
        "summary": _summarize(rows),
        "formal_500k": False,
        "probabilistic_baseline_note": (
            "non-analytic estimator intervals use componentwise maximum residuals from the development validation split, which was also used for checkpoint selection; the held-out test split is not used to configure closed-loop radii; this development box has no conformal, distribution-free, conditional, or repeated-time coverage guarantee"
        ),
        "state_radius_calibration_partition": {
            "source": "validation.npz and validation_predictions.npz",
            "slice": "full_development_validation_split",
            "checkpoint_selection_reused_same_split": True,
            "heldout_test_used_for_radii": False,
            "artifact_estimator_source_sha256": estimation_summary.get(
                "source_sha256"
            ),
            "current_estimator_source_sha256": current_estimator_source_sha256,
            "artifact_source_matches_current_code": bool(
                estimation_summary.get("source_sha256")
                == current_estimator_source_sha256
            ),
            "recorded_split_seed_separation": estimation_summary.get(
                "split_seed_separation"
            ),
            "coverage_claim": "development maximum-residual box; no coverage guarantee",
        },
    }
    (output / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output / "config.json").write_text(
        json.dumps(asdict(config), indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary
