from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from time import perf_counter
from typing import Callable, Iterable

import numpy as np

from experiments.uav_safety_filter.benchmark import (
    HORIZONTAL_A_MAX,
    HORIZONTAL_V_MAX,
    LIDAR_PERIOD_STEPS,
    POLICY_PERIOD_STEPS,
    SAFETY_DT,
    UAV_RADIUS,
    VERTICAL_A_MAX,
    VERTICAL_V_MAX,
    normalized_action_to_acceleration,
    sac_observation,
)
from review_bundle.safety.collision.feasibility import (
    FeasibilityMarginResult,
    cylindrical_input_support,
    joint_feasibility_margin,
)
from review_bundle.safety.collision.filter import (
    SafetyFilterConfig,
    SafetyFilterMethod,
    UAVSafetyActionFilter,
)
from review_bundle.safety.collision.hocbf import (
    BarrierConstraint,
    HOCBFConfig,
    SphericalObstacle,
    actuator_polygon_constraints,
    hocbf_sampled_data_residual_bound,
    emergency_braking_acceleration,
    sphere_hocbf_constraint,
    strengthen_constraint_for_sample_hold,
    velocity_polygon_constraints,
)


class PerceptionMode(str, Enum):
    CURRENT_FRAME = "current_frame"
    HOLD_LAST = "hold_last"
    CONSTANT_VELOCITY = "constant_velocity"
    BOUNDED_ACCELERATION = "bounded_acceleration"


@dataclass(frozen=True)
class AccelerationEvent:
    step: int
    acceleration: np.ndarray

    def __post_init__(self) -> None:
        acceleration = np.asarray(self.acceleration, dtype=np.float64)
        if self.step < 0 or acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("invalid acceleration event")
        object.__setattr__(self, "acceleration", acceleration.copy())


@dataclass(frozen=True)
class DynamicObstacleSpec:
    identifier: str
    center: np.ndarray
    radius: float
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_events: tuple[AccelerationEvent, ...] = ()
    maximum_speed: float = 12.0
    maximum_acceleration: float = 4.0
    maximum_jerk: float = 8.0

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        velocity = np.asarray(self.velocity, dtype=np.float64)
        acceleration = np.asarray(self.acceleration, dtype=np.float64)
        jerk = np.asarray(self.jerk, dtype=np.float64)
        if any(value.shape != (3,) for value in (center, velocity, acceleration, jerk)):
            raise ValueError("dynamic obstacle vectors must have shape (3,)")
        if not all(np.all(np.isfinite(value)) for value in (center, velocity, acceleration, jerk)):
            raise ValueError("dynamic obstacle vectors must be finite")
        if self.radius <= 0.0:
            raise ValueError("obstacle radius must be positive")
        if min(self.maximum_speed, self.maximum_acceleration, self.maximum_jerk) <= 0.0:
            raise ValueError("motion bounds must be positive")
        if np.linalg.norm(acceleration) > self.maximum_acceleration + 1e-10:
            raise ValueError("initial obstacle acceleration exceeds its bound")
        if np.linalg.norm(jerk) > self.maximum_jerk + 1e-10:
            raise ValueError("obstacle jerk exceeds its bound")
        object.__setattr__(self, "center", center.copy())
        object.__setattr__(self, "velocity", velocity.copy())
        object.__setattr__(self, "acceleration", acceleration.copy())
        object.__setattr__(self, "jerk", jerk.copy())
        object.__setattr__(
            self,
            "acceleration_events",
            tuple(sorted(self.acceleration_events, key=lambda event: event.step)),
        )


@dataclass(frozen=True)
class VisibilityWindow:
    start_step: int
    end_step: int
    obstacle_identifiers: tuple[str, ...]
    cause: str = "dropout"

    def __post_init__(self) -> None:
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("visibility window must be a nonempty half-open interval")
        if not self.obstacle_identifiers:
            raise ValueError("visibility window must name an obstacle")

    def hides(self, step: int, identifier: str) -> bool:
        return (
            self.start_step <= step < self.end_step
            and identifier in self.obstacle_identifiers
        )


@dataclass(frozen=True)
class HardScenario:
    identifier: str
    family: str
    start: np.ndarray
    goal: np.ndarray
    initial_velocity: np.ndarray
    obstacles: tuple[DynamicObstacleSpec, ...]
    visibility_windows: tuple[VisibilityWindow, ...] = ()
    max_steps: int = 80
    world_lower: np.ndarray = field(default_factory=lambda: np.zeros(3))
    world_upper: np.ndarray = field(
        default_factory=lambda: np.array([4000.0, 4000.0, 400.0])
    )

    def __post_init__(self) -> None:
        vectors = {
            "start": np.asarray(self.start, dtype=np.float64),
            "goal": np.asarray(self.goal, dtype=np.float64),
            "initial_velocity": np.asarray(self.initial_velocity, dtype=np.float64),
            "world_lower": np.asarray(self.world_lower, dtype=np.float64),
            "world_upper": np.asarray(self.world_upper, dtype=np.float64),
        }
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors.values()):
            raise ValueError("scenario vectors must be finite with shape (3,)")
        if self.max_steps <= 1 or np.any(vectors["world_upper"] <= vectors["world_lower"]):
            raise ValueError("invalid scenario horizon or world bounds")
        if not self.obstacles:
            raise ValueError("hard scenarios require at least one obstacle")
        for name, value in vectors.items():
            object.__setattr__(self, name, value.copy())


@dataclass(frozen=True)
class RecoverabilityDiagnosticConfig:
    safety_dt: float = SAFETY_DT
    policy_period_steps: int = POLICY_PERIOD_STEPS
    sensing_period_steps: int = LIDAR_PERIOD_STEPS
    horizontal_acceleration_limit: float = HORIZONTAL_A_MAX
    vertical_acceleration_limit: float = VERTICAL_A_MAX
    horizontal_velocity_limit: float = HORIZONTAL_V_MAX
    vertical_velocity_limit: float = VERTICAL_V_MAX
    k1: float = 1.0
    k2: float = 1.0
    uav_radius: float = UAV_RADIUS
    sensing_error_bound: float = 0.0
    lookahead_steps: int = 12
    positive_margin_threshold: float = 1e-6

    def __post_init__(self) -> None:
        positive = (
            self.safety_dt,
            self.policy_period_steps,
            self.sensing_period_steps,
            self.horizontal_acceleration_limit,
            self.vertical_acceleration_limit,
            self.horizontal_velocity_limit,
            self.vertical_velocity_limit,
            self.k1,
            self.k2,
            self.lookahead_steps,
        )
        if any(value <= 0 for value in positive) or self.uav_radius < 0.0:
            raise ValueError("diagnostic timing, bounds, and gains must be positive")
        if self.sensing_error_bound < 0.0:
            raise ValueError("sensing error bound must be nonnegative")


@dataclass(frozen=True)
class PointwiseMargin:
    rho: float | None
    unconstrained: bool
    feasible: bool
    constraints: tuple[BarrierConstraint, ...]
    result: FeasibilityMarginResult | None


@dataclass(frozen=True)
class StepRecord:
    step: int
    time: float
    position: list[float]
    velocity: list[float]
    nominal_action: list[float]
    executed_action: list[float]
    true_obstacles: list[dict[str, object]]
    perceived_obstacles: list[dict[str, object]]
    visible_obstacles: list[str]
    stale_obstacle_ages: dict[str, float]
    rho_true: float | None
    rho_true_unstrengthened: float | None
    rho_perceived: float | None
    true_feasible: bool
    perceived_feasible: bool
    filter_feasible: bool
    fallback_used: bool
    fallback_satisfies_constraints: bool
    minimum_ttc: float | None
    minimum_clearance: float
    required_braking: float
    available_braking: float
    relative_velocities: dict[str, list[float]]
    active_constraint_identifiers: list[str]
    dual_weights: list[float]
    boundary_contact: bool
    filter_seconds: float
    rho_seconds: float
    hocbf_rows: list[list[float]]
    hocbf_bounds: list[float]
    hocbf_constraint_identifiers: list[str]
    actuator_rows: list[list[float]]
    actuator_bounds: list[float]
    velocity_rows: list[list[float]]
    velocity_bounds: list[float]
    margin_maximizing_action: list[float] | None


@dataclass(frozen=True)
class CounterexampleRecord:
    scenario_id: str
    family: str
    perception_mode: str
    feasible_step: int
    infeasible_step: int
    horizon: int
    rho_t: float
    rho_future: list[float | None]
    cause: str
    obstacle_constraint_disappeared: bool
    obstacle_reappeared: bool
    sensing_delay_present: bool
    braking_authority_lost: bool
    multi_obstacle_conflict: bool
    minimum_ttc: float | None
    minimum_clearance: float
    required_braking: float
    available_braking: float
    relative_velocities: dict[str, list[float]]
    responsible_constraints: list[str]
    sampled_data_strengthening_caused_infeasibility: bool
    sampled_recovery_action_exists: bool
    best_sampled_next_rho: float
    best_sampled_action: list[float]
    sampled_recovery_search_seconds: float
    state: dict[str, object]


@dataclass
class _DynamicObstacle:
    spec: DynamicObstacleSpec
    center: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray

    @classmethod
    def from_spec(cls, spec: DynamicObstacleSpec) -> "_DynamicObstacle":
        return cls(spec, spec.center.copy(), spec.velocity.copy(), spec.acceleration.copy())

    def snapshot(self) -> SphericalObstacle:
        return SphericalObstacle(
            self.center,
            self.spec.radius,
            self.velocity,
            self.acceleration,
            self.spec.identifier,
        )

    def advance(self, step: int, dt: float) -> None:
        for event in self.spec.acceleration_events:
            if event.step == step:
                self.acceleration = event.acceleration.copy()
        self.acceleration = self.acceleration + self.spec.jerk * dt
        acceleration_norm = float(np.linalg.norm(self.acceleration))
        if acceleration_norm > self.spec.maximum_acceleration:
            self.acceleration *= self.spec.maximum_acceleration / acceleration_norm
        self.velocity = self.velocity + self.acceleration * dt
        speed = float(np.linalg.norm(self.velocity))
        if speed > self.spec.maximum_speed:
            self.velocity *= self.spec.maximum_speed / speed
        self.center = self.center + self.velocity * dt


@dataclass
class _Track:
    obstacle: SphericalObstacle
    last_seen_step: int
    acceleration_bound: float


def _constraint_set(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: Iterable[SphericalObstacle],
    config: RecoverabilityDiagnosticConfig,
    *,
    sampled_data: bool = True,
) -> tuple[BarrierConstraint, ...]:
    hocbf_config = HOCBFConfig(
        k1=config.k1,
        k2=config.k2,
        uav_radius=config.uav_radius,
    )
    constraints = []
    for obstacle in obstacles:
        raw = sphere_hocbf_constraint(position, velocity, obstacle, hocbf_config)
        if sampled_data:
            residual = hocbf_sampled_data_residual_bound(
                position,
                velocity,
                obstacle,
                hocbf_config,
                hold_dt=config.safety_dt,
                horizontal_acceleration_limit=config.horizontal_acceleration_limit,
                vertical_acceleration_limit=config.vertical_acceleration_limit,
            )
            raw = strengthen_constraint_for_sample_hold(raw, residual)
        constraints.append(raw)
    return tuple(constraints)


def compute_pointwise_margin(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: Iterable[SphericalObstacle],
    config: RecoverabilityDiagnosticConfig,
    *,
    sampled_data: bool = True,
) -> PointwiseMargin:
    constraints = _constraint_set(
        position,
        velocity,
        obstacles,
        config,
        sampled_data=sampled_data,
    )
    if not constraints:
        return PointwiseMargin(None, True, True, (), None)
    result = joint_feasibility_margin(
        np.stack([constraint.row for constraint in constraints]),
        np.asarray([constraint.lower_bound for constraint in constraints]),
        config.horizontal_acceleration_limit,
        config.vertical_acceleration_limit,
    )
    return PointwiseMargin(
        rho=float(result.margin),
        unconstrained=False,
        feasible=result.feasible,
        constraints=constraints,
        result=result,
    )


def _is_hidden(scenario: HardScenario, step: int, identifier: str) -> bool:
    return any(window.hides(step, identifier) for window in scenario.visibility_windows)


def _perceived_obstacles(
    tracks: dict[str, _Track],
    true_obstacles: tuple[SphericalObstacle, ...],
    visible: tuple[str, ...],
    step: int,
    mode: PerceptionMode,
    config: RecoverabilityDiagnosticConfig,
) -> tuple[tuple[SphericalObstacle, ...], dict[str, float]]:
    true_by_id = {obstacle.identifier: obstacle for obstacle in true_obstacles}
    for identifier in visible:
        source = true_by_id[identifier]
        tracks[identifier] = _Track(
            obstacle=source,
            last_seen_step=step,
            acceleration_bound=max(float(np.linalg.norm(source.acceleration)), 4.0),
        )
    perceived = []
    stale_ages: dict[str, float] = {}
    identifiers = visible if mode is PerceptionMode.CURRENT_FRAME else tuple(tracks)
    for identifier in identifiers:
        track = tracks[identifier]
        age = (step - track.last_seen_step) * config.safety_dt
        stale_ages[identifier] = float(age)
        source = track.obstacle
        if identifier in visible or mode is PerceptionMode.HOLD_LAST:
            center = source.center
            velocity = source.velocity
            radius = source.radius
        else:
            center = source.center + source.velocity * age
            velocity = source.velocity
            radius = source.radius
            if mode is PerceptionMode.BOUNDED_ACCELERATION:
                radius += (
                    config.sensing_error_bound
                    + 0.5 * track.acceleration_bound * age**2
                )
        perceived.append(
            SphericalObstacle(
                center,
                radius,
                velocity,
                np.zeros(3),
                identifier,
            )
        )
    return tuple(perceived), stale_ages


def _clip_velocity(velocity: np.ndarray, config: RecoverabilityDiagnosticConfig) -> np.ndarray:
    clipped = np.asarray(velocity, dtype=np.float64).copy()
    horizontal_speed = float(np.linalg.norm(clipped[:2]))
    if horizontal_speed > config.horizontal_velocity_limit:
        clipped[:2] *= config.horizontal_velocity_limit / horizontal_speed
    clipped[2] = np.clip(
        clipped[2],
        -config.vertical_velocity_limit,
        config.vertical_velocity_limit,
    )
    return clipped


def _ttc_and_braking(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: tuple[SphericalObstacle, ...],
    config: RecoverabilityDiagnosticConfig,
) -> tuple[float | None, float, float, float]:
    minimum_ttc = math.inf
    minimum_clearance = math.inf
    maximum_required = 0.0
    minimum_available = math.inf
    for obstacle in obstacles:
        relative_position = position - obstacle.center
        distance = float(np.linalg.norm(relative_position))
        safe_distance = obstacle.radius + config.uav_radius
        clearance = distance - safe_distance
        minimum_clearance = min(minimum_clearance, clearance)
        if distance <= 1e-12:
            closing_speed = math.inf
            unit = np.array([1.0, 0.0, 0.0])
        else:
            unit = relative_position / distance
            relative_velocity = velocity - obstacle.velocity
            closing_speed = max(0.0, -float(unit @ relative_velocity))
            a = float(relative_velocity @ relative_velocity)
            b = 2.0 * float(relative_position @ relative_velocity)
            c = float(relative_position @ relative_position - safe_distance**2)
            if a > 1e-12 and b < 0.0:
                discriminant = b * b - 4.0 * a * c
                if discriminant >= 0.0:
                    roots = [
                        (-b - math.sqrt(discriminant)) / (2.0 * a),
                        (-b + math.sqrt(discriminant)) / (2.0 * a),
                    ]
                    positive = [value for value in roots if value >= 0.0]
                    if positive:
                        minimum_ttc = min(minimum_ttc, min(positive))
        required = (
            closing_speed**2 / max(2.0 * max(clearance, 1e-9), 1e-9)
            if np.isfinite(closing_speed)
            else math.inf
        )
        available = cylindrical_input_support(
            unit,
            config.horizontal_acceleration_limit,
            config.vertical_acceleration_limit,
        )
        maximum_required = max(maximum_required, required)
        minimum_available = min(minimum_available, available)
    return (
        None if not np.isfinite(minimum_ttc) else float(minimum_ttc),
        float(minimum_clearance),
        float(maximum_required),
        float(minimum_available),
    )


def _obstacle_dict(obstacle: SphericalObstacle) -> dict[str, object]:
    return {
        "identifier": obstacle.identifier,
        "center": obstacle.center.tolist(),
        "radius": obstacle.radius,
        "velocity": obstacle.velocity.tolist(),
        "acceleration": obstacle.acceleration.tolist(),
    }


def _relative_velocities(
    velocity: np.ndarray,
    obstacles: tuple[SphericalObstacle, ...],
) -> dict[str, list[float]]:
    return {
        obstacle.identifier: (velocity - obstacle.velocity).tolist()
        for obstacle in obstacles
    }


def run_scenario(
    scenario: HardScenario,
    policy_action: Callable[[np.ndarray], np.ndarray],
    *,
    perception_mode: PerceptionMode = PerceptionMode.CURRENT_FRAME,
    config: RecoverabilityDiagnosticConfig | None = None,
) -> list[StepRecord]:
    cfg = RecoverabilityDiagnosticConfig() if config is None else config
    position = scenario.start.copy()
    velocity = scenario.initial_velocity.copy()
    dynamic_obstacles = [_DynamicObstacle.from_spec(spec) for spec in scenario.obstacles]
    tracks: dict[str, _Track] = {}
    perceived: tuple[SphericalObstacle, ...] = ()
    stale_ages: dict[str, float] = {}
    nominal = np.zeros(3, dtype=np.float64)
    records: list[StepRecord] = []
    action_filter = UAVSafetyActionFilter(
        SafetyFilterConfig(
            method=SafetyFilterMethod.SAMPLED_DATA_HOCBF,
            safety_dt=cfg.safety_dt,
            deadline_seconds=cfg.safety_dt,
            horizontal_acceleration_limit=cfg.horizontal_acceleration_limit,
            vertical_acceleration_limit=cfg.vertical_acceleration_limit,
            horizontal_velocity_limit=cfg.horizontal_velocity_limit,
            vertical_velocity_limit=cfg.vertical_velocity_limit,
            sampled_data_robust=True,
        ),
        HOCBFConfig(k1=cfg.k1, k2=cfg.k2, uav_radius=cfg.uav_radius),
    )
    actuator_rows, actuator_bounds = actuator_polygon_constraints(
        cfg.horizontal_acceleration_limit,
        cfg.vertical_acceleration_limit,
        action_filter.config.actuator_facets,
    )
    for step in range(scenario.max_steps):
        true_obstacles = tuple(obstacle.snapshot() for obstacle in dynamic_obstacles)
        if step % cfg.sensing_period_steps == 0:
            visible = tuple(
                obstacle.identifier
                for obstacle in true_obstacles
                if not _is_hidden(scenario, step, obstacle.identifier)
            )
            perceived, stale_ages = _perceived_obstacles(
                tracks,
                true_obstacles,
                visible,
                step,
                perception_mode,
                cfg,
            )
        else:
            visible = tuple(
                obstacle.identifier
                for obstacle in true_obstacles
                if not _is_hidden(scenario, step, obstacle.identifier)
            )
        if step % cfg.policy_period_steps == 0:
            nominal = normalized_action_to_acceleration(
                policy_action(sac_observation(position, velocity, scenario.goal))
            )
        rho_started = perf_counter()
        true_margin = compute_pointwise_margin(position, velocity, true_obstacles, cfg)
        true_unstrengthened_margin = (
            compute_pointwise_margin(
                position,
                velocity,
                true_obstacles,
                cfg,
                sampled_data=False,
            )
            if true_margin.rho is not None and true_margin.rho < 0.0
            else None
        )
        perceived_margin = compute_pointwise_margin(position, velocity, perceived, cfg)
        rho_seconds = perf_counter() - rho_started
        filter_started = perf_counter()
        filtered = action_filter.filter(position, velocity, nominal, perceived)
        filter_seconds = perf_counter() - filter_started
        ttc, clearance, required, available = _ttc_and_braking(
            position, velocity, true_obstacles, cfg
        )
        active_ids: list[str] = []
        dual_weights: list[float] = []
        if true_margin.result is not None:
            dual_weights = true_margin.result.dual_weights.tolist()
            active_ids = [
                true_margin.constraints[index].identifier
                for index in true_margin.result.active_constraints
            ]
        boundary_contact = bool(
            np.any(position <= scenario.world_lower + cfg.uav_radius)
            or np.any(position >= scenario.world_upper - cfg.uav_radius)
        )
        velocity_rows, velocity_bounds = velocity_polygon_constraints(
            velocity,
            dt=cfg.safety_dt,
            horizontal_limit=cfg.horizontal_velocity_limit,
            vertical_limit=cfg.vertical_velocity_limit,
            facets=action_filter.config.actuator_facets,
        )
        records.append(
            StepRecord(
                step=step,
                time=step * cfg.safety_dt,
                position=position.tolist(),
                velocity=velocity.tolist(),
                nominal_action=nominal.tolist(),
                executed_action=filtered.acceleration.tolist(),
                true_obstacles=[_obstacle_dict(obstacle) for obstacle in true_obstacles],
                perceived_obstacles=[_obstacle_dict(obstacle) for obstacle in perceived],
                visible_obstacles=list(visible),
                stale_obstacle_ages=dict(stale_ages),
                rho_true=true_margin.rho,
                rho_true_unstrengthened=(
                    None
                    if true_unstrengthened_margin is None
                    else true_unstrengthened_margin.rho
                ),
                rho_perceived=perceived_margin.rho,
                true_feasible=true_margin.feasible,
                perceived_feasible=perceived_margin.feasible,
                filter_feasible=filtered.diagnostics.feasible,
                fallback_used=filtered.diagnostics.fallback_used,
                fallback_satisfies_constraints=(
                    filtered.diagnostics.fallback_satisfies_constraints
                ),
                minimum_ttc=ttc,
                minimum_clearance=clearance,
                required_braking=required,
                available_braking=available,
                relative_velocities=_relative_velocities(velocity, true_obstacles),
                active_constraint_identifiers=active_ids,
                dual_weights=dual_weights,
                boundary_contact=boundary_contact,
                filter_seconds=filter_seconds,
                rho_seconds=rho_seconds,
                hocbf_rows=[constraint.row.tolist() for constraint in true_margin.constraints],
                hocbf_bounds=[constraint.lower_bound for constraint in true_margin.constraints],
                hocbf_constraint_identifiers=[
                    constraint.identifier for constraint in true_margin.constraints
                ],
                actuator_rows=actuator_rows.tolist(),
                actuator_bounds=actuator_bounds.tolist(),
                velocity_rows=velocity_rows.tolist(),
                velocity_bounds=velocity_bounds.tolist(),
                margin_maximizing_action=(
                    None
                    if true_margin.result is None
                    else true_margin.result.maximizing_action.tolist()
                ),
            )
        )
        velocity = _clip_velocity(
            velocity + filtered.acceleration * cfg.safety_dt,
            cfg,
        )
        position = position + velocity * cfg.safety_dt
        for obstacle in dynamic_obstacles:
            obstacle.advance(step + 1, cfg.safety_dt)
    return records


def _classify_counterexample(
    records: list[StepRecord],
    start_index: int,
    end_index: int,
) -> tuple[str, dict[str, bool]]:
    window = records[start_index : end_index + 1]
    end = records[end_index]
    true_ids = {
        obstacle["identifier"] for record in window for obstacle in record.true_obstacles
    }
    perceived_by_step = [
        {obstacle["identifier"] for obstacle in record.perceived_obstacles}
        for record in window
    ]
    disappeared = any(ids != true_ids for ids in perceived_by_step)
    reappeared = disappeared and perceived_by_step[-1] == true_ids
    stale = any(any(age > 0.0 for age in record.stale_obstacle_ages.values()) for record in window)
    braking = end.required_braking > end.available_braking + 1e-9
    multi = len(end.active_constraint_identifiers) >= 2 or sum(
        weight > 1e-4 for weight in end.dual_weights
    ) >= 2
    sampled_data_only = bool(
        end.rho_true_unstrengthened is not None
        and end.rho_true_unstrengthened >= 0.0
    )
    if reappeared:
        cause = "obstacle_reappearance_after_constraint_disappearance"
    elif disappeared:
        cause = "constraint_disappearance_during_dropout_or_occlusion"
    elif braking:
        cause = "bounded_actuation_braking_authority_loss"
    elif multi:
        cause = "multi_obstacle_constraint_conflict"
    elif sampled_data_only:
        cause = "sampled_data_residual_strengthening"
    elif stale:
        cause = "sensing_delay_or_stale_perception"
    else:
        cause = "sampled_dynamics_or_obstacle_evolution"
    return cause, {
        "obstacle_constraint_disappeared": disappeared,
        "obstacle_reappeared": reappeared,
        "sensing_delay_present": stale,
        "braking_authority_lost": braking,
        "multi_obstacle_conflict": multi,
        "sampled_data_strengthening_caused_infeasibility": sampled_data_only,
    }


def _obstacle_from_dict(payload: dict[str, object]) -> SphericalObstacle:
    return SphericalObstacle(
        np.asarray(payload["center"], dtype=np.float64),
        float(payload["radius"]),
        np.asarray(payload["velocity"], dtype=np.float64),
        np.asarray(payload["acceleration"], dtype=np.float64),
        str(payload["identifier"]),
    )


def _sampled_one_step_recovery(
    start: StepRecord,
    end: StepRecord,
    config: RecoverabilityDiagnosticConfig,
) -> tuple[bool, float, list[float], float]:
    started = perf_counter()
    position = np.asarray(start.position, dtype=np.float64)
    velocity = np.asarray(start.velocity, dtype=np.float64)
    next_obstacles = tuple(
        _obstacle_from_dict(payload) for payload in end.true_obstacles
    )
    candidates = [
        np.asarray(start.executed_action, dtype=np.float64),
        emergency_braking_acceleration(
            velocity,
            config.horizontal_acceleration_limit,
            config.vertical_acceleration_limit,
        ),
    ]
    if start.margin_maximizing_action is not None:
        candidates.append(np.asarray(start.margin_maximizing_action, dtype=np.float64))
    for vertical in (
        -config.vertical_acceleration_limit,
        0.0,
        config.vertical_acceleration_limit,
    ):
        for angle in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            candidates.append(
                np.array(
                    [
                        config.horizontal_acceleration_limit * np.cos(angle),
                        config.horizontal_acceleration_limit * np.sin(angle),
                        vertical,
                    ]
                )
            )
    best_rho = -math.inf
    best_action = candidates[0]
    for action in candidates:
        next_velocity = _clip_velocity(
            velocity + action * config.safety_dt,
            config,
        )
        next_position = position + next_velocity * config.safety_dt
        margin = compute_pointwise_margin(
            next_position,
            next_velocity,
            next_obstacles,
            config,
        )
        rho = math.inf if margin.rho is None else float(margin.rho)
        if rho > best_rho:
            best_rho = rho
            best_action = action
    return (
        best_rho >= -config.positive_margin_threshold,
        float(best_rho),
        best_action.tolist(),
        float(perf_counter() - started),
    )


def search_counterexamples(
    scenario: HardScenario,
    records: list[StepRecord],
    *,
    perception_mode: PerceptionMode,
    config: RecoverabilityDiagnosticConfig | None = None,
) -> list[CounterexampleRecord]:
    cfg = RecoverabilityDiagnosticConfig() if config is None else config
    counterexamples: list[CounterexampleRecord] = []
    negative_entries = [
        index
        for index, record in enumerate(records)
        if record.rho_true is not None
        and record.rho_true < -cfg.positive_margin_threshold
        and (
            index == 0
            or records[index - 1].rho_true is None
            or records[index - 1].rho_true >= -cfg.positive_margin_threshold
        )
    ]
    for negative_index in negative_entries:
        search_start = max(0, negative_index - cfg.lookahead_steps)
        positive_indices = [
            index
            for index in range(search_start, negative_index)
            if records[index].rho_true is not None
            and records[index].rho_true > cfg.positive_margin_threshold
        ]
        if not positive_indices:
            continue
        start_index = positive_indices[-1]
        record = records[start_index]
        cause, flags = _classify_counterexample(records, start_index, negative_index)
        future = [records[index].rho_true for index in range(start_index, negative_index + 1)]
        end = records[negative_index]
        recovery_exists, best_next_rho, best_action, recovery_seconds = _sampled_one_step_recovery(
            record,
            end,
            cfg,
        )
        counterexamples.append(
            CounterexampleRecord(
                scenario_id=scenario.identifier,
                family=scenario.family,
                perception_mode=perception_mode.value,
                feasible_step=record.step,
                infeasible_step=end.step,
                horizon=end.step - record.step,
                rho_t=float(record.rho_true),
                rho_future=future,
                cause=cause,
                minimum_ttc=end.minimum_ttc,
                minimum_clearance=end.minimum_clearance,
                required_braking=end.required_braking,
                available_braking=end.available_braking,
                relative_velocities=end.relative_velocities,
                responsible_constraints=end.active_constraint_identifiers,
                sampled_recovery_action_exists=recovery_exists,
                best_sampled_next_rho=best_next_rho,
                best_sampled_action=best_action,
                sampled_recovery_search_seconds=recovery_seconds,
                state={
                    "uav_position": record.position,
                    "uav_velocity": record.velocity,
                    "current_action": record.executed_action,
                    "filter_feasible": record.filter_feasible,
                    "fallback_used": record.fallback_used,
                    "fallback_satisfies_constraints": (
                        record.fallback_satisfies_constraints
                    ),
                    "hocbf_rows": record.hocbf_rows,
                    "hocbf_bounds": record.hocbf_bounds,
                    "hocbf_constraint_identifiers": record.hocbf_constraint_identifiers,
                    "actuator_constraint_representation": {
                        "physical_margin_set": "cylindrical_input_set",
                        "filter_solver_set": "inscribed_polygon",
                        "horizontal_constraint": "norm(u_xy)_2 <= horizontal_acceleration_limit",
                        "vertical_constraint": "abs(u_z) <= vertical_acceleration_limit",
                        "horizontal_acceleration_limit": cfg.horizontal_acceleration_limit,
                        "vertical_acceleration_limit": cfg.vertical_acceleration_limit,
                        "actuator_rows": record.actuator_rows,
                        "actuator_bounds": record.actuator_bounds,
                        "velocity_rows": record.velocity_rows,
                        "velocity_bounds": record.velocity_bounds,
                    },
                    "true_obstacles": record.true_obstacles,
                    "perceived_obstacles": record.perceived_obstacles,
                    "sensing_history": [
                        {
                            "step": item.step,
                            "visible": item.visible_obstacles,
                            "stale_ages": item.stale_obstacle_ages,
                        }
                        for item in records[max(0, start_index - 4) : negative_index + 1]
                    ],
                },
                **flags,
            )
        )
    return counterexamples


def _obstacle(
    identifier: str,
    center: tuple[float, float, float],
    radius: float,
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    acceleration: tuple[float, float, float] = (0.0, 0.0, 0.0),
    *,
    events: tuple[AccelerationEvent, ...] = (),
) -> DynamicObstacleSpec:
    return DynamicObstacleSpec(
        identifier=identifier,
        center=np.asarray(center),
        radius=radius,
        velocity=np.asarray(velocity),
        acceleration=np.asarray(acceleration),
        acceleration_events=events,
    )


def make_hand_scenarios() -> list[HardScenario]:
    base_start = np.array([2000.0, 1900.0, 200.0])
    base_goal = np.array([2000.0, 2200.0, 200.0])
    scenarios = [
        HardScenario(
            "two_obstacle_squeeze",
            "two_obstacle_squeeze",
            base_start,
            base_goal,
            np.array([0.0, 18.0, 0.0]),
            (
                _obstacle("left", (1968.0, 2000.0, 200.0), 24.0, (4.0, 0.0, 0.0)),
                _obstacle("right", (2032.0, 2000.0, 200.0), 24.0, (-4.0, 0.0, 0.0)),
            ),
        ),
        HardScenario(
            "symmetric_corridor",
            "symmetric_corridor",
            base_start,
            base_goal,
            np.array([0.0, 17.0, 0.0]),
            tuple(
                _obstacle(f"corridor_{index}", (x, y, 200.0), 26.0)
                for index, (x, y) in enumerate(
                    ((1968.0, 1980.0), (2032.0, 1980.0), (1972.0, 2040.0), (2028.0, 2040.0))
                )
            ),
        ),
        HardScenario(
            "crossing_dynamic",
            "crossing_dynamic_obstacles",
            base_start,
            base_goal,
            np.array([0.0, 18.0, 0.0]),
            (
                _obstacle("cross_a", (1940.0, 2010.0, 200.0), 18.0, (10.0, 0.0, 0.0)),
                _obstacle("cross_b", (2060.0, 2050.0, 200.0), 18.0, (-10.0, 0.0, 0.0)),
            ),
        ),
        HardScenario(
            "emerging_occlusion",
            "obstacle_emerging_from_occlusion",
            base_start,
            base_goal,
            np.array([0.0, 19.0, 0.0]),
            (_obstacle("occluded", (1980.0, 2010.0, 200.0), 20.0, (7.0, 0.0, 0.0)),),
            (VisibilityWindow(0, 62, ("occluded",), "occlusion"),),
        ),
        HardScenario(
            "whole_object_dropout",
            "temporary_lidar_whole_object_dropout",
            base_start,
            base_goal,
            np.array([0.0, 19.0, 0.0]),
            (_obstacle("dropout", (2000.0, 2020.0, 200.0), 23.0),),
            (VisibilityWindow(42, 70, ("dropout",), "whole_object_dropout"),),
        ),
        HardScenario(
            "disappear_reappear",
            "obstacle_disappears_and_reappears",
            base_start,
            base_goal,
            np.array([0.0, 18.0, 0.0]),
            (_obstacle("reappear", (2000.0, 2025.0, 200.0), 22.0, (0.0, -2.0, 0.0)),),
            (VisibilityWindow(48, 64, ("reappear",), "scheduled_dropout"),),
        ),
        HardScenario(
            "sudden_acceleration",
            "suddenly_accelerating_obstacle",
            base_start,
            base_goal,
            np.array([0.0, 18.0, 0.0]),
            (
                _obstacle(
                    "accelerator",
                    (1965.0, 2020.0, 200.0),
                    20.0,
                    (2.0, 0.0, 0.0),
                    events=(AccelerationEvent(12, np.array([4.0, 0.0, 0.0])),),
                ),
            ),
        ),
        HardScenario(
            "high_speed_braking",
            "uav_high_speed_bounded_braking",
            base_start,
            base_goal,
            np.array([0.0, 20.0, 0.0]),
            (_obstacle("blocking", (2000.0, 1990.0, 200.0), 30.0),),
        ),
        HardScenario(
            "boundary_obstacle_conflict",
            "boundary_and_obstacle_conflict",
            np.array([12.0, 1800.0, 200.0]),
            np.array([12.0, 2200.0, 200.0]),
            np.array([0.0, 18.0, 0.0]),
            (_obstacle("wall_side", (36.0, 1995.0, 200.0), 28.0),),
        ),
        HardScenario(
            "vertical_horizontal_conflict",
            "vertical_horizontal_simultaneous_conflict",
            base_start,
            base_goal,
            np.array([0.0, 18.0, 2.5]),
            (
                _obstacle("horizontal", (1975.0, 2020.0, 200.0), 22.0, (4.0, 0.0, 0.0)),
                _obstacle("vertical", (2000.0, 2025.0, 225.0), 22.0, (0.0, 0.0, -3.0)),
            ),
        ),
    ]
    return scenarios


def make_random_hard_scenarios(count: int, seed: int) -> list[HardScenario]:
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    templates = make_hand_scenarios()
    scenarios = []
    for index in range(count):
        template = templates[index % len(templates)]
        translation = np.array(
            [rng.uniform(-80.0, 80.0), rng.uniform(-80.0, 80.0), rng.uniform(-15.0, 15.0)]
        )
        start = template.start + translation
        goal = template.goal + translation
        speed_scale = rng.uniform(0.85, 1.05)
        initial_velocity = template.initial_velocity * speed_scale
        obstacle_specs = []
        for obstacle in template.obstacles:
            center_jitter = rng.normal(0.0, [4.0, 4.0, 2.0])
            velocity_scale = rng.uniform(0.8, 1.2)
            obstacle_specs.append(
                DynamicObstacleSpec(
                    identifier=obstacle.identifier,
                    center=obstacle.center + translation + center_jitter,
                    radius=max(1.0, obstacle.radius + rng.normal(0.0, 1.0)),
                    velocity=obstacle.velocity * velocity_scale,
                    acceleration=obstacle.acceleration,
                    jerk=obstacle.jerk,
                    acceleration_events=obstacle.acceleration_events,
                    maximum_speed=obstacle.maximum_speed,
                    maximum_acceleration=obstacle.maximum_acceleration,
                    maximum_jerk=obstacle.maximum_jerk,
                )
            )
        windows_list = []
        for window in template.visibility_windows:
            start_step = max(
                0,
                window.start_step + int(rng.integers(-2, 3)),
            )
            end_step = max(
                start_step + 1,
                window.end_step + int(rng.integers(-2, 3)),
            )
            windows_list.append(
                VisibilityWindow(
                    start_step,
                    end_step,
                    window.obstacle_identifiers,
                    window.cause,
                )
            )
        windows = tuple(windows_list)
        scenarios.append(
            HardScenario(
                identifier=f"{template.family}_{index:06d}",
                family=template.family,
                start=start,
                goal=goal,
                initial_velocity=initial_velocity,
                obstacles=tuple(obstacle_specs),
                visibility_windows=windows,
                max_steps=template.max_steps,
            )
        )
    return scenarios


def record_to_dict(record: StepRecord | CounterexampleRecord) -> dict[str, object]:
    return asdict(record)
