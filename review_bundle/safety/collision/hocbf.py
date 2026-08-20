from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


def _positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


@dataclass(frozen=True)
class SphericalObstacle:
    center: np.ndarray
    radius: float
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    identifier: str = "obstacle"

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _vec3(self.center, "center"))
        object.__setattr__(self, "radius", _positive(self.radius, "radius"))
        velocity = np.zeros(3) if self.velocity is None else _vec3(self.velocity, "velocity")
        acceleration = (
            np.zeros(3)
            if self.acceleration is None
            else _vec3(self.acceleration, "acceleration")
        )
        object.__setattr__(self, "velocity", velocity)
        object.__setattr__(self, "acceleration", acceleration)


@dataclass(frozen=True)
class HOCBFConfig:
    k1: float = 1.0
    k2: float = 1.0
    uav_radius: float = 0.5
    uncertainty_margin: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "k1", _positive(self.k1, "k1"))
        object.__setattr__(self, "k2", _positive(self.k2, "k2"))
        if self.uav_radius < 0.0 or self.uncertainty_margin < 0.0:
            raise ValueError("safety radii and margins must be nonnegative")


@dataclass(frozen=True)
class BarrierConstraint:
    row: np.ndarray
    lower_bound: float
    h: float
    h_dot: float
    psi1: float
    drift: float
    safe_distance: float
    identifier: str
    sampled_data_margin: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "row", _vec3(self.row, "row"))
        values = (
            self.lower_bound,
            self.h,
            self.h_dot,
            self.psi1,
            self.drift,
            self.safe_distance,
            self.sampled_data_margin,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("barrier diagnostics must be finite")
        if self.sampled_data_margin < 0.0:
            raise ValueError("sampled_data_margin must be nonnegative")

    def slack(self, acceleration: np.ndarray) -> float:
        return float(self.row @ _vec3(acceleration, "acceleration") - self.lower_bound)


@dataclass(frozen=True)
class ProjectionResult:
    acceleration: np.ndarray
    feasible: bool
    converged: bool
    iterations: int
    max_violation: float
    intervention_norm: float
    solver_time_seconds: float
    active_constraints: int
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acceleration",
            _vec3(self.acceleration, "projected acceleration"),
        )


def sphere_hocbf_constraint(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacle: SphericalObstacle,
    config: HOCBFConfig,
) -> BarrierConstraint:
    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    relative_position = p - obstacle.center
    relative_velocity = v - obstacle.velocity
    safe_distance = (
        obstacle.radius + config.uav_radius + config.uncertainty_margin
    )
    h = float(relative_position @ relative_position - safe_distance**2)
    h_dot = float(2.0 * relative_position @ relative_velocity)
    psi1 = float(h_dot + config.k1 * h)
    drift = float(
        2.0 * relative_velocity @ relative_velocity
        - 2.0 * relative_position @ obstacle.acceleration
        + (config.k1 + config.k2) * h_dot
        + config.k1 * config.k2 * h
    )
    row = 2.0 * relative_position
    return BarrierConstraint(
        row=row,
        lower_bound=-drift,
        h=h,
        h_dot=h_dot,
        psi1=psi1,
        drift=drift,
        safe_distance=float(safe_distance),
        identifier=obstacle.identifier,
    )


def hocbf_sampled_data_residual_bound(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacle: SphericalObstacle,
    config: HOCBFConfig,
    *,
    hold_dt: float,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
) -> float:
    """Bound the possible decrease of psi2 during one ZOH interval.

    The bound is for the translational double integrator with a constant UAV
    acceleration and a known constant obstacle acceleration over the hold.
    It is intentionally local: current relative position and velocity are used
    instead of a global LiDAR-range bound.
    """

    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    interval = _positive(hold_dt, "hold_dt")
    horizontal = _positive(
        horizontal_acceleration_limit,
        "horizontal_acceleration_limit",
    )
    vertical = _positive(vertical_acceleration_limit, "vertical_acceleration_limit")
    relative_position = p - obstacle.center
    relative_velocity = v - obstacle.velocity
    uav_acceleration_bound = float(np.hypot(horizontal, vertical))
    obstacle_acceleration_bound = float(np.linalg.norm(obstacle.acceleration))
    relative_acceleration_bound = (
        uav_acceleration_bound + obstacle_acceleration_bound
    )
    velocity_change = relative_acceleration_bound * interval
    position_change = (
        np.linalg.norm(relative_velocity) * interval
        + 0.5 * relative_acceleration_bound * interval**2
    )
    relative_position_bound = float(np.linalg.norm(relative_position) + position_change)
    relative_velocity_bound = float(np.linalg.norm(relative_velocity) + velocity_change)
    gain_sum = config.k1 + config.k2
    gain_product = config.k1 * config.k2
    position_lipschitz = (
        2.0 * obstacle_acceleration_bound
        + 2.0 * gain_sum * relative_velocity_bound
        + 2.0 * gain_product * relative_position_bound
        + 2.0 * uav_acceleration_bound
    )
    velocity_lipschitz = (
        4.0 * relative_velocity_bound
        + 2.0 * gain_sum * relative_position_bound
    )
    return float(
        position_lipschitz * position_change
        + velocity_lipschitz * velocity_change
    )


def strengthen_constraint_for_sample_hold(
    constraint: BarrierConstraint,
    residual_bound: float,
) -> BarrierConstraint:
    margin = float(residual_bound)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("residual_bound must be finite and nonnegative")
    return BarrierConstraint(
        row=constraint.row,
        lower_bound=constraint.lower_bound + margin,
        h=constraint.h,
        h_dot=constraint.h_dot,
        psi1=constraint.psi1,
        drift=constraint.drift - margin,
        safe_distance=constraint.safe_distance,
        identifier=constraint.identifier,
        sampled_data_margin=constraint.sampled_data_margin + margin,
    )


def aggregate_hocbf_constraint(
    constraints: list[BarrierConstraint],
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: list[SphericalObstacle],
    config: HOCBFConfig,
    rho: float,
) -> BarrierConstraint:
    if not constraints or len(constraints) != len(obstacles):
        raise ValueError("aggregate inputs must be aligned and nonempty")
    concentration = _positive(rho, "rho")
    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    h_values = np.asarray([item.h for item in constraints], dtype=np.float64)
    shifted = -concentration * h_values
    shift = float(np.max(shifted))
    exponentials = np.exp(shifted - shift)
    weights = exponentials / np.sum(exponentials)
    h_aggregate = float(-(np.log(np.sum(exponentials)) + shift) / concentration)

    h_dots = np.asarray([item.h_dot for item in constraints], dtype=np.float64)
    h_dot_aggregate = float(weights @ h_dots)
    variance = float(weights @ (h_dots * h_dots) - h_dot_aggregate**2)
    variance = max(0.0, variance)

    rows = np.stack([item.row for item in constraints])
    row = weights @ rows
    h_ddot_drifts = []
    for obstacle in obstacles:
        relative_position = p - obstacle.center
        relative_velocity = v - obstacle.velocity
        h_ddot_drifts.append(
            2.0 * relative_velocity @ relative_velocity
            - 2.0 * relative_position @ obstacle.acceleration
        )
    h_ddot_drift = float(weights @ np.asarray(h_ddot_drifts) - concentration * variance)
    drift = float(
        h_ddot_drift
        + (config.k1 + config.k2) * h_dot_aggregate
        + config.k1 * config.k2 * h_aggregate
    )
    psi1 = float(h_dot_aggregate + config.k1 * h_aggregate)
    return BarrierConstraint(
        row=row,
        lower_bound=-drift,
        h=h_aggregate,
        h_dot=h_dot_aggregate,
        psi1=psi1,
        drift=drift,
        safe_distance=max(item.safe_distance for item in constraints),
        identifier="aggregate_softmin",
    )


def one_step_supporting_constraint(
    position: np.ndarray,
    velocity: np.ndarray,
    nominal_acceleration: np.ndarray,
    obstacle: SphericalObstacle,
    *,
    uav_radius: float,
    uncertainty_margin: float,
    dt: float,
) -> BarrierConstraint:
    p = _vec3(position, "position")
    v = _vec3(velocity, "velocity")
    nominal = _vec3(nominal_acceleration, "nominal_acceleration")
    interval = _positive(dt, "dt")
    safe_distance = obstacle.radius + float(uav_radius) + float(uncertainty_margin)
    relative = p - obstacle.center
    relative_velocity = v - obstacle.velocity
    nominal_next = (
        relative
        + relative_velocity * interval
        + 0.5 * (nominal - obstacle.acceleration) * interval**2
    )
    norm = float(np.linalg.norm(nominal_next))
    if norm <= 1e-12:
        fallback = relative if np.linalg.norm(relative) > 1e-12 else -relative_velocity
        norm = float(np.linalg.norm(fallback))
        if norm <= 1e-12:
            raise ValueError("supporting direction is undefined at obstacle center")
        direction = fallback / norm
    else:
        direction = nominal_next / norm
    row = direction
    lower_bound = float(
        2.0
        * (
            safe_distance
            - direction
            @ (relative + relative_velocity * interval - 0.5 * obstacle.acceleration * interval**2)
        )
        / interval**2
    )
    h = float(relative @ relative - safe_distance**2)
    h_dot = float(2.0 * relative @ relative_velocity)
    return BarrierConstraint(
        row=row,
        lower_bound=lower_bound,
        h=h,
        h_dot=h_dot,
        psi1=np.nan_to_num(h_dot),
        drift=-lower_bound,
        safe_distance=safe_distance,
        identifier=f"one_step:{obstacle.identifier}",
    )


def project_single_halfspace(
    nominal_acceleration: np.ndarray,
    row: np.ndarray,
    lower_bound: float,
    weight: np.ndarray | None = None,
) -> np.ndarray:
    nominal = _vec3(nominal_acceleration, "nominal_acceleration")
    normal = _vec3(row, "row")
    hessian = np.eye(3) if weight is None else np.asarray(weight, dtype=np.float64)
    if hessian.shape != (3, 3) or not np.all(np.isfinite(hessian)):
        raise ValueError("weight must be a finite (3, 3) matrix")
    if np.min(np.linalg.eigvalsh(hessian)) <= 0.0:
        raise ValueError("weight must be positive definite")
    violation = float(lower_bound - normal @ nominal)
    if violation <= 0.0:
        return nominal
    inverse = np.linalg.inv(hessian)
    denominator = float(normal @ inverse @ normal)
    if denominator <= 1e-15:
        raise ValueError("halfspace normal is degenerate in the weighted metric")
    return nominal + violation * (inverse @ normal) / denominator


def actuator_polygon_constraints(
    horizontal_limit: float,
    vertical_limit: float,
    facets: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    horizontal = _positive(horizontal_limit, "horizontal_limit")
    vertical = _positive(vertical_limit, "vertical_limit")
    if facets < 4:
        raise ValueError("actuator polygon requires at least four facets")
    inradius = horizontal * np.cos(np.pi / facets)
    rows = []
    bounds = []
    for index in range(facets):
        angle = 2.0 * np.pi * index / facets
        outward = np.array([np.cos(angle), np.sin(angle), 0.0])
        rows.append(-outward)
        bounds.append(-inradius)
    rows.extend((np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])))
    bounds.extend((-vertical, -vertical))
    return np.asarray(rows, dtype=np.float64), np.asarray(bounds, dtype=np.float64)


def velocity_polygon_constraints(
    velocity: np.ndarray,
    *,
    dt: float,
    horizontal_limit: float,
    vertical_limit: float,
    facets: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    current = _vec3(velocity, "velocity")
    interval = _positive(dt, "dt")
    horizontal = _positive(horizontal_limit, "horizontal_limit")
    vertical = _positive(vertical_limit, "vertical_limit")
    if facets < 4:
        raise ValueError("velocity polygon requires at least four facets")
    inradius = horizontal * np.cos(np.pi / facets)
    rows = []
    bounds = []
    for index in range(facets):
        angle = 2.0 * np.pi * index / facets
        outward = np.array([np.cos(angle), np.sin(angle), 0.0])
        rows.append(-outward)
        bounds.append((float(outward @ current) - inradius) / interval)
    rows.extend((np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0])))
    bounds.extend(
        (
            (current[2] - vertical) / interval,
            (-vertical - current[2]) / interval,
        )
    )
    return np.asarray(rows, dtype=np.float64), np.asarray(bounds, dtype=np.float64)


def energy_aware_quadratic(
    nominal_acceleration: np.ndarray,
    intervention_weight: np.ndarray,
    energy_matrix: np.ndarray,
    *,
    energy_weight: float,
    dt: float,
    energy_gradient: np.ndarray | None = None,
    gradient_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    nominal = _vec3(nominal_acceleration, "nominal_acceleration")
    weight = np.asarray(intervention_weight, dtype=np.float64)
    energy = np.asarray(energy_matrix, dtype=np.float64)
    if weight.shape != (3, 3) or energy.shape != (3, 3):
        raise ValueError("quadratic matrices must have shape (3, 3)")
    if np.min(np.linalg.eigvalsh(weight)) <= 0.0:
        raise ValueError("intervention weight must be positive definite")
    if np.min(np.linalg.eigvalsh(energy)) < -1e-12:
        raise ValueError("energy matrix must be positive semidefinite")
    if energy_weight < 0.0 or gradient_weight < 0.0:
        raise ValueError("energy weights must be nonnegative")
    interval = _positive(dt, "dt")
    hessian = weight + 2.0 * float(energy_weight) * interval * energy
    linear_target = weight @ nominal
    if energy_gradient is not None:
        linear_target -= float(gradient_weight) * _vec3(
            energy_gradient,
            "energy_gradient",
        )
    center = np.linalg.solve(hessian, linear_target)
    return hessian, center


def energy_to_go_action_gradient(
    grad_position: np.ndarray,
    grad_velocity: np.ndarray,
    dt: float,
) -> np.ndarray:
    grad_p = _vec3(grad_position, "grad_position")
    grad_v = _vec3(grad_velocity, "grad_velocity")
    interval = _positive(dt, "dt")
    return 0.5 * interval**2 * grad_p + interval * grad_v


def project_polyhedral_qp(
    center: np.ndarray,
    hessian: np.ndarray,
    rows: np.ndarray,
    lower_bounds: np.ndarray,
    *,
    max_iterations: int = 250,
    tolerance: float = 1e-7,
    stagnation_sweeps: int = 50,
) -> ProjectionResult:
    started = perf_counter()
    reference = _vec3(center, "center")
    metric = np.asarray(hessian, dtype=np.float64)
    matrix = np.asarray(rows, dtype=np.float64)
    bounds = np.asarray(lower_bounds, dtype=np.float64)
    if metric.shape != (3, 3) or np.min(np.linalg.eigvalsh(metric)) <= 0.0:
        raise ValueError("hessian must be positive definite")
    if matrix.ndim != 2 or matrix.shape[1] != 3 or bounds.shape != (matrix.shape[0],):
        raise ValueError("linear constraints must have shapes (m, 3) and (m,)")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bounds)):
        raise ValueError("linear constraints must be finite")
    if max_iterations <= 0 or tolerance <= 0.0 or stagnation_sweeps <= 0:
        raise ValueError("solver iteration and tolerance must be positive")

    inverse = np.linalg.inv(metric)
    acceleration = reference.copy()
    multipliers = np.zeros(matrix.shape[0], dtype=np.float64)
    denominators = np.einsum("ij,jk,ik->i", matrix, inverse, matrix)
    nondegenerate = denominators > 1e-15
    if np.any(~nondegenerate & (bounds > matrix @ acceleration + tolerance)):
        elapsed = perf_counter() - started
        return ProjectionResult(
            acceleration=acceleration,
            feasible=False,
            converged=False,
            iterations=0,
            max_violation=float(np.max(bounds - matrix @ acceleration)),
            intervention_norm=float(np.linalg.norm(acceleration - reference)),
            solver_time_seconds=elapsed,
            active_constraints=0,
            reason="degenerate_infeasible_constraint",
        )

    converged = False
    iterations = 0
    violation_history: list[float] = []
    stagnated = False
    for sweep in range(1, max_iterations + 1):
        largest_delta = 0.0
        for index in range(matrix.shape[0]):
            if not nondegenerate[index]:
                continue
            violation = float(bounds[index] - matrix[index] @ acceleration)
            candidate = max(
                0.0,
                multipliers[index] + violation / denominators[index],
            )
            delta = candidate - multipliers[index]
            if delta != 0.0:
                acceleration += delta * (inverse @ matrix[index])
                multipliers[index] = candidate
                largest_delta = max(largest_delta, abs(delta))
        iterations = sweep
        max_violation = float(max(0.0, np.max(bounds - matrix @ acceleration)))
        violation_history.append(max_violation)
        if max_violation <= tolerance and largest_delta <= tolerance:
            converged = True
            break
        if (
            len(violation_history) > stagnation_sweeps
            and max_violation > max(tolerance * 10.0, 1e-3)
            and max_violation
            >= violation_history[-stagnation_sweeps - 1] * (1.0 - 1e-6)
            - tolerance
        ):
            stagnated = True
            break
    max_violation = float(max(0.0, np.max(bounds - matrix @ acceleration)))
    feasible = bool(max_violation <= tolerance * 10.0)
    elapsed = perf_counter() - started
    return ProjectionResult(
        acceleration=acceleration,
        feasible=feasible,
        converged=converged,
        iterations=iterations,
        max_violation=max_violation,
        intervention_norm=float(np.linalg.norm(acceleration - reference)),
        solver_time_seconds=elapsed,
        active_constraints=int(np.sum(multipliers > tolerance)),
        reason=(
            "optimal"
            if feasible and converged
            else (
                "stagnated_infeasible_or_ill_conditioned"
                if stagnated
                else "iteration_limit_or_infeasible"
            )
        ),
    )


def select_top_k_constraints(
    constraints: list[BarrierConstraint],
    nominal_acceleration: np.ndarray,
    top_k: int,
    *,
    priority: str = "hocbf_slack",
    braking_acceleration: float = 1.0,
) -> list[BarrierConstraint]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    nominal = _vec3(nominal_acceleration, "nominal_acceleration")
    if priority not in {
        "distance",
        "closing_speed",
        "ttc",
        "barrier_value",
        "hocbf_slack",
        "braking_margin",
    }:
        raise ValueError(f"unsupported top-K priority: {priority}")
    braking = _positive(braking_acceleration, "braking_acceleration")

    def score(item: BarrierConstraint) -> float:
        center_distance = float(np.sqrt(max(item.h + item.safe_distance**2, 0.0)))
        surface_distance = center_distance - item.safe_distance
        closing_speed = (
            max(0.0, -item.h_dot / (2.0 * center_distance))
            if center_distance > 1e-12
            else np.inf
        )
        if priority == "distance":
            return surface_distance
        if priority == "closing_speed":
            return -closing_speed
        if priority == "ttc":
            return (
                surface_distance / closing_speed
                if closing_speed > 1e-12
                else np.inf
            )
        if priority == "barrier_value":
            return item.h
        if priority == "braking_margin":
            return surface_distance - closing_speed**2 / (2.0 * braking)
        return item.slack(nominal)

    ranked = sorted(constraints, key=score)
    return ranked[: min(top_k, len(ranked))]


def emergency_braking_acceleration(
    velocity: np.ndarray,
    horizontal_limit: float,
    vertical_limit: float,
) -> np.ndarray:
    v = _vec3(velocity, "velocity")
    horizontal = np.zeros(2, dtype=np.float64)
    speed = float(np.linalg.norm(v[:2]))
    if speed > 1e-12:
        horizontal = -_positive(horizontal_limit, "horizontal_limit") * v[:2] / speed
    vertical = 0.0
    if abs(v[2]) > 1e-12:
        vertical = -np.sign(v[2]) * _positive(vertical_limit, "vertical_limit")
    return np.array([horizontal[0], horizontal[1], vertical], dtype=np.float64)


def energy_viability_residual(
    acceleration: np.ndarray,
    velocity: np.ndarray,
    *,
    energy_matrix: np.ndarray,
    fixed_power: float,
    grad_position: np.ndarray,
    grad_velocity: np.ndarray,
    battery_margin: float,
    gain: float,
) -> float:
    u = _vec3(acceleration, "acceleration")
    v = _vec3(velocity, "velocity")
    grad_p = _vec3(grad_position, "grad_position")
    grad_v = _vec3(grad_velocity, "grad_velocity")
    matrix = np.asarray(energy_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or np.min(np.linalg.eigvalsh(matrix)) < -1e-12:
        raise ValueError("energy_matrix must be positive semidefinite")
    if fixed_power < 0.0 or battery_margin < 0.0:
        raise ValueError("power and battery margin must be nonnegative")
    k = _positive(gain, "gain")
    left = float(u @ matrix @ u + grad_v @ u)
    right = float(k * battery_margin - fixed_power - grad_p @ v)
    return right - left
