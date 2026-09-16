from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter

import numpy as np
from scipy.optimize import linprog, minimize

from .feasibility import joint_feasibility_margin
from .projection_geometry import ProjectionProblem

from .hocbf import (
    BarrierConstraint,
    HOCBFConfig,
    ProjectionResult,
    SphericalObstacle,
    actuator_polygon_constraints,
    aggregate_hocbf_constraint,
    emergency_braking_acceleration,
    energy_aware_quadratic,
    hocbf_sampled_data_residual_bound,
    one_step_supporting_constraint,
    project_polyhedral_qp,
    project_polyhedral_qp_constraint_generation,
    select_top_k_constraints,
    sphere_hocbf_constraint,
    strengthen_constraint_for_sample_hold,
    velocity_polygon_constraints,
)


class SafetyFilterMethod(str, Enum):
    ONE_STEP = "one_step_projection"
    HOCBF = "second_order_hocbf"
    AGGREGATE_HOCBF = "aggregate_hocbf"
    ENERGY_AWARE_HOCBF = "energy_aware_hocbf"
    ENERGY_GRADIENT_HOCBF = "energy_gradient_hocbf"
    SAMPLED_DATA_HOCBF = "sampled_data_hocbf"
    SAMPLED_DATA_ENERGY_AWARE_HOCBF = "sampled_data_energy_aware_hocbf"
    SAMPLED_DATA_ENERGY_GRADIENT_HOCBF = "sampled_data_energy_gradient_hocbf"
    SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF = (
        "sampled_data_lexicographic_energy_hocbf"
    )


@dataclass(frozen=True)
class SafetyFilterConfig:
    method: SafetyFilterMethod = SafetyFilterMethod.HOCBF
    horizontal_acceleration_limit: float = 5.0
    vertical_acceleration_limit: float = 3.0
    actuator_facets: int = 32
    top_k: int | None = None
    top_k_priority: str = "hocbf_slack"
    aggregate_rho: float = 1.0
    safety_dt: float = 0.05
    deadline_seconds: float = 0.05
    energy_weight: float = 1.0
    energy_matrix: np.ndarray | None = None
    energy_characteristic: float = 1.0
    energy_gradient_weight: float = 1.0
    energy_gradient_characteristic: float = 1.0
    energy_gradient_normalization: str = "fixed"
    sampled_data_robust: bool = False
    horizontal_velocity_limit: float | None = None
    vertical_velocity_limit: float | None = None

    def __post_init__(self) -> None:
        if self.horizontal_acceleration_limit <= 0.0 or self.vertical_acceleration_limit <= 0.0:
            raise ValueError("acceleration limits must be positive")
        if self.actuator_facets < 4:
            raise ValueError("actuator_facets must be at least four")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be positive when configured")
        if self.top_k_priority not in {
            "distance",
            "closing_speed",
            "ttc",
            "barrier_value",
            "hocbf_slack",
            "braking_margin",
        }:
            raise ValueError("unsupported top_k_priority")
        if self.aggregate_rho <= 0.0 or self.safety_dt <= 0.0:
            raise ValueError("aggregate_rho and safety_dt must be positive")
        if (
            self.deadline_seconds <= 0.0
            or self.energy_weight < 0.0
            or self.energy_gradient_weight < 0.0
        ):
            raise ValueError("deadline must be positive and energy_weight nonnegative")
        if self.energy_characteristic <= 0.0:
            raise ValueError("energy_characteristic must be positive")
        if self.energy_gradient_characteristic <= 0.0:
            raise ValueError("energy_gradient_characteristic must be positive")
        if self.energy_gradient_normalization not in {"fixed", "unit_direction"}:
            raise ValueError("unsupported energy_gradient_normalization")
        velocity_limits = (
            self.horizontal_velocity_limit,
            self.vertical_velocity_limit,
        )
        if (velocity_limits[0] is None) != (velocity_limits[1] is None):
            raise ValueError("horizontal and vertical velocity limits must be set together")
        if velocity_limits[0] is not None and (
            velocity_limits[0] <= 0.0 or velocity_limits[1] <= 0.0
        ):
            raise ValueError("velocity limits must be positive")
        matrix = (
            np.zeros((3, 3), dtype=np.float64)
            if self.energy_matrix is None
            else np.asarray(self.energy_matrix, dtype=np.float64)
        )
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("energy_matrix must be finite with shape (3, 3)")
        if np.min(np.linalg.eigvalsh(matrix)) < -1e-12:
            raise ValueError("energy_matrix must be positive semidefinite")
        object.__setattr__(self, "energy_matrix", matrix.copy())


@dataclass(frozen=True)
class SafetyFilterDiagnostics:
    method: str
    feasible: bool
    fallback_used: bool
    fallback_satisfies_constraints: bool
    candidate_constraints: int
    active_constraints: int
    dropped_constraints: int
    minimum_h: float | None
    minimum_psi1: float | None
    minimum_nominal_slack: float | None
    minimum_executed_slack: float | None
    intervention_norm: float
    constraint_build_seconds: float
    solver_seconds: float
    total_seconds: float
    deadline_missed: bool
    solver_iterations: int
    solver_reason: str
    feasibility_margin: float | None = None
    feasibility_margin_duality_gap: float | None = None
    required_progress: float | None = None
    achieved_progress: float | None = None


@dataclass(frozen=True)
class SafetyFilterOutput:
    acceleration: np.ndarray
    diagnostics: SafetyFilterDiagnostics
    projection_problem: ProjectionProblem | None = None


@dataclass(frozen=True)
class _BarrierConstraintBatch:
    rows: np.ndarray
    lower_bounds: np.ndarray
    h: np.ndarray
    h_dot: np.ndarray
    psi1: np.ndarray
    safe_distances: np.ndarray


def _vectorized_hocbf_batch(
    position: np.ndarray,
    velocity: np.ndarray,
    obstacles: list[SphericalObstacle],
    config: HOCBFConfig,
) -> _BarrierConstraintBatch:
    """Build the ordinary spherical HOCBF constraints without Python objects."""

    if not obstacles:
        empty = np.empty(0, dtype=np.float64)
        return _BarrierConstraintBatch(
            rows=np.empty((0, 3), dtype=np.float64),
            lower_bounds=empty,
            h=empty,
            h_dot=empty,
            psi1=empty,
            safe_distances=empty,
        )
    p = np.asarray(position, dtype=np.float64)
    v = np.asarray(velocity, dtype=np.float64)
    if p.shape != (3,) or v.shape != (3,) or not (
        np.all(np.isfinite(p)) and np.all(np.isfinite(v))
    ):
        raise ValueError("position and velocity must be finite (3,) vectors")
    centers = np.stack([obstacle.center for obstacle in obstacles])
    obstacle_velocities = np.stack([obstacle.velocity for obstacle in obstacles])
    obstacle_accelerations = np.stack([obstacle.acceleration for obstacle in obstacles])
    radii = np.fromiter(
        (obstacle.radius for obstacle in obstacles),
        dtype=np.float64,
        count=len(obstacles),
    )
    return _vectorized_hocbf_arrays(
        p,
        v,
        centers=centers,
        radii=radii,
        obstacle_velocities=obstacle_velocities,
        obstacle_accelerations=obstacle_accelerations,
        config=config,
    )


def _vectorized_hocbf_arrays(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    centers: np.ndarray,
    radii: np.ndarray,
    obstacle_velocities: np.ndarray,
    obstacle_accelerations: np.ndarray,
    config: HOCBFConfig,
) -> _BarrierConstraintBatch:
    relative_position = position[None, :] - centers
    relative_velocity = velocity[None, :] - obstacle_velocities
    safe_distances = radii + config.uav_radius + config.uncertainty_margin
    h = np.einsum("ij,ij->i", relative_position, relative_position) - (
        safe_distances * safe_distances
    )
    h_dot = 2.0 * np.einsum(
        "ij,ij->i",
        relative_position,
        relative_velocity,
    )
    drift = (
        2.0 * np.einsum("ij,ij->i", relative_velocity, relative_velocity)
        - 2.0
        * np.einsum("ij,ij->i", relative_position, obstacle_accelerations)
        + (config.k1 + config.k2) * h_dot
        + config.k1 * config.k2 * h
    )
    return _BarrierConstraintBatch(
        rows=2.0 * relative_position,
        lower_bounds=-drift,
        h=h,
        h_dot=h_dot,
        psi1=h_dot + config.k1 * h,
        safe_distances=safe_distances,
    )


def _sampled_data_residual_bounds_arrays(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    centers: np.ndarray,
    obstacle_velocities: np.ndarray,
    obstacle_accelerations: np.ndarray,
    config: HOCBFConfig,
    hold_dt: float,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
) -> np.ndarray:
    """Vectorized form of ``hocbf_sampled_data_residual_bound``.

    This is an algebraic batching of the scalar theorem implementation: no
    approximation, obstacle selection, or change to the residual bound is made.
    """

    interval = float(hold_dt)
    horizontal = float(horizontal_acceleration_limit)
    vertical = float(vertical_acceleration_limit)
    if not np.isfinite(interval) or interval <= 0.0:
        raise ValueError("hold_dt must be finite and positive")
    if not np.isfinite(horizontal) or horizontal <= 0.0:
        raise ValueError(
            "horizontal_acceleration_limit must be finite and positive"
        )
    if not np.isfinite(vertical) or vertical <= 0.0:
        raise ValueError("vertical_acceleration_limit must be finite and positive")

    relative_position = position[None, :] - centers
    relative_velocity = velocity[None, :] - obstacle_velocities
    uav_acceleration_bound = float(np.hypot(horizontal, vertical))
    obstacle_acceleration_bound = np.linalg.norm(
        obstacle_accelerations,
        axis=1,
    )
    relative_acceleration_bound = (
        uav_acceleration_bound + obstacle_acceleration_bound
    )
    velocity_change = relative_acceleration_bound * interval
    position_change = (
        np.linalg.norm(relative_velocity, axis=1) * interval
        + 0.5 * relative_acceleration_bound * interval**2
    )
    relative_position_bound = (
        np.linalg.norm(relative_position, axis=1) + position_change
    )
    relative_velocity_bound = (
        np.linalg.norm(relative_velocity, axis=1) + velocity_change
    )
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
    return (
        position_lipschitz * position_change
        + velocity_lipschitz * velocity_change
    )


def _top_k_batch_indices(
    batch: _BarrierConstraintBatch,
    nominal_acceleration: np.ndarray,
    top_k: int,
    *,
    priority: str,
    braking_acceleration: float,
) -> np.ndarray:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    nominal = np.asarray(nominal_acceleration, dtype=np.float64)
    if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
        raise ValueError("nominal_acceleration must be a finite (3,) vector")
    center_distance = np.sqrt(
        np.maximum(batch.h + batch.safe_distances * batch.safe_distances, 0.0)
    )
    surface_distance = center_distance - batch.safe_distances
    closing_speed = np.full_like(center_distance, np.inf)
    nonzero = center_distance > 1e-12
    closing_speed[nonzero] = np.maximum(
        0.0,
        -batch.h_dot[nonzero] / (2.0 * center_distance[nonzero]),
    )
    if priority == "distance":
        scores = surface_distance
    elif priority == "closing_speed":
        scores = -closing_speed
    elif priority == "ttc":
        scores = np.full_like(center_distance, np.inf)
        closing = closing_speed > 1e-12
        scores[closing] = surface_distance[closing] / closing_speed[closing]
    elif priority == "barrier_value":
        scores = batch.h
    elif priority == "braking_margin":
        if not np.isfinite(braking_acceleration) or braking_acceleration <= 0.0:
            raise ValueError("braking_acceleration must be finite and positive")
        scores = (
            surface_distance
            - closing_speed * closing_speed / (2.0 * braking_acceleration)
        )
    elif priority == "hocbf_slack":
        scores = batch.rows @ nominal - batch.lower_bounds
    else:
        raise ValueError(f"unsupported top-K priority: {priority}")
    return np.argsort(scores, kind="stable")[: min(top_k, scores.size)]


class UAVSafetyActionFilter:
    def __init__(
        self,
        config: SafetyFilterConfig,
        hocbf_config: HOCBFConfig | None = None,
    ) -> None:
        self.config = config
        self.hocbf_config = HOCBFConfig() if hocbf_config is None else hocbf_config
        self._actuator_rows, self._actuator_bounds = actuator_polygon_constraints(
            config.horizontal_acceleration_limit,
            config.vertical_acceleration_limit,
            config.actuator_facets,
        )

    def filter(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        nominal_acceleration: np.ndarray,
        obstacles: list[SphericalObstacle] | tuple[SphericalObstacle, ...],
        *,
        energy_gradient: np.ndarray | None = None,
        progress_direction: np.ndarray | None = None,
        _constraint_batch: _BarrierConstraintBatch | None = None,
    ) -> SafetyFilterOutput:
        total_started = perf_counter()
        nominal = np.asarray(nominal_acceleration, dtype=np.float64)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_acceleration must be a finite (3,) vector")
        build_started = perf_counter()
        obstacle_list = list(obstacles)
        if _constraint_batch is not None:
            if obstacle_list:
                raise ValueError("prebuilt constraints require an empty obstacle list")
            batch = _constraint_batch
        elif (
            self.config.method is SafetyFilterMethod.HOCBF
            and not self.config.sampled_data_robust
        ):
            batch = _vectorized_hocbf_batch(
                position,
                velocity,
                obstacle_list,
                self.hocbf_config,
            )
        else:
            constraints = self._build_constraints(
                position,
                velocity,
                nominal,
                obstacle_list,
            )
            batch = _BarrierConstraintBatch(
                rows=(
                    np.stack([constraint.row for constraint in constraints])
                    if constraints
                    else np.empty((0, 3), dtype=np.float64)
                ),
                lower_bounds=np.asarray(
                    [constraint.lower_bound for constraint in constraints],
                    dtype=np.float64,
                ),
                h=np.asarray([constraint.h for constraint in constraints], dtype=np.float64),
                h_dot=np.asarray(
                    [constraint.h_dot for constraint in constraints],
                    dtype=np.float64,
                ),
                psi1=np.asarray(
                    [constraint.psi1 for constraint in constraints],
                    dtype=np.float64,
                ),
                safe_distances=np.asarray(
                    [constraint.safe_distance for constraint in constraints],
                    dtype=np.float64,
                ),
            )
        candidate_count = int(batch.rows.shape[0])
        if self.config.top_k is None:
            selected_indices = np.arange(candidate_count)
        else:
            selected_indices = _top_k_batch_indices(
                batch,
                nominal,
                self.config.top_k,
                priority=self.config.top_k_priority,
                braking_acceleration=float(
                    np.hypot(
                        self.config.horizontal_acceleration_limit,
                        self.config.vertical_acceleration_limit,
                    )
                ),
            )
        dropped = candidate_count - int(selected_indices.size)
        build_seconds = perf_counter() - build_started

        barrier_rows = batch.rows[selected_indices]
        barrier_bounds = batch.lower_bounds[selected_indices]
        physical_rows = [self._actuator_rows]
        physical_bounds = [self._actuator_bounds]
        if self.config.horizontal_velocity_limit is not None:
            velocity_rows, velocity_bounds = velocity_polygon_constraints(
                velocity,
                dt=self.config.safety_dt,
                horizontal_limit=self.config.horizontal_velocity_limit,
                vertical_limit=self.config.vertical_velocity_limit,
                facets=self.config.actuator_facets,
            )
            physical_rows.append(velocity_rows)
            physical_bounds.append(velocity_bounds)
        rows = np.vstack((barrier_rows, *physical_rows))
        bounds = np.concatenate((barrier_bounds, *physical_bounds))

        hessian = np.eye(3, dtype=np.float64)
        center = nominal
        feasibility_margin = None
        feasibility_margin_duality_gap = None
        required_progress = None
        achieved_progress = None
        lexicographic_reason = None
        lexicographic_start = None
        if self.config.method is SafetyFilterMethod.SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF:
            if progress_direction is None:
                raise ValueError("lexicographic filter requires progress_direction")
            progress = np.asarray(progress_direction, dtype=np.float64)
            if progress.shape != (3,) or not np.all(np.isfinite(progress)):
                raise ValueError("progress_direction must be a finite (3,) vector")
            progress_norm = float(np.linalg.norm(progress))
            if progress_norm <= 1e-12:
                raise ValueError("progress_direction must be nonzero")
            progress = progress / progress_norm
            if barrier_rows.shape[0] > 0:
                margin_result = joint_feasibility_margin(
                    barrier_rows,
                    barrier_bounds,
                    self.config.horizontal_acceleration_limit,
                    self.config.vertical_acceleration_limit,
                )
                feasibility_margin = margin_result.margin
                feasibility_margin_duality_gap = margin_result.duality_gap
            progress_program = linprog(
                -progress,
                A_ub=-rows,
                b_ub=-bounds,
                bounds=[(None, None)] * 3,
                method="highs",
            )
            if progress_program.success:
                maximum_progress = float(progress @ progress_program.x)
                required_progress = min(float(progress @ nominal), maximum_progress)
                rows = np.vstack((rows, progress))
                bounds = np.concatenate((bounds, [required_progress]))
                lexicographic_start = np.asarray(progress_program.x, dtype=np.float64)
                hessian = self.config.energy_matrix / self.config.energy_characteristic
                minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(hessian)))
                if minimum_eigenvalue <= 1e-12:
                    raise ValueError(
                        "lexicographic energy objective requires a positive-definite physical energy matrix"
                    )
                center = np.zeros(3, dtype=np.float64)
                lexicographic_reason = "energy_minimization_at_maximally_preserved_nominal_progress"
            else:
                lexicographic_reason = "stage_one_safe_set_infeasible"
        if self.config.method in (
            SafetyFilterMethod.ENERGY_AWARE_HOCBF,
            SafetyFilterMethod.ENERGY_GRADIENT_HOCBF,
            SafetyFilterMethod.SAMPLED_DATA_ENERGY_AWARE_HOCBF,
            SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF,
        ):
            action_characteristic = float(
                np.hypot(
                    self.config.horizontal_acceleration_limit,
                    self.config.vertical_acceleration_limit,
                )
            )
            normalized_gradient = None
            gradient_weight = 0.0
            if self.config.method in (
                SafetyFilterMethod.ENERGY_GRADIENT_HOCBF,
                SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF,
            ):
                if energy_gradient is None:
                    raise ValueError("energy-gradient HOCBF requires energy_gradient")
                gradient = np.asarray(energy_gradient, dtype=np.float64)
                if gradient.shape != (3,) or not np.all(np.isfinite(gradient)):
                    raise ValueError("energy_gradient must be a finite (3,) vector")
                gradient_scale = self.config.energy_gradient_characteristic
                if self.config.energy_gradient_normalization == "unit_direction":
                    gradient_scale = max(
                        float(np.linalg.norm(gradient)) * action_characteristic,
                        1e-12,
                    )
                normalized_gradient = gradient / gradient_scale
                gradient_weight = self.config.energy_gradient_weight
            hessian, center = energy_aware_quadratic(
                nominal,
                np.eye(3) / action_characteristic**2,
                self.config.energy_matrix / self.config.energy_characteristic,
                energy_weight=self.config.energy_weight,
                dt=self.config.safety_dt,
                energy_gradient=normalized_gradient,
                gradient_weight=gradient_weight,
            )
        if (
            self.config.method
            is SafetyFilterMethod.SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF
            and lexicographic_start is not None
        ):
            solver_started = perf_counter()
            lexicographic_solution = minimize(
                lambda action: 0.5 * float(action @ hessian @ action),
                lexicographic_start,
                jac=lambda action: hessian @ action,
                method="SLSQP",
                constraints=[
                    {
                        "type": "ineq",
                        "fun": lambda action: rows @ action - bounds,
                        "jac": lambda action: rows,
                    }
                ],
                options={"ftol": 1e-10, "maxiter": 300, "disp": False},
            )
            lexicographic_action = np.asarray(
                lexicographic_solution.x,
                dtype=np.float64,
            )
            violations = bounds - rows @ lexicographic_action
            maximum_violation = float(max(0.0, np.max(violations)))
            projection = ProjectionResult(
                acceleration=lexicographic_action,
                feasible=maximum_violation <= 1e-6,
                converged=bool(lexicographic_solution.success),
                iterations=int(getattr(lexicographic_solution, "nit", 0)),
                max_violation=maximum_violation,
                intervention_norm=float(
                    np.linalg.norm(lexicographic_action - nominal)
                ),
                solver_time_seconds=float(perf_counter() - solver_started),
                active_constraints=int(
                    np.sum(np.abs(rows @ lexicographic_action - bounds) <= 1e-6)
                ),
                reason=(
                    "optimal"
                    if maximum_violation <= 1e-6
                    else f"slsqp:{lexicographic_solution.message}"
                ),
            )
        else:
            projection_solver = (
                project_polyhedral_qp_constraint_generation
                if rows.shape[0] >= 128
                else project_polyhedral_qp
            )
            projection = projection_solver(center, hessian, rows, bounds)
        executed = projection.acceleration
        if self.config.method is SafetyFilterMethod.SAMPLED_DATA_LEXICOGRAPHIC_ENERGY_HOCBF:
            if lexicographic_reason == "stage_one_safe_set_infeasible":
                projection = ProjectionResult(
                    acceleration=projection.acceleration,
                    feasible=False,
                    converged=False,
                    iterations=projection.iterations,
                    max_violation=projection.max_violation,
                    intervention_norm=projection.intervention_norm,
                    solver_time_seconds=projection.solver_time_seconds,
                    active_constraints=projection.active_constraints,
                    reason=lexicographic_reason,
                )
            achieved_progress = (
                None
                if progress_direction is None
                else float(
                    np.asarray(progress_direction, dtype=np.float64)
                    @ projection.acceleration
                    / np.linalg.norm(progress_direction)
                )
            )
        fallback_used = not projection.feasible
        fallback_satisfies = True
        if fallback_used:
            executed = emergency_braking_acceleration(
                velocity,
                self.config.horizontal_acceleration_limit,
                self.config.vertical_acceleration_limit,
            )
            fallback_satisfies = bool(np.all(rows @ executed >= bounds - 1e-8))

        nominal_slacks = batch.rows @ nominal - batch.lower_bounds
        executed_slacks = batch.rows @ executed - batch.lower_bounds
        total_seconds = perf_counter() - total_started
        diagnostics = SafetyFilterDiagnostics(
            method=self.config.method.value,
            feasible=projection.feasible,
            fallback_used=fallback_used,
            fallback_satisfies_constraints=fallback_satisfies,
            candidate_constraints=candidate_count,
            active_constraints=projection.active_constraints,
            dropped_constraints=dropped,
            minimum_h=(float(np.min(batch.h)) if candidate_count else None),
            minimum_psi1=(float(np.min(batch.psi1)) if candidate_count else None),
            minimum_nominal_slack=(
                float(np.min(nominal_slacks)) if candidate_count else None
            ),
            minimum_executed_slack=(
                float(np.min(executed_slacks)) if candidate_count else None
            ),
            intervention_norm=float(np.linalg.norm(executed - nominal)),
            constraint_build_seconds=build_seconds,
            solver_seconds=projection.solver_time_seconds,
            total_seconds=total_seconds,
            deadline_missed=total_seconds > self.config.deadline_seconds,
            solver_iterations=projection.iterations,
            solver_reason=(
                projection.reason
                if lexicographic_reason is None
                else f"{lexicographic_reason}:{projection.reason}"
            ),
            feasibility_margin=feasibility_margin,
            feasibility_margin_duality_gap=feasibility_margin_duality_gap,
            required_progress=required_progress,
            achieved_progress=achieved_progress,
        )
        projection_problem = ProjectionProblem(
            center=center,
            hessian=hessian,
            rows=rows,
            lower_bounds=bounds,
            projected_acceleration=projection.acceleration,
            barrier_constraint_count=barrier_rows.shape[0],
            feasible=projection.feasible,
            converged=projection.converged,
            solver_reason=projection.reason,
        )
        return SafetyFilterOutput(
            executed.copy(),
            diagnostics,
            projection_problem,
        )

    def filter_lidar_points(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        nominal_acceleration: np.ndarray,
        points: np.ndarray,
        *,
        point_radius: float = 0.0,
    ) -> SafetyFilterOutput:
        """Array-native path for static point obstacles emitted by LiDAR.

        Both ordinary and sampled-data robust HOCBF use the same equations as
        the object-based path.  The robust branch adds the vectorized form of
        ``hocbf_sampled_data_residual_bound`` to each lower bound.
        """

        if self.config.method is not SafetyFilterMethod.HOCBF:
            raise ValueError("LiDAR point fast path requires HOCBF")
        centers = np.asarray(points, dtype=np.float64)
        if centers.ndim != 2 or centers.shape[1:] != (3,) or not np.all(
            np.isfinite(centers)
        ):
            raise ValueError("points must be a finite (n, 3) array")
        radius = max(float(point_radius), 1e-9)
        if not np.isfinite(radius):
            raise ValueError("point_radius must be finite")
        count = centers.shape[0]
        position_array = np.asarray(position, dtype=np.float64)
        velocity_array = np.asarray(velocity, dtype=np.float64)
        obstacle_velocities = np.zeros((count, 3), dtype=np.float64)
        obstacle_accelerations = np.zeros((count, 3), dtype=np.float64)
        batch = _vectorized_hocbf_arrays(
            position_array,
            velocity_array,
            centers=centers,
            radii=np.full(count, radius, dtype=np.float64),
            obstacle_velocities=obstacle_velocities,
            obstacle_accelerations=obstacle_accelerations,
            config=self.hocbf_config,
        )
        if self.config.sampled_data_robust:
            residual_bounds = _sampled_data_residual_bounds_arrays(
                position_array,
                velocity_array,
                centers=centers,
                obstacle_velocities=obstacle_velocities,
                obstacle_accelerations=obstacle_accelerations,
                config=self.hocbf_config,
                hold_dt=self.config.safety_dt,
                horizontal_acceleration_limit=(
                    self.config.horizontal_acceleration_limit
                ),
                vertical_acceleration_limit=(
                    self.config.vertical_acceleration_limit
                ),
            )
            batch = _BarrierConstraintBatch(
                rows=batch.rows,
                lower_bounds=batch.lower_bounds + residual_bounds,
                h=batch.h,
                h_dot=batch.h_dot,
                psi1=batch.psi1,
                safe_distances=batch.safe_distances,
            )
        return self.filter(
            position,
            velocity,
            nominal_acceleration,
            (),
            _constraint_batch=batch,
        )

    def _build_constraints(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        nominal: np.ndarray,
        obstacles: list[SphericalObstacle],
    ) -> list[BarrierConstraint]:
        if not obstacles:
            return []
        if self.config.method is SafetyFilterMethod.ONE_STEP:
            constraints = [
                one_step_supporting_constraint(
                    position,
                    velocity,
                    nominal,
                    obstacle,
                    uav_radius=self.hocbf_config.uav_radius,
                    uncertainty_margin=self.hocbf_config.uncertainty_margin,
                    dt=self.config.safety_dt,
                )
                for obstacle in obstacles
            ]
            return constraints

        constraints = [
            sphere_hocbf_constraint(
                position,
                velocity,
                obstacle,
                self.hocbf_config,
            )
            for obstacle in obstacles
        ]
        if self.config.sampled_data_robust:
            if self.config.method is SafetyFilterMethod.AGGREGATE_HOCBF:
                raise ValueError(
                    "sampled-data strengthening is not derived for the aggregate barrier"
                )
            constraints = [
                strengthen_constraint_for_sample_hold(
                    constraint,
                    hocbf_sampled_data_residual_bound(
                        position,
                        velocity,
                        obstacle,
                        self.hocbf_config,
                        hold_dt=self.config.safety_dt,
                        horizontal_acceleration_limit=(
                            self.config.horizontal_acceleration_limit
                        ),
                        vertical_acceleration_limit=(
                            self.config.vertical_acceleration_limit
                        ),
                    ),
                )
                for constraint, obstacle in zip(constraints, obstacles)
            ]
        if self.config.method is SafetyFilterMethod.AGGREGATE_HOCBF:
            return [
                aggregate_hocbf_constraint(
                    constraints,
                    position,
                    velocity,
                    obstacles,
                    self.hocbf_config,
                    self.config.aggregate_rho,
                )
            ]
        return constraints
