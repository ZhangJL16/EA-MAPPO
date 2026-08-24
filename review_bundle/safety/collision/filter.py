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
    ) -> SafetyFilterOutput:
        total_started = perf_counter()
        nominal = np.asarray(nominal_acceleration, dtype=np.float64)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal_acceleration must be a finite (3,) vector")
        build_started = perf_counter()
        constraints = self._build_constraints(
            position,
            velocity,
            nominal,
            list(obstacles),
        )
        candidate_count = len(constraints)
        selected = constraints
        if self.config.top_k is not None:
            selected = select_top_k_constraints(
                constraints,
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
        dropped = candidate_count - len(selected)
        build_seconds = perf_counter() - build_started

        barrier_rows = (
            np.stack([constraint.row for constraint in selected])
            if selected
            else np.empty((0, 3), dtype=np.float64)
        )
        barrier_bounds = np.asarray(
            [constraint.lower_bound for constraint in selected],
            dtype=np.float64,
        )
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
            projection = project_polyhedral_qp(
                center,
                hessian,
                rows,
                bounds,
            )
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

        nominal_slacks = [constraint.slack(nominal) for constraint in constraints]
        executed_slacks = [constraint.slack(executed) for constraint in constraints]
        total_seconds = perf_counter() - total_started
        diagnostics = SafetyFilterDiagnostics(
            method=self.config.method.value,
            feasible=projection.feasible,
            fallback_used=fallback_used,
            fallback_satisfies_constraints=fallback_satisfies,
            candidate_constraints=candidate_count,
            active_constraints=projection.active_constraints,
            dropped_constraints=dropped,
            minimum_h=min((constraint.h for constraint in constraints), default=None),
            minimum_psi1=min(
                (constraint.psi1 for constraint in constraints),
                default=None,
            ),
            minimum_nominal_slack=min(nominal_slacks, default=None),
            minimum_executed_slack=min(executed_slacks, default=None),
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
