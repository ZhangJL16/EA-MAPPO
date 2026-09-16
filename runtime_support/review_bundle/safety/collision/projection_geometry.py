from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


_DIMENSION = 3


def _matrix3(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (_DIMENSION, _DIMENSION) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite (3, 3) matrix")
    return matrix.copy()


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (_DIMENSION,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector.copy()


def _constraint_matrix(value: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != _DIMENSION:
        raise ValueError(f"{name} must have shape (m, 3)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite")
    return matrix.copy()


@dataclass(frozen=True)
class ProjectionProblem:
    center: np.ndarray
    hessian: np.ndarray
    rows: np.ndarray
    lower_bounds: np.ndarray
    projected_acceleration: np.ndarray
    barrier_constraint_count: int
    feasible: bool
    converged: bool
    solver_reason: str

    def __post_init__(self) -> None:
        center = _vector3(self.center, "center")
        hessian = _matrix3(self.hessian, "hessian")
        rows = _constraint_matrix(self.rows, "rows")
        bounds = np.asarray(self.lower_bounds, dtype=np.float64)
        if bounds.shape != (rows.shape[0],) or not np.all(np.isfinite(bounds)):
            raise ValueError("lower_bounds must be finite with shape (m,)")
        projected = _vector3(self.projected_acceleration, "projected_acceleration")
        barrier_count = int(self.barrier_constraint_count)
        if not 0 <= barrier_count <= rows.shape[0]:
            raise ValueError("barrier_constraint_count is outside the constraint rows")
        if np.min(np.linalg.eigvalsh(hessian)) <= 0.0:
            raise ValueError("hessian must be positive definite")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "hessian", hessian)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "lower_bounds", bounds.copy())
        object.__setattr__(self, "projected_acceleration", projected)
        object.__setattr__(self, "barrier_constraint_count", barrier_count)


@dataclass(frozen=True)
class ProjectionGeometry:
    jacobian_total: np.ndarray
    jacobian_barrier: np.ndarray
    jacobian_physical_total: np.ndarray
    jacobian_physical_barrier: np.ndarray
    singular_values: np.ndarray
    rank: int
    action_authority: float
    blocked_fraction: float
    normal_action_fraction: float
    active_barrier_constraints: int
    active_physical_constraints: int
    active_constraint_indices: tuple[int, ...]
    nominal_safe: bool
    minimum_nominal_slack: float | None
    minimum_executed_slack: float | None
    valid: bool
    active_set_stable: bool
    coordinate_map_stable: bool
    kkt_residual: float | None
    active_multiplier_minimum: float | None
    inactive_slack_minimum: float | None
    reason: str

    def __post_init__(self) -> None:
        for name in (
            "jacobian_total",
            "jacobian_barrier",
            "jacobian_physical_total",
            "jacobian_physical_barrier",
        ):
            object.__setattr__(self, name, _matrix3(getattr(self, name), name))
        singular_values = np.asarray(self.singular_values, dtype=np.float64)
        if singular_values.shape != (_DIMENSION,) or not np.all(np.isfinite(singular_values)):
            raise ValueError("singular_values must be a finite (3,) vector")
        object.__setattr__(self, "singular_values", singular_values.copy())
        if not 0 <= int(self.rank) <= _DIMENSION:
            raise ValueError("rank must lie in [0, 3]")
        if not np.isfinite(self.action_authority) or not np.isfinite(self.blocked_fraction):
            raise ValueError("authority diagnostics must be finite")
        if not np.isfinite(self.normal_action_fraction):
            raise ValueError("normal_action_fraction must be finite")

    @classmethod
    def invalid(
        cls,
        reason: str,
        *,
        nominal_safe: bool = False,
        minimum_nominal_slack: float | None = None,
        minimum_executed_slack: float | None = None,
    ) -> "ProjectionGeometry":
        zero = np.zeros((_DIMENSION, _DIMENSION), dtype=np.float64)
        return cls(
            jacobian_total=zero,
            jacobian_barrier=zero,
            jacobian_physical_total=zero,
            jacobian_physical_barrier=zero,
            singular_values=np.zeros(_DIMENSION, dtype=np.float64),
            rank=0,
            action_authority=0.0,
            blocked_fraction=1.0,
            normal_action_fraction=0.0,
            active_barrier_constraints=0,
            active_physical_constraints=0,
            active_constraint_indices=(),
            nominal_safe=bool(nominal_safe),
            minimum_nominal_slack=minimum_nominal_slack,
            minimum_executed_slack=minimum_executed_slack,
            valid=False,
            active_set_stable=False,
            coordinate_map_stable=False,
            kkt_residual=None,
            active_multiplier_minimum=None,
            inactive_slack_minimum=None,
            reason=str(reason),
        )

    def invalidated(self, reason: str) -> "ProjectionGeometry":
        return replace(
            self,
            valid=False,
            active_set_stable=False,
            reason=str(reason),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "jacobian_total": self.jacobian_total.tolist(),
            "jacobian_barrier": self.jacobian_barrier.tolist(),
            "jacobian_physical_total": self.jacobian_physical_total.tolist(),
            "jacobian_physical_barrier": self.jacobian_physical_barrier.tolist(),
            "singular_values": self.singular_values.tolist(),
            "rank": int(self.rank),
            "action_authority": float(self.action_authority),
            "blocked_fraction": float(self.blocked_fraction),
            "normal_action_fraction": float(self.normal_action_fraction),
            "active_barrier_constraints": int(self.active_barrier_constraints),
            "active_physical_constraints": int(self.active_physical_constraints),
            "active_constraint_indices": list(self.active_constraint_indices),
            "nominal_safe": bool(self.nominal_safe),
            "minimum_nominal_slack": self.minimum_nominal_slack,
            "minimum_executed_slack": self.minimum_executed_slack,
            "valid": bool(self.valid),
            "active_set_stable": bool(self.active_set_stable),
            "coordinate_map_stable": bool(self.coordinate_map_stable),
            "kkt_residual": self.kkt_residual,
            "active_multiplier_minimum": self.active_multiplier_minimum,
            "inactive_slack_minimum": self.inactive_slack_minimum,
            "reason": self.reason,
        }


def stable_pseudo_inverse(
    matrix: np.ndarray,
    *,
    relative_tolerance: float = 1e-10,
) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("matrix must be finite and two dimensional")
    if relative_tolerance <= 0.0:
        raise ValueError("relative_tolerance must be positive")
    left, singular_values, right_transpose = np.linalg.svd(values, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((values.shape[1], values.shape[0]), dtype=np.float64)
    threshold = relative_tolerance * max(values.shape) * max(float(singular_values[0]), 1.0)
    inverse_values = np.zeros_like(singular_values)
    retained = singular_values > threshold
    inverse_values[retained] = 1.0 / singular_values[retained]
    return (right_transpose.T * inverse_values) @ left.T


def weighted_projection_jacobian(
    active_rows: np.ndarray,
    hessian: np.ndarray,
    *,
    pseudo_inverse_tolerance: float = 1e-10,
) -> np.ndarray:
    rows = _constraint_matrix(active_rows, "active_rows")
    metric = _matrix3(hessian, "hessian")
    if np.min(np.linalg.eigvalsh(metric)) <= 0.0:
        raise ValueError("hessian must be positive definite")
    if rows.shape[0] == 0:
        return np.eye(_DIMENSION, dtype=np.float64)
    inverse = np.linalg.inv(metric)
    gram = rows @ inverse @ rows.T
    pseudo_inverse = stable_pseudo_inverse(
        gram,
        relative_tolerance=pseudo_inverse_tolerance,
    )
    jacobian = np.eye(_DIMENSION) - inverse @ rows.T @ pseudo_inverse @ rows
    return np.asarray(jacobian, dtype=np.float64)


def normalized_to_physical_action_jacobian(
    action: np.ndarray,
    *,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
    switching_tolerance: float = 1e-6,
) -> tuple[np.ndarray, bool, str]:
    normalized = _vector3(action, "action")
    if horizontal_acceleration_limit <= 0.0 or vertical_acceleration_limit <= 0.0:
        raise ValueError("acceleration limits must be positive")
    if switching_tolerance <= 0.0:
        raise ValueError("switching_tolerance must be positive")
    if np.any(np.abs(np.abs(normalized) - 1.0) <= switching_tolerance):
        return np.zeros((3, 3)), False, "normalized_box_switching_boundary"
    horizontal = normalized[:2]
    radius = float(np.linalg.norm(horizontal))
    jacobian = np.zeros((3, 3), dtype=np.float64)
    if abs(radius - 1.0) <= switching_tolerance:
        return jacobian, False, "horizontal_radial_switching_boundary"
    if radius < 1.0:
        jacobian[:2, :2] = horizontal_acceleration_limit * np.eye(2)
    else:
        direction = horizontal / radius
        jacobian[:2, :2] = (
            horizontal_acceleration_limit
            / radius
            * (np.eye(2) - np.outer(direction, direction))
        )
    jacobian[2, 2] = vertical_acceleration_limit
    return jacobian, True, "stable"


def physical_to_normalized_action_jacobian(
    acceleration: np.ndarray,
    *,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
    switching_tolerance: float = 1e-6,
) -> tuple[np.ndarray, bool, str]:
    physical = _vector3(acceleration, "acceleration")
    if horizontal_acceleration_limit <= 0.0 or vertical_acceleration_limit <= 0.0:
        raise ValueError("acceleration limits must be positive")
    scaled = np.array(
        [
            physical[0] / horizontal_acceleration_limit,
            physical[1] / horizontal_acceleration_limit,
            physical[2] / vertical_acceleration_limit,
        ],
        dtype=np.float64,
    )
    if np.any(np.abs(np.abs(scaled) - 1.0) <= switching_tolerance):
        return np.zeros((3, 3)), False, "physical_box_switching_boundary"
    horizontal = scaled[:2]
    radius = float(np.linalg.norm(horizontal))
    jacobian = np.zeros((3, 3), dtype=np.float64)
    if abs(radius - 1.0) <= switching_tolerance:
        return jacobian, False, "physical_radial_switching_boundary"
    if radius < 1.0:
        jacobian[:2, :2] = np.eye(2) / horizontal_acceleration_limit
    else:
        direction = horizontal / radius
        jacobian[:2, :2] = (
            1.0
            / (horizontal_acceleration_limit * radius)
            * (np.eye(2) - np.outer(direction, direction))
        )
    jacobian[2, 2] = 1.0 / vertical_acceleration_limit
    return jacobian, True, "stable"


def _active_set(
    problem: ProjectionProblem,
    *,
    active_tolerance: float,
    multiplier_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, float, float | None]:
    slacks = problem.rows @ problem.projected_acceleration - problem.lower_bounds
    tight = np.flatnonzero(np.abs(slacks) <= active_tolerance)
    stationarity = problem.hessian @ (
        problem.projected_acceleration - problem.center
    )
    if tight.size == 0:
        return tight, np.empty(0, dtype=np.float64), float(np.linalg.norm(stationarity)), None
    tight_rows = problem.rows[tight]
    multipliers = stable_pseudo_inverse(tight_rows.T) @ stationarity
    residual = float(np.linalg.norm(tight_rows.T @ multipliers - stationarity))
    positive = multipliers > multiplier_tolerance
    active = tight[positive]
    active_multipliers = multipliers[positive]
    minimum = None if active_multipliers.size == 0 else float(np.min(active_multipliers))
    return active, active_multipliers, residual, minimum


def analyze_projection_geometry(
    problem: ProjectionProblem,
    *,
    nominal_normalized_action: np.ndarray,
    executed_normalized_action: np.ndarray,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
    fallback_used: bool = False,
    emergency_brake: bool = False,
    minimum_nominal_slack: float | None = None,
    minimum_executed_slack: float | None = None,
    active_tolerance: float = 1e-6,
    inactive_switch_tolerance: float = 1e-5,
    multiplier_tolerance: float = 1e-8,
    kkt_tolerance: float = 1e-5,
    pseudo_inverse_tolerance: float = 1e-10,
    rank_tolerance: float = 1e-7,
) -> ProjectionGeometry:
    nominal_action = _vector3(nominal_normalized_action, "nominal_normalized_action")
    executed_action = _vector3(executed_normalized_action, "executed_normalized_action")
    barrier_rows = problem.rows[: problem.barrier_constraint_count]
    barrier_nominal_slacks = barrier_rows @ problem.center - problem.lower_bounds[
        : problem.barrier_constraint_count
    ]
    nominal_safe = bool(
        barrier_nominal_slacks.size == 0
        or np.all(barrier_nominal_slacks >= -active_tolerance)
    )
    if fallback_used:
        return ProjectionGeometry.invalid(
            "fallback_used",
            nominal_safe=nominal_safe,
            minimum_nominal_slack=minimum_nominal_slack,
            minimum_executed_slack=minimum_executed_slack,
        )
    if emergency_brake:
        return ProjectionGeometry.invalid(
            "emergency_brake",
            nominal_safe=nominal_safe,
            minimum_nominal_slack=minimum_nominal_slack,
            minimum_executed_slack=minimum_executed_slack,
        )
    if not problem.feasible or not problem.converged:
        return ProjectionGeometry.invalid(
            f"solver:{problem.solver_reason}",
            nominal_safe=nominal_safe,
            minimum_nominal_slack=minimum_nominal_slack,
            minimum_executed_slack=minimum_executed_slack,
        )
    active, multipliers, kkt_residual, minimum_multiplier = _active_set(
        problem,
        active_tolerance=active_tolerance,
        multiplier_tolerance=multiplier_tolerance,
    )
    full_slacks = problem.rows @ problem.projected_acceleration - problem.lower_bounds
    inactive_mask = np.ones(problem.rows.shape[0], dtype=bool)
    inactive_mask[active] = False
    inactive_minimum = (
        None
        if not np.any(inactive_mask)
        else float(np.min(full_slacks[inactive_mask]))
    )
    active_barrier = active[active < problem.barrier_constraint_count]
    active_physical = active[active >= problem.barrier_constraint_count]
    physical_total = weighted_projection_jacobian(
        problem.rows[active],
        problem.hessian,
        pseudo_inverse_tolerance=pseudo_inverse_tolerance,
    )
    physical_barrier = weighted_projection_jacobian(
        problem.rows[active_barrier],
        problem.hessian,
        pseudo_inverse_tolerance=pseudo_inverse_tolerance,
    )
    input_map, input_stable, input_reason = normalized_to_physical_action_jacobian(
        nominal_action,
        horizontal_acceleration_limit=horizontal_acceleration_limit,
        vertical_acceleration_limit=vertical_acceleration_limit,
    )
    output_map, output_stable, output_reason = physical_to_normalized_action_jacobian(
        problem.projected_acceleration,
        horizontal_acceleration_limit=horizontal_acceleration_limit,
        vertical_acceleration_limit=vertical_acceleration_limit,
    )
    normalized_total = output_map @ physical_total @ input_map
    normalized_barrier = output_map @ physical_barrier @ input_map
    singular_values = np.linalg.svd(normalized_total, compute_uv=False)
    rank = int(np.sum(singular_values > rank_tolerance))
    authority = float(np.clip(np.trace(normalized_total) / _DIMENSION, 0.0, 1.0))
    normal_fraction = float(
        np.linalg.norm((np.eye(_DIMENSION) - normalized_total) @ nominal_action)
        / (np.linalg.norm(nominal_action) + 1e-8)
    )
    coordinate_stable = bool(input_stable and output_stable)
    local_active_stable = bool(
        kkt_residual <= kkt_tolerance
        and (
            active.size == 0
            or (multipliers.size == active.size and np.all(multipliers > multiplier_tolerance))
        )
        and (inactive_minimum is None or inactive_minimum > inactive_switch_tolerance)
    )
    valid = bool(coordinate_stable and local_active_stable)
    if not input_stable:
        reason = input_reason
    elif not output_stable:
        reason = output_reason
    elif kkt_residual > kkt_tolerance:
        reason = "kkt_residual_too_large"
    elif active.size > 0 and minimum_multiplier is None:
        reason = "weakly_active_constraints"
    elif inactive_minimum is not None and inactive_minimum <= inactive_switch_tolerance:
        reason = "inactive_constraint_near_switch"
    else:
        reason = "valid_fixed_active_set"
    return ProjectionGeometry(
        jacobian_total=normalized_total,
        jacobian_barrier=normalized_barrier,
        jacobian_physical_total=physical_total,
        jacobian_physical_barrier=physical_barrier,
        singular_values=singular_values,
        rank=rank,
        action_authority=authority,
        blocked_fraction=1.0 - authority,
        normal_action_fraction=normal_fraction,
        active_barrier_constraints=int(active_barrier.size),
        active_physical_constraints=int(active_physical.size),
        active_constraint_indices=tuple(int(index) for index in active),
        nominal_safe=nominal_safe,
        minimum_nominal_slack=minimum_nominal_slack,
        minimum_executed_slack=minimum_executed_slack,
        valid=valid,
        active_set_stable=local_active_stable,
        coordinate_map_stable=coordinate_stable,
        kkt_residual=kkt_residual,
        active_multiplier_minimum=minimum_multiplier,
        inactive_slack_minimum=inactive_minimum,
        reason=reason,
    )


def identity_projection_geometry(
    nominal_normalized_action: np.ndarray,
    *,
    reason: str = "identity_nominal_passthrough",
) -> ProjectionGeometry:
    nominal = _vector3(nominal_normalized_action, "nominal_normalized_action")
    identity = np.eye(_DIMENSION, dtype=np.float64)
    return ProjectionGeometry(
        jacobian_total=identity,
        jacobian_barrier=identity,
        jacobian_physical_total=identity,
        jacobian_physical_barrier=identity,
        singular_values=np.ones(_DIMENSION, dtype=np.float64),
        rank=_DIMENSION,
        action_authority=1.0,
        blocked_fraction=0.0,
        normal_action_fraction=float(
            np.linalg.norm((identity - identity) @ nominal)
            / (np.linalg.norm(nominal) + 1e-8)
        ),
        active_barrier_constraints=0,
        active_physical_constraints=0,
        active_constraint_indices=(),
        nominal_safe=True,
        minimum_nominal_slack=None,
        minimum_executed_slack=None,
        valid=True,
        active_set_stable=True,
        coordinate_map_stable=True,
        kkt_residual=0.0,
        active_multiplier_minimum=None,
        inactive_slack_minimum=None,
        reason=reason,
    )
