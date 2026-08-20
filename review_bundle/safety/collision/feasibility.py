from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution, linprog, minimize


def _finite_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite ({size},) vector")
    return vector.copy()


def _constraint_arrays(rows: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(rows, dtype=np.float64)
    vector = np.asarray(bounds, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.all(np.isfinite(matrix)):
        raise ValueError("rows must be a finite (m, 3) matrix")
    if vector.shape != (matrix.shape[0],) or not np.all(np.isfinite(vector)):
        raise ValueError("bounds must be a finite (m,) vector aligned with rows")
    if matrix.shape[0] == 0:
        raise ValueError("at least one safety constraint is required")
    return matrix.copy(), vector.copy()


def cylindrical_input_support(
    direction: np.ndarray,
    horizontal_limit: float,
    vertical_limit: float,
) -> float:
    """Support function of ||u_xy|| <= a_xy and |u_z| <= a_z."""

    vector = _finite_vector(direction, 3, "direction")
    horizontal = float(horizontal_limit)
    vertical = float(vertical_limit)
    if horizontal <= 0.0 or vertical <= 0.0:
        raise ValueError("input limits must be positive")
    return float(horizontal * np.linalg.norm(vector[:2]) + vertical * abs(vector[2]))


def single_constraint_feasibility_margin(
    row: np.ndarray,
    bound: float,
    horizontal_limit: float,
    vertical_limit: float,
) -> float:
    return cylindrical_input_support(row, horizontal_limit, vertical_limit) - float(bound)


def _recover_kkt_dual_weights(
    matrix: np.ndarray,
    vector: np.ndarray,
    action: np.ndarray,
    primal_margin: float,
    horizontal_limit: float,
    vertical_limit: float,
) -> np.ndarray | None:
    """Recover a minimax dual certificate from primal KKT conditions.

    The support function is nonsmooth when its horizontal or vertical dual
    direction vanishes. SLSQP can stop at a nearby dual point in those cases.
    At a primal optimum, complementary slackness restricts nonzero dual
    weights to minimum-slack rows, while the weighted row direction belongs to
    the normal cone of the cylindrical input set at the maximizing action.
    Those conditions are linear in the weights and can be solved reliably by
    a tiny feasibility LP.
    """

    slacks = matrix @ action - vector
    horizontal_norm = float(np.linalg.norm(action[:2]))
    for active_tolerance in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        active = np.flatnonzero(slacks <= primal_margin + active_tolerance)
        active_rows = matrix[active]
        count = int(active.size)
        equality_rows = [np.ones(count, dtype=np.float64)]
        equality_bounds = [1.0]
        inequality_rows: list[np.ndarray] = []
        inequality_bounds: list[float] = []

        if horizontal_norm < horizontal_limit - 1e-6:
            equality_rows.extend((active_rows[:, 0], active_rows[:, 1]))
            equality_bounds.extend((0.0, 0.0))
        else:
            perpendicular = np.array(
                [-action[1], action[0]], dtype=np.float64
            ) / max(horizontal_norm, 1e-15)
            equality_rows.append(active_rows[:, :2] @ perpendicular)
            equality_bounds.append(0.0)
            inequality_rows.append(-(active_rows[:, :2] @ action[:2]))
            inequality_bounds.append(0.0)

        if action[2] > vertical_limit - 1e-6:
            inequality_rows.append(-active_rows[:, 2])
            inequality_bounds.append(0.0)
        elif action[2] < -vertical_limit + 1e-6:
            inequality_rows.append(active_rows[:, 2])
            inequality_bounds.append(0.0)
        else:
            equality_rows.append(active_rows[:, 2])
            equality_bounds.append(0.0)

        result = linprog(
            np.zeros(count, dtype=np.float64),
            A_ub=np.asarray(inequality_rows) if inequality_rows else None,
            b_ub=np.asarray(inequality_bounds) if inequality_rows else None,
            A_eq=np.asarray(equality_rows),
            b_eq=np.asarray(equality_bounds),
            bounds=[(0.0, None)] * count,
            method="highs",
        )
        if result.success:
            weights = np.zeros(matrix.shape[0], dtype=np.float64)
            weights[active] = result.x
            return weights
    return None


@dataclass(frozen=True)
class FeasibilityMarginResult:
    margin: float
    maximizing_action: np.ndarray
    dual_weights: np.ndarray
    primal_success: bool
    dual_success: bool
    duality_gap: float
    minimum_slack: float
    active_constraints: tuple[int, ...]
    solve_seconds: float
    global_certificate_fallback_used: bool

    @property
    def feasible(self) -> bool:
        return self.margin >= -1e-8


def joint_feasibility_margin(
    rows: np.ndarray,
    bounds: np.ndarray,
    horizontal_limit: float,
    vertical_limit: float,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 300,
    global_certificate_fallback: bool = False,
) -> FeasibilityMarginResult:
    """Compute max_u min_i (A_i u - b_i) over a cylindrical input set.

    The primal result is cross-checked against the minimax dual

        min_{lambda in simplex} sigma_U(A.T lambda) - b.T lambda.

    This is a pointwise feasibility diagnostic. It does not by itself imply
    that the margin remains nonnegative at a later sample.
    """

    matrix, vector = _constraint_arrays(rows, bounds)
    horizontal = float(horizontal_limit)
    vertical = float(vertical_limit)
    if horizontal <= 0.0 or vertical <= 0.0:
        raise ValueError("input limits must be positive")
    started = perf_counter()
    def primal_objective(candidate: np.ndarray) -> float:
        return -float(candidate[3])

    def primal_jacobian(candidate: np.ndarray) -> np.ndarray:
        del candidate
        return np.array([0.0, 0.0, 0.0, -1.0], dtype=np.float64)

    primal_constraints = [
        {
            "type": "ineq",
            "fun": lambda candidate: matrix @ candidate[:3] - vector - candidate[3],
            "jac": lambda candidate: np.column_stack(
                (matrix, -np.ones(matrix.shape[0], dtype=np.float64))
            ),
        },
        {
            "type": "ineq",
            "fun": lambda candidate: horizontal**2 - float(candidate[0] ** 2 + candidate[1] ** 2),
            "jac": lambda candidate: np.array(
                [-2.0 * candidate[0], -2.0 * candidate[1], 0.0, 0.0],
                dtype=np.float64,
            ),
        },
    ]
    action_starts = [np.zeros(3, dtype=np.float64)]
    for row in matrix:
        action = np.zeros(3, dtype=np.float64)
        horizontal_norm = float(np.linalg.norm(row[:2]))
        if horizontal_norm > 1e-14:
            action[:2] = horizontal * row[:2] / horizontal_norm
        action[2] = vertical * np.sign(row[2])
        action_starts.append(action)

    def solve_primal(action: np.ndarray):
        initial_margin = float(np.min(matrix @ action - vector))
        return minimize(
            primal_objective,
            np.concatenate((action, [initial_margin])),
            jac=primal_jacobian,
            method="SLSQP",
            bounds=[(-horizontal, horizontal), (-horizontal, horizontal), (-vertical, vertical), (None, None)],
            constraints=primal_constraints,
            options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
        )

    primal = solve_primal(action_starts[0])
    if not primal.success:
        candidates = [primal, *(solve_primal(action) for action in action_starts[1:])]
        primal = max(
            candidates,
            key=lambda item: float(np.min(matrix @ item.x[:3] - vector)),
        )
    action = np.asarray(primal.x[:3], dtype=np.float64)
    slacks = matrix @ action - vector
    primal_margin = float(np.min(slacks))

    constraint_count = matrix.shape[0]
    if constraint_count == 1:
        row = matrix[0]
        action = np.zeros(3, dtype=np.float64)
        horizontal_row_norm = float(np.linalg.norm(row[:2]))
        if horizontal_row_norm > 1e-14:
            action[:2] = horizontal * row[:2] / horizontal_row_norm
        action[2] = vertical * np.sign(row[2])
        slacks = matrix @ action - vector
        primal_margin = float(slacks[0])
        dual_weights = np.ones(1, dtype=np.float64)
        dual_value = single_constraint_feasibility_margin(
            matrix[0], vector[0], horizontal, vertical
        )
        dual_success = True
    else:
        def dual_objective(weights: np.ndarray) -> float:
            direction = matrix.T @ weights
            return cylindrical_input_support(direction, horizontal, vertical) - float(vector @ weights)

        def dual_jacobian(weights: np.ndarray) -> np.ndarray:
            direction = matrix.T @ weights
            horizontal_direction = direction[:2]
            horizontal_norm = float(np.linalg.norm(horizontal_direction))
            horizontal_gradient = np.zeros(3, dtype=np.float64)
            if horizontal_norm > 1e-12:
                horizontal_gradient[:2] = horizontal * horizontal_direction / horizontal_norm
            vertical_gradient = np.zeros(3, dtype=np.float64)
            if abs(direction[2]) > 1e-12:
                vertical_gradient[2] = vertical * np.sign(direction[2])
            return matrix @ (horizontal_gradient + vertical_gradient) - vector

        def solve_dual(initial_weights: np.ndarray):
            return minimize(
                dual_objective,
                initial_weights,
                jac=dual_jacobian,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * constraint_count,
                constraints=[
                    {
                        "type": "eq",
                        "fun": lambda weights: float(np.sum(weights) - 1.0),
                        "jac": lambda weights: np.ones(constraint_count, dtype=np.float64),
                    }
                ],
                options={"ftol": tolerance, "maxiter": max_iterations, "disp": False},
            )

        dual = solve_dual(
            np.full(constraint_count, 1.0 / constraint_count, dtype=np.float64),
        )
        provisional_gap = abs(primal_margin - float(dual_objective(dual.x)))
        if not dual.success or provisional_gap > max(1e-7, 100.0 * tolerance):
            starts = np.eye(constraint_count, dtype=np.float64)
            dual_candidates = [dual, *(solve_dual(start) for start in starts)]
            dual = min(dual_candidates, key=lambda item: float(dual_objective(item.x)))
        dual_weights = np.asarray(dual.x, dtype=np.float64)
        dual_weights = np.maximum(dual_weights, 0.0)
        dual_weights /= np.sum(dual_weights)
        dual_value = float(dual_objective(dual_weights))
        dual_success = bool(dual.success)

    if abs(primal_margin - dual_value) > max(1e-7, 100.0 * tolerance):
        dual_direction = matrix.T @ dual_weights
        dual_action = np.zeros(3, dtype=np.float64)
        dual_horizontal_norm = float(np.linalg.norm(dual_direction[:2]))
        if dual_horizontal_norm > 1e-14:
            dual_action[:2] = horizontal * dual_direction[:2] / dual_horizontal_norm
        dual_action[2] = vertical * np.sign(dual_direction[2])
        candidates = [
            primal,
            solve_primal(dual_action),
            *(solve_primal(action) for action in action_starts[1:]),
        ]
        primal = max(
            candidates,
            key=lambda item: float(np.min(matrix @ item.x[:3] - vector)),
        )
        action = np.asarray(primal.x[:3], dtype=np.float64)
        slacks = matrix @ action - vector
        primal_margin = float(np.min(slacks))

    if (
        not dual_success
        or abs(primal_margin - dual_value) > max(1e-7, 100.0 * tolerance)
    ):
        recovered_weights = _recover_kkt_dual_weights(
            matrix,
            vector,
            action,
            primal_margin,
            horizontal,
            vertical,
        )
        if recovered_weights is not None:
            recovered_direction = matrix.T @ recovered_weights
            recovered_value = cylindrical_input_support(
                recovered_direction,
                horizontal,
                vertical,
            ) - float(vector @ recovered_weights)
            if abs(primal_margin - recovered_value) < abs(primal_margin - dual_value):
                dual_weights = recovered_weights
                dual_value = float(recovered_value)
                dual_success = True

    global_fallback_used = False
    if constraint_count > 1 and global_certificate_fallback and (
        not dual_success
        or abs(primal_margin - dual_value) > max(1e-7, 100.0 * tolerance)
    ):
        def stick_breaking(parameters: np.ndarray) -> np.ndarray:
            remaining = 1.0
            weights = []
            for parameter in parameters:
                weights.append(remaining * float(parameter))
                remaining *= 1.0 - float(parameter)
            weights.append(remaining)
            return np.asarray(weights, dtype=np.float64)

        def global_dual_objective(parameters: np.ndarray) -> float:
            return dual_objective(stick_breaking(parameters))

        global_dual = differential_evolution(
            global_dual_objective,
            [(0.0, 1.0)] * (constraint_count - 1),
            seed=0,
            popsize=12,
            maxiter=500,
            tol=max(tolerance, 1e-10),
            polish=True,
            workers=1,
            updating="immediate",
        )
        global_weights = stick_breaking(global_dual.x)
        global_value = float(dual_objective(global_weights))
        if abs(primal_margin - global_value) < abs(primal_margin - dual_value):
            dual_weights = global_weights
            dual_value = global_value
            dual_success = bool(global_dual.success)
            global_fallback_used = True

    margin = primal_margin
    active_tolerance = max(1e-7, 10.0 * np.sqrt(tolerance))
    active = tuple(int(index) for index in np.flatnonzero(slacks <= primal_margin + active_tolerance))
    duality_gap = float(abs(primal_margin - dual_value))
    certificate_tolerance = max(1e-5, 1_000.0 * tolerance)
    primal_certified = bool(
        np.isfinite(primal_margin)
        and np.isfinite(dual_value)
        and duality_gap <= certificate_tolerance
    )
    dual_certified = bool(
        np.all(np.isfinite(dual_weights))
        and np.all(dual_weights >= -certificate_tolerance)
        and abs(float(np.sum(dual_weights)) - 1.0) <= certificate_tolerance
        and np.isfinite(dual_value)
        and duality_gap <= certificate_tolerance
    )
    return FeasibilityMarginResult(
        margin=float(margin),
        maximizing_action=action,
        dual_weights=dual_weights,
        primal_success=primal_certified,
        dual_success=dual_certified,
        duality_gap=duality_gap,
        minimum_slack=primal_margin,
        active_constraints=active,
        solve_seconds=float(perf_counter() - started),
        global_certificate_fallback_used=global_fallback_used,
    )


@dataclass(frozen=True)
class IntersampleClearanceResult:
    minimum_barrier: float
    minimum_clearance: float
    minimizing_time: float
    critical_times: tuple[float, ...]
    endpoint_minimum_barrier: float

    @property
    def safe(self) -> bool:
        return self.minimum_barrier >= -1e-10


def exact_zoh_sphere_clearance(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    relative_acceleration: np.ndarray,
    safe_distance: float,
    hold_dt: float,
) -> IntersampleClearanceResult:
    """Exact minimum sphere barrier over one constant-acceleration hold.

    For r(t)=r+v*t+0.5*a*t^2, the barrier ||r(t)||^2-d^2 is quartic.
    Its derivative is cubic, so the exact interval minimum is attained at an
    endpoint or at a real cubic root inside the hold interval.
    """

    position = _finite_vector(relative_position, 3, "relative_position")
    velocity = _finite_vector(relative_velocity, 3, "relative_velocity")
    acceleration = _finite_vector(relative_acceleration, 3, "relative_acceleration")
    distance = float(safe_distance)
    interval = float(hold_dt)
    if distance < 0.0 or interval <= 0.0:
        raise ValueError("safe_distance must be nonnegative and hold_dt positive")

    coefficients = np.asarray(
        [
            float(acceleration @ acceleration),
            3.0 * float(velocity @ acceleration),
            2.0 * float(position @ acceleration + velocity @ velocity),
            2.0 * float(position @ velocity),
        ],
        dtype=np.float64,
    )
    nonzero = np.flatnonzero(np.abs(coefficients) > 1e-14)
    roots: list[float] = []
    if nonzero.size:
        polynomial = coefficients[nonzero[0] :]
        for root in np.roots(polynomial):
            if abs(float(np.imag(root))) <= 1e-9:
                value = float(np.real(root))
                if 1e-12 < value < interval - 1e-12:
                    roots.append(value)
    candidate_times = sorted({0.0, interval, *roots})

    def barrier(time: float) -> tuple[float, float]:
        relative = position + velocity * time + 0.5 * acceleration * time**2
        norm = float(np.linalg.norm(relative))
        return norm**2 - distance**2, norm - distance

    values = [barrier(time) for time in candidate_times]
    minimum_index = int(np.argmin([value[0] for value in values]))
    endpoint_minimum = min(values[0][0], values[-1][0])
    return IntersampleClearanceResult(
        minimum_barrier=float(values[minimum_index][0]),
        minimum_clearance=float(values[minimum_index][1]),
        minimizing_time=float(candidate_times[minimum_index]),
        critical_times=tuple(float(time) for time in candidate_times),
        endpoint_minimum_barrier=float(endpoint_minimum),
    )


def exact_constant_jerk_sphere_clearance(
    relative_position: np.ndarray,
    relative_velocity: np.ndarray,
    relative_acceleration: np.ndarray,
    relative_jerk: np.ndarray,
    safe_distance: float,
    hold_dt: float,
) -> IntersampleClearanceResult:
    """Exact sphere clearance for a constant-relative-jerk hold.

    The relative position is a cubic polynomial.  Its squared norm is degree
    six and its derivative is degree five, so every interval minimum occurs at
    an endpoint or a real root of that derivative inside the hold interval.
    """

    position = _finite_vector(relative_position, 3, "relative_position")
    velocity = _finite_vector(relative_velocity, 3, "relative_velocity")
    acceleration = _finite_vector(
        relative_acceleration,
        3,
        "relative_acceleration",
    )
    jerk = _finite_vector(relative_jerk, 3, "relative_jerk")
    distance = float(safe_distance)
    interval = float(hold_dt)
    if distance < 0.0 or interval <= 0.0:
        raise ValueError("safe_distance must be nonnegative and hold_dt positive")

    coordinate_polynomials = [
        np.polynomial.Polynomial(
            [
                position[axis],
                velocity[axis],
                0.5 * acceleration[axis],
                jerk[axis] / 6.0,
            ]
        )
        for axis in range(3)
    ]
    squared_distance = sum(
        (polynomial * polynomial for polynomial in coordinate_polynomials),
        start=np.polynomial.Polynomial([0.0]),
    )
    derivative = squared_distance.deriv()
    roots: list[float] = []
    if np.any(np.abs(derivative.coef) > 1e-14):
        for root in derivative.roots():
            if abs(float(np.imag(root))) <= 1e-9:
                value = float(np.real(root))
                if 1e-12 < value < interval - 1e-12:
                    roots.append(value)
    candidate_times = sorted({0.0, interval, *roots})

    def barrier(time: float) -> tuple[float, float]:
        relative = (
            position
            + velocity * time
            + 0.5 * acceleration * time**2
            + jerk * time**3 / 6.0
        )
        norm = float(np.linalg.norm(relative))
        return norm**2 - distance**2, norm - distance

    values = [barrier(time) for time in candidate_times]
    minimum_index = int(np.argmin([value[0] for value in values]))
    endpoint_minimum = min(values[0][0], values[-1][0])
    return IntersampleClearanceResult(
        minimum_barrier=float(values[minimum_index][0]),
        minimum_clearance=float(values[minimum_index][1]),
        minimizing_time=float(candidate_times[minimum_index]),
        critical_times=tuple(float(time) for time in candidate_times),
        endpoint_minimum_barrier=float(endpoint_minimum),
    )


@dataclass(frozen=True)
class EnergyOptimalActionResult:
    action: np.ndarray
    objective: float
    feasible: bool
    minimum_safety_slack: float
    progress_slack: float | None
    solve_seconds: float


def energy_optimal_safe_action(
    rows: np.ndarray,
    bounds: np.ndarray,
    energy_matrix: np.ndarray,
    horizontal_limit: float,
    vertical_limit: float,
    *,
    required_margin: float = 0.0,
    progress_direction: np.ndarray | None = None,
    minimum_progress: float | None = None,
    initial_action: np.ndarray | None = None,
) -> EnergyOptimalActionResult:
    """Minimize u.T R u over a fixed hard-safe action set.

    The result has pointwise energy optimality only. No trajectory-energy
    dominance follows without an additional value/supersolution argument.
    """

    matrix, vector = _constraint_arrays(rows, bounds)
    energy = np.asarray(energy_matrix, dtype=np.float64)
    if energy.shape != (3, 3) or not np.all(np.isfinite(energy)):
        raise ValueError("energy_matrix must be finite with shape (3, 3)")
    energy = 0.5 * (energy + energy.T)
    if float(np.min(np.linalg.eigvalsh(energy))) < -1e-10:
        raise ValueError("energy_matrix must be positive semidefinite")
    horizontal = float(horizontal_limit)
    vertical = float(vertical_limit)
    if horizontal <= 0.0 or vertical <= 0.0:
        raise ValueError("input limits must be positive")
    progress = None
    if progress_direction is not None:
        progress = _finite_vector(progress_direction, 3, "progress_direction")
        if minimum_progress is None:
            raise ValueError("minimum_progress is required with progress_direction")
    elif minimum_progress is not None:
        raise ValueError("progress_direction is required with minimum_progress")

    initial = np.zeros(3, dtype=np.float64) if initial_action is None else _finite_vector(initial_action, 3, "initial_action")
    initial[2] = np.clip(initial[2], -vertical, vertical)
    horizontal_norm = float(np.linalg.norm(initial[:2]))
    if horizontal_norm > horizontal:
        initial[:2] *= horizontal / horizontal_norm
    target_bounds = vector + float(required_margin)
    constraints: list[dict[str, object]] = [
        {"type": "ineq", "fun": lambda action: matrix @ action - target_bounds},
        {"type": "ineq", "fun": lambda action: horizontal**2 - float(action[0] ** 2 + action[1] ** 2)},
    ]
    if progress is not None:
        constraints.append(
            {"type": "ineq", "fun": lambda action: float(progress @ action - float(minimum_progress))}
        )
    started = perf_counter()
    result = minimize(
        lambda action: float(action @ energy @ action),
        initial,
        method="SLSQP",
        bounds=[(-horizontal, horizontal), (-horizontal, horizontal), (-vertical, vertical)],
        constraints=constraints,
        options={"ftol": 1e-10, "maxiter": 300, "disp": False},
    )
    action = np.asarray(result.x, dtype=np.float64)
    minimum_slack = float(np.min(matrix @ action - target_bounds))
    progress_slack = None if progress is None else float(progress @ action - float(minimum_progress))
    feasible = bool(
        result.success
        and minimum_slack >= -1e-7
        and horizontal**2 - float(action[0] ** 2 + action[1] ** 2) >= -1e-7
        and abs(action[2]) <= vertical + 1e-7
        and (progress_slack is None or progress_slack >= -1e-7)
    )
    return EnergyOptimalActionResult(
        action=action,
        objective=float(action @ energy @ action),
        feasible=feasible,
        minimum_safety_slack=minimum_slack,
        progress_slack=progress_slack,
        solve_seconds=float(perf_counter() - started),
    )


def energy_to_go_smooth_upper_bound(
    reference_value: float,
    next_state_gradient: np.ndarray,
    control_to_state_jacobian: np.ndarray,
    action_delta: np.ndarray,
    gradient_lipschitz_constant: float,
) -> float:
    """Descent-lemma upper bound under an explicitly supplied smoothness constant."""

    gradient = np.asarray(next_state_gradient, dtype=np.float64)
    jacobian = np.asarray(control_to_state_jacobian, dtype=np.float64)
    delta = _finite_vector(action_delta, 3, "action_delta")
    if jacobian.ndim != 2 or jacobian.shape[1] != 3 or gradient.shape != (jacobian.shape[0],):
        raise ValueError("gradient and control_to_state_jacobian shapes are incompatible")
    constant = float(gradient_lipschitz_constant)
    if not np.isfinite(reference_value) or constant < 0.0 or not np.isfinite(constant):
        raise ValueError("reference value must be finite and Lipschitz constant nonnegative")
    state_delta = jacobian @ delta
    return float(reference_value + gradient @ state_delta + 0.5 * constant * (state_delta @ state_delta))


__all__ = [
    "EnergyOptimalActionResult",
    "FeasibilityMarginResult",
    "IntersampleClearanceResult",
    "cylindrical_input_support",
    "energy_optimal_safe_action",
    "energy_to_go_smooth_upper_bound",
    "exact_zoh_sphere_clearance",
    "joint_feasibility_margin",
    "single_constraint_feasibility_margin",
]
