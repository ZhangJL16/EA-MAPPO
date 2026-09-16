from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linprog


class HistoryTubeMode(str, Enum):
    HISTORY_INFORMED = "history_informed"
    ROBUST_BASE = "robust_base"
    INFEASIBLE = "infeasible"


@dataclass(frozen=True)
class SetMembershipBounds:
    feasible: bool
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    window_length: int

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "acceleration"):
            interval = np.asarray(getattr(self, name), dtype=np.float64)
            if interval.shape != (3, 2):
                raise ValueError(f"{name} must have shape (3, 2)")
            if self.feasible and (
                not np.all(np.isfinite(interval))
                or np.any(interval[:, 0] > interval[:, 1])
            ):
                raise ValueError(f"{name} must contain finite ordered intervals")
            object.__setattr__(self, name, interval.copy())

    @property
    def center_position(self) -> np.ndarray:
        return np.mean(self.position, axis=1)

    @property
    def center_velocity(self) -> np.ndarray:
        return np.mean(self.velocity, axis=1)

    @property
    def center_acceleration(self) -> np.ndarray:
        return np.mean(self.acceleration, axis=1)

    @property
    def radius_position(self) -> np.ndarray:
        return 0.5 * (self.position[:, 1] - self.position[:, 0])

    @property
    def radius_velocity(self) -> np.ndarray:
        return 0.5 * (self.velocity[:, 1] - self.velocity[:, 0])

    @property
    def radius_acceleration(self) -> np.ndarray:
        return 0.5 * (self.acceleration[:, 1] - self.acceleration[:, 0])


@dataclass(frozen=True)
class AdaptiveHistoryTubeResult:
    mode: HistoryTubeMode
    bounds: SetMembershipBounds
    jerk_bound_used: np.ndarray
    tested_windows: tuple[int, ...]
    requires_history_motion_bound_assumption: bool

    def __post_init__(self) -> None:
        jerk = np.asarray(self.jerk_bound_used, dtype=np.float64)
        if jerk.shape != (3,) or np.any(jerk < 0.0) or not np.all(np.isfinite(jerk)):
            raise ValueError("jerk_bound_used must be a finite nonnegative (3,) vector")
        object.__setattr__(self, "jerk_bound_used", jerk.copy())

    @property
    def certified_under_declared_robust_model(self) -> bool:
        return (
            self.mode is HistoryTubeMode.ROBUST_BASE
            and self.bounds.feasible
            and not self.requires_history_motion_bound_assumption
        )


def _positive_vector(value: np.ndarray | float, name: str, *, allow_zero: bool) -> np.ndarray:
    vector = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()
    lower_ok = vector >= 0.0 if allow_zero else vector > 0.0
    if not np.all(np.isfinite(vector)) or not np.all(lower_ok):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} scalar or (3,) vector")
    return vector


def _axis_membership_bounds(
    times: np.ndarray,
    measurements: np.ndarray,
    *,
    sensor_error: float,
    velocity_bound: float,
    acceleration_bound: float,
    jerk_bound: float,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    count = times.size
    jerk_count = count - 1
    position_offset = 0
    velocity_offset = count
    acceleration_offset = 2 * count
    jerk_offset = 3 * count
    variables = 3 * count + jerk_count

    equalities: list[np.ndarray] = []
    equality_rhs: list[float] = []
    for index, delta_time in enumerate(np.diff(times)):
        row_position = np.zeros(variables)
        row_position[position_offset + index + 1] = 1.0
        row_position[position_offset + index] = -1.0
        row_position[velocity_offset + index] = -delta_time
        row_position[acceleration_offset + index] = -0.5 * delta_time**2
        row_position[jerk_offset + index] = -(delta_time**3) / 6.0
        equalities.append(row_position)
        equality_rhs.append(0.0)

        row_velocity = np.zeros(variables)
        row_velocity[velocity_offset + index + 1] = 1.0
        row_velocity[velocity_offset + index] = -1.0
        row_velocity[acceleration_offset + index] = -delta_time
        row_velocity[jerk_offset + index] = -0.5 * delta_time**2
        equalities.append(row_velocity)
        equality_rhs.append(0.0)

        row_acceleration = np.zeros(variables)
        row_acceleration[acceleration_offset + index + 1] = 1.0
        row_acceleration[acceleration_offset + index] = -1.0
        row_acceleration[jerk_offset + index] = -delta_time
        equalities.append(row_acceleration)
        equality_rhs.append(0.0)

    inequalities: list[np.ndarray] = []
    inequality_rhs: list[float] = []
    for index, measurement in enumerate(measurements):
        upper = np.zeros(variables)
        upper[position_offset + index] = 1.0
        inequalities.append(upper)
        inequality_rhs.append(float(measurement + sensor_error))
        lower = np.zeros(variables)
        lower[position_offset + index] = -1.0
        inequalities.append(lower)
        inequality_rhs.append(float(-measurement + sensor_error))

    variable_bounds = (
        [(None, None)] * count
        + [(-velocity_bound, velocity_bound)] * count
        + [(-acceleration_bound, acceleration_bound)] * count
        + [(-jerk_bound, jerk_bound)] * jerk_count
    )
    a_eq = np.asarray(equalities, dtype=np.float64)
    b_eq = np.asarray(equality_rhs, dtype=np.float64)
    a_ub = np.asarray(inequalities, dtype=np.float64)
    b_ub = np.asarray(inequality_rhs, dtype=np.float64)

    current_indices = (
        position_offset + count - 1,
        velocity_offset + count - 1,
        acceleration_offset + count - 1,
    )
    intervals: list[np.ndarray] = []
    for current_index in current_indices:
        endpoints: list[float] = []
        for sign in (1.0, -1.0):
            objective = np.zeros(variables)
            objective[current_index] = sign
            result = linprog(
                objective,
                A_ub=a_ub,
                b_ub=b_ub,
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=variable_bounds,
                method="highs",
            )
            if not result.success:
                empty = np.full(2, np.nan)
                return False, empty, empty, empty
            endpoints.append(float(result.fun * sign))
        intervals.append(np.array([endpoints[0], endpoints[1]]))
    return True, intervals[0], intervals[1], intervals[2]


def jerk_bounded_set_membership(
    times: np.ndarray,
    measured_positions: np.ndarray,
    *,
    sensor_error: np.ndarray | float,
    velocity_bound: np.ndarray | float,
    acceleration_bound: np.ndarray | float,
    jerk_bound: np.ndarray | float,
) -> SetMembershipBounds:
    timestamps = np.asarray(times, dtype=np.float64)
    measurements = np.asarray(measured_positions, dtype=np.float64)
    if timestamps.ndim != 1 or timestamps.size < 2:
        raise ValueError("times must be a vector with at least two samples")
    if measurements.shape != (timestamps.size, 3):
        raise ValueError("measured_positions must have shape (len(times), 3)")
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise ValueError("times must be finite and strictly increasing")
    if not np.all(np.isfinite(measurements)):
        raise ValueError("measured_positions must be finite")
    sensor = _positive_vector(sensor_error, "sensor_error", allow_zero=True)
    velocity = _positive_vector(velocity_bound, "velocity_bound", allow_zero=False)
    acceleration = _positive_vector(
        acceleration_bound, "acceleration_bound", allow_zero=False
    )
    jerk = _positive_vector(jerk_bound, "jerk_bound", allow_zero=True)

    position_intervals = np.empty((3, 2))
    velocity_intervals = np.empty((3, 2))
    acceleration_intervals = np.empty((3, 2))
    for axis in range(3):
        feasible, position, axis_velocity, axis_acceleration = _axis_membership_bounds(
            timestamps,
            measurements[:, axis],
            sensor_error=float(sensor[axis]),
            velocity_bound=float(velocity[axis]),
            acceleration_bound=float(acceleration[axis]),
            jerk_bound=float(jerk[axis]),
        )
        if not feasible:
            invalid = np.full((3, 2), np.nan)
            return SetMembershipBounds(
                feasible=False,
                position=invalid,
                velocity=invalid,
                acceleration=invalid,
                window_length=int(timestamps.size),
            )
        position_intervals[axis] = position
        velocity_intervals[axis] = axis_velocity
        acceleration_intervals[axis] = axis_acceleration
    return SetMembershipBounds(
        feasible=True,
        position=position_intervals,
        velocity=velocity_intervals,
        acceleration=acceleration_intervals,
        window_length=int(timestamps.size),
    )


def adaptive_history_set_membership(
    times: np.ndarray,
    measured_positions: np.ndarray,
    *,
    sensor_error: np.ndarray | float,
    velocity_bound: np.ndarray | float,
    acceleration_bound: np.ndarray | float,
    history_jerk_bound: np.ndarray | float,
    robust_jerk_bound: np.ndarray | float,
    candidate_windows: tuple[int, ...] = (2, 4, 8, 16),
    minimum_confident_window: int = 4,
) -> AdaptiveHistoryTubeResult:
    timestamps = np.asarray(times, dtype=np.float64)
    measurements = np.asarray(measured_positions, dtype=np.float64)
    available = tuple(
        sorted({int(length) for length in candidate_windows if 2 <= length <= len(timestamps)}, reverse=True)
    )
    if not available:
        raise ValueError("candidate_windows must contain an available length of at least two")
    if minimum_confident_window < 2:
        raise ValueError("minimum_confident_window must be at least two")
    history_jerk = _positive_vector(
        history_jerk_bound, "history_jerk_bound", allow_zero=True
    )
    robust_jerk = _positive_vector(
        robust_jerk_bound, "robust_jerk_bound", allow_zero=True
    )
    if np.any(history_jerk > robust_jerk):
        raise ValueError("history_jerk_bound must not exceed robust_jerk_bound")

    tested: list[int] = []
    for length in available:
        if length < minimum_confident_window:
            continue
        tested.append(length)
        bounds = jerk_bounded_set_membership(
            timestamps[-length:],
            measurements[-length:],
            sensor_error=sensor_error,
            velocity_bound=velocity_bound,
            acceleration_bound=acceleration_bound,
            jerk_bound=history_jerk,
        )
        if bounds.feasible:
            return AdaptiveHistoryTubeResult(
                mode=HistoryTubeMode.HISTORY_INFORMED,
                bounds=bounds,
                jerk_bound_used=history_jerk,
                tested_windows=tuple(tested),
                requires_history_motion_bound_assumption=True,
            )

    robust_length = min(available)
    robust_bounds = jerk_bounded_set_membership(
        timestamps[-robust_length:],
        measurements[-robust_length:],
        sensor_error=sensor_error,
        velocity_bound=velocity_bound,
        acceleration_bound=acceleration_bound,
        jerk_bound=robust_jerk,
    )
    return AdaptiveHistoryTubeResult(
        mode=(HistoryTubeMode.ROBUST_BASE if robust_bounds.feasible else HistoryTubeMode.INFEASIBLE),
        bounds=robust_bounds,
        jerk_bound_used=robust_jerk,
        tested_windows=tuple(tested),
        requires_history_motion_bound_assumption=False,
    )


def future_position_error_tube(
    horizons: np.ndarray,
    bounds: SetMembershipBounds,
    *,
    future_jerk_bound: np.ndarray | float,
    true_state_containment_verified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not bounds.feasible:
        raise ValueError("bounds must be feasible")
    if not true_state_containment_verified:
        raise ValueError(
            "future tube semantics require an independently valid premise that "
            "the true present state is contained in bounds"
        )
    horizon = np.asarray(horizons, dtype=np.float64)
    if horizon.ndim != 1 or np.any(horizon < 0.0) or not np.all(np.isfinite(horizon)):
        raise ValueError("horizons must be a finite nonnegative vector")
    jerk = _positive_vector(future_jerk_bound, "future_jerk_bound", allow_zero=True)
    component_radius = (
        bounds.radius_position[None, :]
        + horizon[:, None] * bounds.radius_velocity[None, :]
        + 0.5 * horizon[:, None] ** 2 * bounds.radius_acceleration[None, :]
        + (horizon[:, None] ** 3 / 6.0) * jerk[None, :]
    )
    euclidean_radius = np.linalg.norm(component_radius, axis=1)
    return component_radius, euclidean_radius


def switchable_acceleration_future_tube_lower_bound(
    horizons: np.ndarray,
    future_acceleration_bound: np.ndarray | float,
) -> np.ndarray:
    horizon = np.asarray(horizons, dtype=np.float64)
    if horizon.ndim != 1 or np.any(horizon < 0.0) or not np.all(np.isfinite(horizon)):
        raise ValueError("horizons must be a finite nonnegative vector")
    acceleration = _positive_vector(
        future_acceleration_bound, "future_acceleration_bound", allow_zero=True
    )
    return 0.5 * horizon[:, None] ** 2 * acceleration[None, :]


def bounded_jerk_future_tube_lower_bound(
    horizons: np.ndarray,
    future_jerk_bound: np.ndarray | float,
) -> np.ndarray:
    horizon = np.asarray(horizons, dtype=np.float64)
    if horizon.ndim != 1 or np.any(horizon < 0.0) or not np.all(np.isfinite(horizon)):
        raise ValueError("horizons must be a finite nonnegative vector")
    jerk = _positive_vector(
        future_jerk_bound, "future_jerk_bound", allow_zero=True
    )
    return horizon[:, None] ** 3 / 6.0 * jerk[None, :]
