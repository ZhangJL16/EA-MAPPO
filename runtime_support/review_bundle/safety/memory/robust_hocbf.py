from __future__ import annotations

import numpy as np

from review_bundle.safety.collision.hocbf import BarrierConstraint, HOCBFConfig

from .observer import IntervalObserverState
from .tube import box_support


def _vec3(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite (3,) vector")
    return vector


def directional_interval_hocbf_constraint(
    uav_position: np.ndarray,
    uav_velocity: np.ndarray,
    obstacle_state: IntervalObserverState,
    obstacle_radius: float,
    config: HOCBFConfig,
    *,
    identifier: str = "memory_obstacle",
    allow_uncertified_interval: bool = False,
) -> BarrierConstraint:
    """Robust second-order supporting-plane HOCBF for interval state error.

    The supporting direction is frozen at the current nominal relative
    position. Soundness is conditional on the obstacle position, velocity and
    acceleration errors lying in the supplied axis-aligned intervals.
    """

    if not obstacle_state.certification_valid and not allow_uncertified_interval:
        raise ValueError(
            "cannot construct a certified HOCBF from an invalid observer state: "
            f"{obstacle_state.invalid_reason}"
        )
    if not np.isfinite(obstacle_radius) or obstacle_radius < 0.0:
        raise ValueError("obstacle_radius must be finite and nonnegative")
    position = _vec3(uav_position, "uav_position")
    velocity = _vec3(uav_velocity, "uav_velocity")
    relative_position = position - obstacle_state.position
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1e-12:
        relative_velocity = velocity - obstacle_state.velocity
        distance = float(np.linalg.norm(relative_velocity))
        if distance <= 1e-12:
            raise ValueError("supporting direction is undefined")
        direction = -relative_velocity / distance
    else:
        direction = relative_position / distance
    safe_distance = float(obstacle_radius + config.uav_radius + config.uncertainty_margin)
    nominal_h = float(direction @ relative_position - safe_distance)
    nominal_h_dot = float(direction @ (velocity - obstacle_state.velocity))
    position_support = box_support(obstacle_state.position_radius, direction)
    velocity_support = box_support(obstacle_state.velocity_radius, direction)
    acceleration_support = box_support(obstacle_state.acceleration_radius, direction)
    robust_h = nominal_h - position_support
    robust_h_dot = nominal_h_dot - velocity_support
    robust_psi1 = robust_h_dot + config.k1 * robust_h
    nominal_drift = float(
        -direction @ obstacle_state.acceleration
        + (config.k1 + config.k2) * nominal_h_dot
        + config.k1 * config.k2 * nominal_h
    )
    uncertainty_margin = float(
        acceleration_support
        + (config.k1 + config.k2) * velocity_support
        + config.k1 * config.k2 * position_support
    )
    robust_drift = nominal_drift - uncertainty_margin
    return BarrierConstraint(
        row=direction,
        lower_bound=-robust_drift,
        h=robust_h,
        h_dot=robust_h_dot,
        psi1=robust_psi1,
        drift=robust_drift,
        safe_distance=safe_distance,
        identifier=identifier,
    )


def directional_hocbf_preconditions_hold(
    constraint: BarrierConstraint,
    *,
    tolerance: float = 0.0,
) -> bool:
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and nonnegative")
    return bool(
        constraint.h >= tolerance
        and constraint.psi1 >= tolerance
    )


def directional_hocbf_sample_hold_margin(
    uav_velocity: np.ndarray,
    obstacle_state: IntervalObserverState,
    direction: np.ndarray,
    config: HOCBFConfig,
    *,
    hold_dt: float,
    horizontal_acceleration_limit: float,
    vertical_acceleration_limit: float,
    true_obstacle_jerk_bound: np.ndarray | float,
    allow_uncertified_interval: bool = False,
) -> float:
    """Bound the possible decrease of directional psi2 over a ZOH hold."""

    velocity = _vec3(uav_velocity, "uav_velocity")
    normal = _vec3(direction, "direction")
    if not obstacle_state.certification_valid and not allow_uncertified_interval:
        raise ValueError(
            "cannot construct a certified hold margin from an invalid observer state: "
            f"{obstacle_state.invalid_reason}"
        )
    if abs(float(np.linalg.norm(normal)) - 1.0) > 1e-8:
        raise ValueError("direction must be unit length")
    if (
        hold_dt <= 0.0
        or horizontal_acceleration_limit <= 0.0
        or vertical_acceleration_limit <= 0.0
    ):
        raise ValueError("hold time and acceleration limits must be positive")
    jerk = np.broadcast_to(
        np.asarray(true_obstacle_jerk_bound, dtype=np.float64),
        (3,),
    ).copy()
    if not np.all(np.isfinite(jerk)) or np.any(jerk < 0.0):
        raise ValueError("true_obstacle_jerk_bound must be finite and nonnegative")
    input_bound = float(
        np.hypot(horizontal_acceleration_limit, vertical_acceleration_limit)
    )
    obstacle_acceleration_bound_at_sample = float(
        np.linalg.norm(obstacle_state.acceleration)
        + np.linalg.norm(obstacle_state.acceleration_radius)
    )
    obstacle_acceleration_bound = float(
        obstacle_acceleration_bound_at_sample
        + np.linalg.norm(jerk) * hold_dt
    )
    relative_velocity_bound = float(
        np.linalg.norm(velocity - obstacle_state.velocity)
        + np.linalg.norm(obstacle_state.velocity_radius)
        + (input_bound + obstacle_acceleration_bound) * hold_dt
    )
    derivative_bound = float(
        box_support(jerk, normal)
        + (config.k1 + config.k2)
        * (input_bound + obstacle_acceleration_bound)
        + config.k1 * config.k2 * relative_velocity_bound
    )
    return derivative_bound * float(hold_dt)
