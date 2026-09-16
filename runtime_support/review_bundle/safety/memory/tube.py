from __future__ import annotations

import numpy as np


def _nonnegative_vec3(value: np.ndarray | float, name: str) -> np.ndarray:
    vector = np.broadcast_to(np.asarray(value, dtype=np.float64), (3,)).copy()
    if not np.all(np.isfinite(vector)) or np.any(vector < 0.0):
        raise ValueError(f"{name} must be a finite nonnegative scalar or (3,) vector")
    return vector


def box_support(radius: np.ndarray | float, direction: np.ndarray) -> float:
    half_width = _nonnegative_vec3(radius, "radius")
    normal = np.asarray(direction, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ValueError("direction must be a finite (3,) vector")
    return float(np.abs(normal) @ half_width)


def isotropic_enclosing_radius(radius: np.ndarray | float) -> float:
    return float(np.linalg.norm(_nonnegative_vec3(radius, "radius")))


def future_position_box(
    position_radius: np.ndarray | float,
    velocity_radius: np.ndarray | float,
    acceleration_radius: np.ndarray | float,
    jerk_residual_bound: np.ndarray | float,
    horizons: np.ndarray,
) -> np.ndarray:
    time = np.asarray(horizons, dtype=np.float64)
    if time.ndim != 1 or not np.all(np.isfinite(time)) or np.any(time < 0.0):
        raise ValueError("horizons must be a finite nonnegative vector")
    position = _nonnegative_vec3(position_radius, "position_radius")
    velocity = _nonnegative_vec3(velocity_radius, "velocity_radius")
    acceleration = _nonnegative_vec3(acceleration_radius, "acceleration_radius")
    jerk = _nonnegative_vec3(jerk_residual_bound, "jerk_residual_bound")
    return (
        position[None, :]
        + time[:, None] * velocity[None, :]
        + 0.5 * time[:, None] ** 2 * acceleration[None, :]
        + (time[:, None] ** 3 / 6.0) * jerk[None, :]
    )


def position_box_contains(
    truth: np.ndarray,
    center: np.ndarray,
    radius: np.ndarray | float,
    *,
    tolerance: float = 1e-10,
) -> bool:
    actual = np.asarray(truth, dtype=np.float64)
    nominal = np.asarray(center, dtype=np.float64)
    if actual.shape != (3,) or nominal.shape != (3,):
        raise ValueError("truth and center must have shape (3,)")
    return bool(
        np.all(np.abs(actual - nominal) <= _nonnegative_vec3(radius, "radius") + tolerance)
    )
