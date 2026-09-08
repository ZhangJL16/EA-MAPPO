from __future__ import annotations

import math

import numpy as np


def correction_distance(
    nominal_acceleration: np.ndarray,
    executed_acceleration: np.ndarray,
    horizontal_limit: float,
    vertical_limit: float,
) -> float:
    """Physical correction / diameter of the bounded physical action set.

    A measured correction is defined even for emergency output, but is only a
    distance to a convex set when the actual QP output is a valid projection.
    """
    nominal = np.asarray(nominal_acceleration, dtype=np.float64)
    executed = np.asarray(executed_acceleration, dtype=np.float64)
    if nominal.shape != (3,) or executed.shape != (3,):
        raise ValueError("accelerations must have shape (3,)")
    if not np.isfinite(np.r_[nominal, executed, horizontal_limit, vertical_limit]).all():
        raise ValueError("nonfinite acceleration or limit")
    if horizontal_limit <= 0 or vertical_limit <= 0:
        raise ValueError("limits must be positive")
    for action in (nominal, executed):
        if np.linalg.norm(action[:2]) > horizontal_limit + 1e-5 or abs(action[2]) > vertical_limit + 1e-5:
            raise ValueError("acceleration outside declared physical limits")
    diameter = 2.0 * math.hypot(horizontal_limit, vertical_limit)
    return float(np.clip(np.linalg.norm(nominal - executed) / diameter, 0.0, 1.0))


def survival_weight(*, contact: bool, correction_integral: float, kappa: float) -> float:
    """Integral is sum_j d_j**2 * physics_dt, not square(mean(d_j))*dt."""
    if not np.isfinite([correction_integral, kappa]).all() or min(correction_integral, kappa) < 0:
        raise ValueError("hazard inputs must be finite and nonnegative")
    return 0.0 if contact else math.exp(-kappa * correction_integral)


def shaped_transition(
    *, goal: bool, contact: bool, correction_integral: float,
    kappa: float, gamma: float, potential: float, next_potential: float,
) -> tuple[float, float]:
    """Return shaped reward and continuation coefficient (no entropy in Q)."""
    if not 0 < gamma < 1:
        raise ValueError("gamma must lie strictly between zero and one")
    if not np.isfinite([potential, next_potential]).all():
        raise ValueError("potentials must be finite")
    safe_goal = bool(goal and not contact)
    q = survival_weight(contact=contact, correction_integral=correction_integral, kappa=kappa)
    beta = gamma * q * (not safe_goal)
    reward = q * safe_goal + beta * next_potential - potential
    return float(reward), float(beta)
