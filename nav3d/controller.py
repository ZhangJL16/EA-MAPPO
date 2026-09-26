"""Waypoint tracking plus three-dimensional second-order CBF/QP projection.

The CBF is a local sufficient condition for the translational double integrator.
The returned action is additionally checked over its finite zero-order hold; an
infeasible or unverified action is reported rather than silently called safe.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin, sqrt
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse

from .geometry import World3D, vec3


@dataclass(frozen=True)
class CBFConfig:
    dt: float = 0.05
    body_radius: float = 0.5
    clearance_margin: float = 0.25
    max_horizontal_speed: float = 20.0
    max_vertical_speed: float = 5.0
    max_horizontal_acceleration: float = 5.0
    max_vertical_acceleration: float = 3.0
    k1: float = 2.0
    k2: float = 2.0
    velocity_gain: float = 2.0
    activation_distance: float = 50.0
    facets: int = 16

    def __post_init__(self) -> None:
        positive = (self.dt, self.max_horizontal_speed, self.max_vertical_speed, self.max_horizontal_acceleration, self.max_vertical_acceleration, self.k1, self.k2, self.velocity_gain, self.activation_distance)
        if not all(np.isfinite(x) and x > 0 for x in positive) or not np.isfinite(self.body_radius) or not np.isfinite(self.clearance_margin) or self.body_radius < 0 or self.clearance_margin < 0 or self.facets < 8:
            raise ValueError("invalid CBF configuration")

    @property
    def inflation(self) -> float:
        return self.body_radius + self.clearance_margin


@dataclass(frozen=True)
class ControlResult:
    acceleration: np.ndarray
    feasible: bool
    intervened: bool
    reason: str
    qp_seconds: float
    min_hold_clearance: float
    active_barriers: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "acceleration", vec3(self.acceleration, "acceleration"))


class CBFController:
    def __init__(self, world: World3D, config: CBFConfig | None = None) -> None:
        self.world = world
        self.config = CBFConfig() if config is None else config

    def nominal_acceleration(self, position: object, velocity: object, waypoint: object) -> np.ndarray:
        p, v, target = vec3(position, "position"), vec3(velocity, "velocity"), vec3(waypoint, "waypoint")
        delta = target - p
        horizontal_distance = float(np.linalg.norm(delta[:2]))
        desired = np.zeros(3)
        if horizontal_distance > 1e-10:
            desired[:2] = delta[:2] / horizontal_distance * min(
                self.config.max_horizontal_speed,
                sqrt(2.0 * self.config.max_horizontal_acceleration * horizontal_distance),
            )
        if abs(delta[2]) > 1e-10:
            desired[2] = np.sign(delta[2]) * min(
                self.config.max_vertical_speed,
                sqrt(2.0 * self.config.max_vertical_acceleration * abs(delta[2])),
            )
        nominal = self.config.velocity_gain * (desired - v)
        horizontal_norm = float(np.linalg.norm(nominal[:2]))
        if horizontal_norm > self.config.max_horizontal_acceleration:
            nominal[:2] *= self.config.max_horizontal_acceleration / horizontal_norm
        nominal[2] = np.clip(nominal[2], -self.config.max_vertical_acceleration, self.config.max_vertical_acceleration)
        return nominal

    def _hold_clearance(self, p: np.ndarray, v: np.ndarray, u: np.ndarray) -> float:
        previous = p
        minimum = self.world.clearance(p, self.config.inflation)
        for fraction in (0.25, 0.5, 0.75, 1.0):
            t = fraction * self.config.dt
            current = p + t * v + 0.5 * t * t * u
            minimum = min(minimum, self.world.clearance(current, self.config.inflation))
            if not self.world.segment_clear(previous, current, self.config.inflation):
                return min(minimum, -1e-6)
            previous = current
        return float(minimum)

    def _qp_rows(self, p: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        facets = self.config.facets
        apothem = cos(pi / facets)
        for index in range(facets):
            theta = 2.0 * pi * index / facets
            direction = np.array([cos(theta), sin(theta), 0.0])
            rows.append(direction)
            lower.append(-np.inf)
            upper.append(self.config.max_horizontal_acceleration * apothem)
            rows.append(direction)
            lower.append(-np.inf)
            upper.append((self.config.max_horizontal_speed * apothem - direction @ v) / self.config.dt)
        vertical = np.array([0.0, 0.0, 1.0])
        rows.extend((vertical, vertical))
        lower.extend((-self.config.max_vertical_acceleration, (-self.config.max_vertical_speed - v[2]) / self.config.dt))
        upper.extend((self.config.max_vertical_acceleration, (self.config.max_vertical_speed - v[2]) / self.config.dt))
        active = 0
        for barrier in self.world.barriers(p, self.config.inflation):
            if barrier.h > self.config.activation_distance:
                continue
            active += 1
            normal = barrier.normal
            h_dot = float(normal @ v)
            # Outside a convex obstacle, signed distance has nonnegative
            # curvature. Dropping the curvature term is conservative locally.
            rows.append(normal)
            lower.append(-(self.config.k1 + self.config.k2) * h_dot - self.config.k1 * self.config.k2 * barrier.h)
            upper.append(np.inf)
            if barrier.h + self.config.dt * h_dot < 0.25:
                rows.append(normal)
                lower.append(-2.0 * (barrier.h + self.config.dt * h_dot) / (self.config.dt**2))
                upper.append(np.inf)
        return np.stack(rows), np.asarray(lower), np.asarray(upper), active

    @staticmethod
    def _solve(nominal: np.ndarray, rows: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray | None, float, str]:
        begin = perf_counter()
        slacks_low = rows @ nominal - lower
        slacks_high = upper - rows @ nominal
        if np.min(slacks_low) >= -1e-7 and np.min(slacks_high) >= -1e-7:
            return nominal.copy(), perf_counter() - begin, "nominal_feasible"
        solver = osqp.OSQP()
        solver.setup(
            P=sparse.eye(3, format="csc"),
            q=-nominal,
            A=sparse.csc_matrix(rows),
            l=lower,
            u=upper,
            verbose=False,
            polishing=False,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=2000,
        )
        result = solver.solve(raise_error=False)
        elapsed = perf_counter() - begin
        if result.x is None or result.info.status not in ("solved", "solved inaccurate"):
            return None, elapsed, result.info.status
        solution = np.asarray(result.x, dtype=np.float64)
        if not np.all(np.isfinite(solution)) or np.any(rows @ solution < lower - 2e-5) or np.any(rows @ solution > upper + 2e-5):
            return None, elapsed, "constraint_violation"
        return solution, elapsed, result.info.status

    def action(self, position: object, velocity: object, waypoint: object) -> ControlResult:
        p, v = vec3(position, "position"), vec3(velocity, "velocity")
        nominal = self.nominal_acceleration(p, v, waypoint)
        initial_barrier_conditions = all(
            barrier.h >= -1e-8 and barrier.normal @ v + self.config.k1 * barrier.h >= -1e-8
            for barrier in self.world.barriers(p, self.config.inflation)
            if barrier.h <= self.config.activation_distance
        )
        rows, lower, upper, active = self._qp_rows(p, v)
        projected, elapsed, reason = self._solve(nominal, rows, lower, upper)
        if projected is not None:
            clearance = self._hold_clearance(p, v, projected)
            if clearance >= -1e-8:
                return ControlResult(
                    projected,
                    initial_barrier_conditions,
                    float(np.linalg.norm(projected - nominal)) > 1e-6,
                    reason if initial_barrier_conditions else "initial_barrier_condition_violated",
                    elapsed,
                    clearance,
                    active,
                )
            reason = "finite_hold_clearance_failed"
        # A failed QP does not silently authorize a nominal or colliding action.
        emergency = -v / self.config.dt
        horizontal_norm = float(np.linalg.norm(emergency[:2]))
        if horizontal_norm > self.config.max_horizontal_acceleration:
            emergency[:2] *= self.config.max_horizontal_acceleration / horizontal_norm
        emergency[2] = np.clip(emergency[2], -self.config.max_vertical_acceleration, self.config.max_vertical_acceleration)
        clearance = self._hold_clearance(p, v, emergency)
        return ControlResult(emergency, False, True, f"emergency_brake_after_{reason}", elapsed, clearance, active)
