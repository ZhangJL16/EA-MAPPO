"""Track the planned 3D segments; retain the v1 CBF/QP safety filter."""

from __future__ import annotations

from math import sqrt

import numpy as np

from nav3d.controller import CBFConfig, CBFController
from nav3d.geometry import World3D, vec3
from nav3d.planner import VisibilityPlanner
from nav3d.simulation import FlightResult, simulate_flight


PLANNING_CLEARANCE_MARGIN = 1.0
WAYPOINT_RADIUS = 0.4


class SegmentTrackingController(CBFController):
    """Feed the CBF a nominal acceleration tangent to each planned 3D leg.

    The reference is the entire segment rather than its endpoint. Starting on
    a straight segment with zero cross-track velocity produces acceleration
    along that segment, so altitude at an obstacle crossing follows the route.
    """

    def __init__(self, world: World3D, config: CBFConfig) -> None:
        super().__init__(world, config)
        self._segment_start: np.ndarray | None = None
        self._last_waypoint: np.ndarray | None = None

    def nominal_acceleration(self, position: object, velocity: object, waypoint: object) -> np.ndarray:
        p, v, target = vec3(position, "position"), vec3(velocity, "velocity"), vec3(waypoint, "waypoint")
        if self._last_waypoint is None:
            self._segment_start = p.copy()
        elif np.linalg.norm(target - self._last_waypoint) > 1e-7:
            self._segment_start = self._last_waypoint.copy()
        self._last_waypoint = target.copy()
        assert self._segment_start is not None
        delta = target - self._segment_start
        length = float(np.linalg.norm(delta))
        if length < 1e-9:
            return super().nominal_acceleration(p, v, target)
        tangent = delta / length
        horizontal_fraction = float(np.linalg.norm(tangent[:2]))
        vertical_fraction = abs(float(tangent[2]))
        speed_limit = min(
            self.config.max_horizontal_speed / max(horizontal_fraction, 1e-9),
            self.config.max_vertical_speed / max(vertical_fraction, 1e-9),
        )
        along_acceleration_limit = min(
            self.config.max_horizontal_acceleration / max(horizontal_fraction, 1e-9),
            self.config.max_vertical_acceleration / max(vertical_fraction, 1e-9),
        )
        raw_progress = float((p - self._segment_start) @ tangent)
        progress = float(np.clip(raw_progress, 0.0, length))
        reference = self._segment_start + progress * tangent
        cross_error = p - reference
        along_speed = float(v @ tangent)
        cross_speed = v - along_speed * tangent
        remaining = max(0.0, length - raw_progress)
        desired_along_speed = min(speed_limit, sqrt(2.0 * along_acceleration_limit * remaining))
        if raw_progress > length:
            desired_along_speed = -min(1.0, sqrt(2.0 * along_acceleration_limit * (raw_progress - length)))
        along_acceleration = np.clip(
            2.0 * (desired_along_speed - along_speed),
            -along_acceleration_limit,
            along_acceleration_limit,
        )
        nominal = along_acceleration * tangent - 3.0 * cross_error - 2.5 * cross_speed
        horizontal_norm = float(np.linalg.norm(nominal[:2]))
        if horizontal_norm > self.config.max_horizontal_acceleration:
            nominal[:2] *= self.config.max_horizontal_acceleration / horizontal_norm
        nominal[2] = np.clip(nominal[2], -self.config.max_vertical_acceleration, self.config.max_vertical_acceleration)
        return nominal


def fly_segment_route(world: World3D, start: object, goal: object, *,
                      config: CBFConfig, max_steps: int = 1200,
                      goal_radius: float = 0.8) -> FlightResult:
    planner = VisibilityPlanner(
        world, body_radius=config.body_radius,
        clearance_margin=PLANNING_CLEARANCE_MARGIN,
    )
    controller = SegmentTrackingController(world, config)
    return simulate_flight(
        world, start, goal, config=config, max_steps=max_steps,
        goal_radius=goal_radius, waypoint_radius=WAYPOINT_RADIUS,
        planner=planner, controller=controller,
    )
