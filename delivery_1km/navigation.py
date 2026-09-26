"""Conservative route fallback for the 1 km finite-obstacle delivery maps.

The frozen v2 tracker and CBF/QP controller are reused unchanged.  This
planner checks every proposed leg against the full inflated map.  When the
direct leg is blocked, it climbs above the tallest obstacle, crosses at that
altitude, and descends.  It is a feasibility fallback, not a shortest-route
or energy-optimal planner.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np

from nav3d.controller import CBFConfig
from nav3d.geometry import BoxObstacle, World3D, vec3
from nav3d.planner import NoRouteError, Route
from nav3d.simulation import FlightResult, simulate_flight
from nav3d_v2.controller import PLANNING_CLEARANCE_MARGIN, WAYPOINT_RADIUS, SegmentTrackingController


class DeliveryRoutePlanner:
    def __init__(self, world: World3D, *, body_radius: float,
                 clearance_margin: float = PLANNING_CLEARANCE_MARGIN) -> None:
        self.world = world
        self.inflation = body_radius + clearance_margin
        if self.inflation <= 0:
            raise ValueError("route inflation must be positive")

    def plan(self, start: object, goal: object) -> Route:
        begun = perf_counter()
        origin, target = vec3(start, "start"), vec3(goal, "goal")
        if self.world.clearance(origin, self.inflation) <= 0 or self.world.clearance(target, self.inflation) <= 0:
            raise NoRouteError("start or goal violates map clearance")
        if np.linalg.norm(origin - target) < 1e-10:
            return Route((origin,), 0.0, 0.0, perf_counter() - begun, 1, 0)
        if self.world.segment_clear(origin, target, self.inflation):
            length = float(np.linalg.norm(target - origin))
            return Route((origin, target), length, length, perf_counter() - begun, 2, 1)

        highest = max((float(obstacle.high[2]) if isinstance(obstacle, BoxObstacle)
                       else float(obstacle.z_high) for obstacle in self.world.obstacles), default=0.0)
        cruise_z = max(float(origin[2]), float(target[2]), highest + self.inflation + 2.0)
        if cruise_z >= self.world.size[2] - self.inflation:
            raise NoRouteError("no overhead corridor inside the map")
        climb = np.array([origin[0], origin[1], cruise_z])
        descent = np.array([target[0], target[1], cruise_z])
        points = [origin]
        for point in (climb, descent, target):
            if np.linalg.norm(point - points[-1]) > 1e-10:
                points.append(point)
        checked = 1
        for a, b in zip(points, points[1:]):
            checked += 1
            if not self.world.segment_clear(a, b, self.inflation):
                raise NoRouteError("overflight leg violates inflated map clearance")
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
        return Route(tuple(points), length, length, perf_counter() - begun, len(points), checked)


def fly_delivery_route(world: World3D, start: object, goal: object, *,
                       config: CBFConfig, max_steps: int, goal_radius: float) -> FlightResult:
    return simulate_flight(
        world, start, goal, config=config, max_steps=max_steps,
        goal_radius=goal_radius, waypoint_radius=WAYPOINT_RADIUS,
        planner=DeliveryRoutePlanner(world, body_radius=config.body_radius),
        controller=SegmentTrackingController(world, config),
    )
