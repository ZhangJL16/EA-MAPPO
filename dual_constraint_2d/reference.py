"""Short, deterministic planar reference flight for calibration and diagnostics.

This is not a return-safety certificate: a geometric route followed by a local
CBF controller can still time out or be conservatively rejected.  Calibration
must record those failures instead of treating geometric length as execution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nav3d.controller import CBFConfig, CBFController
from nav3d.simulation import FlightState, step_flight

from .routes import PlanarRouter
from .synthetic import SyntheticEnergy, SyntheticMap


class PlanarTrackingController(CBFController):
    """Low-speed waypoint arrival without a terminal velocity reset."""

    def nominal_acceleration(self, position: object, velocity: object, waypoint: object) -> np.ndarray:
        p = np.asarray(position, dtype=float)
        v = np.asarray(velocity, dtype=float)
        target = np.asarray(waypoint, dtype=float)
        delta = target[:2] - p[:2]
        distance = float(np.linalg.norm(delta))
        desired = np.zeros(3)
        if distance > 1e-12:
            desired[:2] = delta / distance * min(self.config.max_horizontal_speed, 0.8 * distance)
        nominal = self.config.velocity_gain * (desired - v)
        norm = float(np.linalg.norm(nominal[:2]))
        if norm > self.config.max_horizontal_acceleration:
            nominal[:2] *= self.config.max_horizontal_acceleration / norm
        nominal[2] = 0.0
        return nominal


@dataclass(frozen=True)
class ReferenceLeg:
    arrived: bool
    reason: str
    seconds: float
    energy: float
    steps: int
    route_length_m: float
    collision_count: int
    infeasible_steps: int
    final_position_xy: tuple[float, float]
    final_speed_mps: float


def simulate_reference_leg(
    case: SyntheticMap,
    start_xy: object,
    goal_xy: object,
    *,
    router: PlanarRouter | None = None,
    energy_model: SyntheticEnergy | None = None,
    max_steps: int = 3000,
    goal_radius_m: float = 1.5,
    arrival_speed_mps: float = 0.1,
    waypoint_radius_m: float = 0.8,
) -> ReferenceLeg:
    if max_steps <= 0 or goal_radius_m <= 0 or arrival_speed_mps <= 0 or waypoint_radius_m <= 0:
        raise ValueError("invalid reference flight limits")
    config = case.config
    route = (router or PlanarRouter(case)).plan(start_xy, goal_xy)
    power = energy_model or SyntheticEnergy()
    cbf = CBFConfig(
        dt=config.physics_dt_s,
        body_radius=config.body_radius_m,
        clearance_margin=config.planning_margin_m,
        max_horizontal_speed=config.max_speed_mps,
        max_vertical_speed=1.0,
        max_horizontal_acceleration=config.max_acceleration_mps2,
        max_vertical_acceleration=1.0,
        activation_distance=8.0,
    )
    controller = PlanarTrackingController(case.world, cbf)
    state = FlightState(case.xyz(start_xy), np.zeros(3))
    target = case.xyz(goal_xy)
    waypoint_index = min(1, len(route.waypoints_xy) - 1)
    energy = 0.0
    infeasible = 0
    if np.linalg.norm(state.position - target) <= goal_radius_m:
        return ReferenceLeg(True, "arrived", 0.0, 0.0, 0, route.length_m, 0, 0, tuple(state.position[:2]), 0.0)
    for step in range(1, max_steps + 1):
        while (
            waypoint_index < len(route.waypoints_xy) - 1
            and np.linalg.norm(state.position[:2] - route.waypoints_xy[waypoint_index]) <= waypoint_radius_m
        ):
            waypoint_index += 1
        waypoint = case.xyz(route.waypoints_xy[waypoint_index])
        control = controller.action(state.position, state.velocity, waypoint)
        if abs(control.acceleration[2]) > 1e-8:
            raise RuntimeError("planar reference controller commanded vertical acceleration")
        infeasible += int(not control.feasible)
        energy += power.flight_cost(state.velocity[:2], control.acceleration[:2], config.physics_dt_s)
        state, _event = step_flight(case.world, state, control.acceleration, cbf)
        if abs(state.position[2] - config.flight_height_m) > 1e-8 or abs(state.velocity[2]) > 1e-8:
            raise RuntimeError("planar reference left its fixed-height plane")
        speed = float(np.linalg.norm(state.velocity[:2]))
        if np.linalg.norm(state.position - target) <= goal_radius_m and speed <= arrival_speed_mps:
            return ReferenceLeg(
                True, "arrived", step * config.physics_dt_s, energy, step,
                route.length_m, state.collision_count, infeasible, tuple(state.position[:2]), speed,
            )
    return ReferenceLeg(
        False, "navigation_timeout", max_steps * config.physics_dt_s, energy,
        max_steps, route.length_m, state.collision_count, infeasible, tuple(state.position[:2]),
        float(np.linalg.norm(state.velocity[:2])),
    )
