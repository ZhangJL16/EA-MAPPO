"""Deterministic executable return rollout from the complete flight state.

Success is a simulation certificate under this fixed map, controller and
energy model only.  It is neither a continuous-time proof nor an uncertainty
robust bound.  A failed rollout is conservative and does not prove that no
other return path exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from nav3d.controller import CBFConfig
from nav3d.simulation import FlightState, step_flight

from .reference import PlanarTrackingController
from .routes import NoPlanarRoute, PlanarRouter
from .synthetic import SyntheticEnergy, SyntheticMap


@dataclass(frozen=True)
class ReturnRollout:
    feasible: bool
    reason: str
    energy_needed: float
    seconds: float
    steps: int
    actions: tuple[np.ndarray, ...]
    positions_xy: tuple[tuple[float, float], ...]
    velocities_xy: tuple[tuple[float, float], ...]
    cumulative_energy: tuple[float, ...]


def return_rollout(
    case: SyntheticMap,
    state: FlightState,
    energy_available: float,
    *,
    router: PlanarRouter | None = None,
    energy_model: SyntheticEnergy | None = None,
    max_steps: int = 3000,
    goal_radius_m: float = 1.5,
    arrival_speed_mps: float = 0.1,
) -> ReturnRollout:
    if not isfinite(energy_available) or energy_available < 0:
        raise ValueError("available energy must be finite and nonnegative")
    if max_steps <= 0 or goal_radius_m <= 0 or arrival_speed_mps <= 0:
        raise ValueError("invalid return rollout limits")
    config = case.config
    if abs(state.position[2] - config.flight_height_m) > 1e-8 or abs(state.velocity[2]) > 1e-8:
        raise ValueError("return state must be in the fixed flight plane")
    p = case.xyz(case.station_xy)
    actions: list[np.ndarray] = []
    positions = [tuple(float(x) for x in state.position[:2])]
    velocities = [tuple(float(x) for x in state.velocity[:2])]
    cumulative = [0.0]

    def result(feasible: bool, reason: str) -> ReturnRollout:
        return ReturnRollout(
            feasible, reason, cumulative[-1], len(actions) * config.physics_dt_s,
            len(actions), tuple(actions), tuple(positions), tuple(velocities), tuple(cumulative),
        )

    if np.linalg.norm(state.position - p) <= goal_radius_m and np.linalg.norm(state.velocity[:2]) <= arrival_speed_mps:
        return result(True, "already_at_station")
    route_planner = router or PlanarRouter(case)
    try:
        route = route_planner.plan(state.position[:2], case.station_xy)
    except NoPlanarRoute:
        return result(False, "no_planar_route")
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
    power = energy_model or SyntheticEnergy()
    waypoint_index = min(1, len(route.waypoints_xy) - 1)
    for _ in range(max_steps):
        while (
            waypoint_index < len(route.waypoints_xy) - 1
            and np.linalg.norm(state.position[:2] - route.waypoints_xy[waypoint_index]) <= 0.8
        ):
            waypoint_index += 1
        waypoint = case.xyz(route.waypoints_xy[waypoint_index])
        control = controller.action(state.position, state.velocity, waypoint)
        if not control.feasible or abs(control.acceleration[2]) > 1e-8:
            return result(False, "controller_infeasible")
        step_energy = power.flight_cost(state.velocity[:2], control.acceleration[:2], config.physics_dt_s)
        if cumulative[-1] + step_energy > energy_available + 1e-10:
            return result(False, "insufficient_energy")
        next_state, event = step_flight(case.world, state, control.acceleration, cbf)
        if event.unified_contact:
            return result(False, "contact_on_backup")
        actions.append(control.acceleration.copy())
        cumulative.append(cumulative[-1] + step_energy)
        positions.append(tuple(float(x) for x in next_state.position[:2]))
        velocities.append(tuple(float(x) for x in next_state.velocity[:2]))
        state = next_state
        if np.linalg.norm(state.position - p) <= goal_radius_m and np.linalg.norm(state.velocity[:2]) <= arrival_speed_mps:
            return result(True, "arrived")
    return result(False, "return_timeout")
