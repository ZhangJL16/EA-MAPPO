"""Focused low-level simulation; contact repair preserves the same task."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .controller import CBFController, CBFConfig
from .geometry import World3D, vec3
from .planner import Route, VisibilityPlanner


@dataclass(frozen=True)
class FlightState:
    position: np.ndarray
    velocity: np.ndarray
    previous_contact: bool = False
    collision_count: int = 0
    safety_cost: int = 0
    raw_contact_penalty: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", vec3(self.position, "position"))
        object.__setattr__(self, "velocity", vec3(self.velocity, "velocity"))


@dataclass(frozen=True)
class StepEvent:
    boundary_contact: bool
    obstacle_contact: bool
    unified_contact: bool
    raw_penalty: float


def step_flight(
    world: World3D,
    state: FlightState,
    acceleration: object,
    config: CBFConfig,
) -> tuple[FlightState, StepEvent]:
    u = vec3(acceleration, "acceleration")
    dt = config.dt
    candidate_position = state.position + dt * state.velocity + 0.5 * dt * dt * u
    candidate_velocity = state.velocity + dt * u
    # A correct QP action respects speed limits. This numerical guard mirrors
    # the plant limit and keeps a forced/invalid test action bounded.
    horizontal_speed = float(np.linalg.norm(candidate_velocity[:2]))
    if horizontal_speed > config.max_horizontal_speed:
        candidate_velocity[:2] *= config.max_horizontal_speed / horizontal_speed
    candidate_velocity[2] = np.clip(candidate_velocity[2], -config.max_vertical_speed, config.max_vertical_speed)
    realized = (candidate_velocity - state.velocity) / dt
    candidate_position = state.position + dt * state.velocity + 0.5 * dt * dt * realized
    boundary, obstacle = world.segment_contacts(state.position, candidate_position, config.body_radius)
    contact = boundary or obstacle
    unit_penalty = -1.2 * (0.35 if state.previous_contact else 1.0)
    penalty = unit_penalty * (int(boundary) + int(obstacle)) if contact else 0.0
    next_state = FlightState(
        position=state.position if contact else candidate_position,
        velocity=np.zeros(3) if contact else candidate_velocity,
        previous_contact=contact,
        collision_count=state.collision_count + int(contact),
        safety_cost=state.safety_cost + int(contact),
        raw_contact_penalty=state.raw_contact_penalty + penalty,
    )
    return next_state, StepEvent(boundary, obstacle, contact, penalty)


@dataclass(frozen=True)
class FlightResult:
    arrived: bool
    reason: str
    elapsed_seconds: float
    steps: int
    route: Route
    final_state: FlightState
    controller_infeasible_steps: int
    intervention_steps: int
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    control_seconds: np.ndarray
    qp_seconds: np.ndarray


def simulate_flight(
    world: World3D,
    start: object,
    goal: object,
    *,
    config: CBFConfig | None = None,
    max_steps: int = 4000,
    goal_radius: float = 0.8,
    waypoint_radius: float = 0.8,
    planner: VisibilityPlanner | None = None,
    controller: CBFController | None = None,
) -> FlightResult:
    cfg = CBFConfig() if config is None else config
    if max_steps <= 0 or goal_radius <= 0 or waypoint_radius <= 0:
        raise ValueError("invalid flight limits")
    p, target = vec3(start, "start"), vec3(goal, "goal")
    route_planner = planner or VisibilityPlanner(world, body_radius=cfg.body_radius, clearance_margin=cfg.clearance_margin)
    control = controller or CBFController(world, cfg)
    route = route_planner.plan(p, target)
    state = FlightState(p, np.zeros(3))
    positions = [state.position.copy()]
    velocities = [state.velocity.copy()]
    accelerations: list[np.ndarray] = []
    control_seconds: list[float] = []
    qp_seconds: list[float] = []
    waypoint_index = min(1, len(route.waypoints) - 1)
    infeasible = 0
    interventions = 0
    if np.linalg.norm(p - target) <= goal_radius:
        return FlightResult(True, "arrived", 0.0, 0, route, state, 0, 0, np.stack(positions), np.stack(velocities), np.zeros((0, 3)), np.zeros(0), np.zeros(0))
    for step in range(1, max_steps + 1):
        while waypoint_index < len(route.waypoints) - 1 and np.linalg.norm(state.position - route.waypoints[waypoint_index]) <= waypoint_radius:
            waypoint_index += 1
        control_started = perf_counter()
        result = control.action(state.position, state.velocity, route.waypoints[waypoint_index])
        control_seconds.append(perf_counter() - control_started)
        qp_seconds.append(result.qp_seconds)
        infeasible += int(not result.feasible)
        interventions += int(result.intervened)
        state, _event = step_flight(world, state, result.acceleration, cfg)
        positions.append(state.position.copy())
        velocities.append(state.velocity.copy())
        accelerations.append(result.acceleration.copy())
        if np.linalg.norm(state.position - target) <= goal_radius:
            state = FlightState(
                position=state.position,
                velocity=np.zeros(3),
                previous_contact=state.previous_contact,
                collision_count=state.collision_count,
                safety_cost=state.safety_cost,
                raw_contact_penalty=state.raw_contact_penalty,
            )
            velocities[-1] = state.velocity.copy()
            return FlightResult(True, "arrived", step * cfg.dt, step, route, state, infeasible, interventions, np.stack(positions), np.stack(velocities), np.stack(accelerations), np.asarray(control_seconds), np.asarray(qp_seconds))
    return FlightResult(False, "navigation_timeout", max_steps * cfg.dt, max_steps, route, state, infeasible, interventions, np.stack(positions), np.stack(velocities), np.stack(accelerations), np.asarray(control_seconds), np.asarray(qp_seconds))
