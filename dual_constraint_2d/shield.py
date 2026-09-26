"""Finite-block collision/return-energy filter for the synthetic 2D model.

One policy command lasts ten 0.05 s steps.  Before executing any of them, the
filter validates the entire block plus a return rollout from its endpoint.
At each intermediate state the remaining block and return rollout provide an
explicit continuation.  This is simulator-model evidence, not a formal proof
or a guarantee under disturbances/model error.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from nav3d.controller import CBFConfig, CBFController
from nav3d.simulation import FlightState, step_flight

from .backup import ReturnRollout, return_rollout
from .routes import PlanarRouter
from .synthetic import SyntheticEnergy, SyntheticMap


class PlanarVelocityController(CBFController):
    """Treat the waypoint argument as a desired planar velocity vector."""

    def nominal_acceleration(self, position: object, velocity: object, waypoint: object) -> np.ndarray:
        v = np.asarray(velocity, dtype=float)
        target = np.asarray(waypoint, dtype=float)
        desired = np.array((target[0], target[1], 0.0), dtype=float)
        nominal = self.config.velocity_gain * (desired - v)
        norm = float(np.linalg.norm(nominal[:2]))
        if norm > self.config.max_horizontal_acceleration:
            nominal[:2] *= self.config.max_horizontal_acceleration / norm
        nominal[2] = 0.0
        return nominal


def _controller(case: SyntheticMap) -> PlanarVelocityController:
    cfg = case.config
    return PlanarVelocityController(case.world, CBFConfig(
        dt=cfg.physics_dt_s,
        body_radius=cfg.body_radius_m,
        clearance_margin=cfg.planning_margin_m,
        max_horizontal_speed=cfg.max_speed_mps,
        max_vertical_speed=1.0,
        max_horizontal_acceleration=cfg.max_acceleration_mps2,
        max_vertical_acceleration=1.0,
        activation_distance=8.0,
    ))


@dataclass(frozen=True)
class CertifiedBlock:
    accepted_policy: bool
    reason: str
    actions: tuple[np.ndarray, ...]
    positions_xy: tuple[tuple[float, float], ...]
    velocities_xy: tuple[tuple[float, float], ...]
    cumulative_energy: tuple[float, ...]
    endpoint_backup: ReturnRollout | None
    collision_intervention_steps: int
    energy_filter_triggered: bool


def _slice_return(rollout: ReturnRollout, steps: int) -> ReturnRollout:
    if steps > rollout.steps:
        raise ValueError("cannot slice past return rollout")
    offset = rollout.cumulative_energy[steps]
    return ReturnRollout(
        True, "return_suffix", rollout.energy_needed - offset,
        (rollout.steps - steps) * (rollout.seconds / rollout.steps) if rollout.steps else 0.0,
        rollout.steps - steps, rollout.actions[steps:],
        rollout.positions_xy[steps:], rollout.velocities_xy[steps:],
        tuple(x - offset for x in rollout.cumulative_energy[steps:]),
    )


def _return_takeover(rollout: ReturnRollout, hold_steps: int, reason: str) -> CertifiedBlock:
    steps = min(hold_steps, rollout.steps)
    return CertifiedBlock(
        False, reason, rollout.actions[:steps], rollout.positions_xy[:steps + 1],
        rollout.velocities_xy[:steps + 1], rollout.cumulative_energy[:steps + 1],
        _slice_return(rollout, steps), 0, reason == "energy_filter",
    )


def continue_verified_return(rollout: ReturnRollout, *, hold_steps: int = 10) -> CertifiedBlock:
    """Advance a previously verified return without reopening policy control."""
    if not rollout.feasible or hold_steps <= 0:
        raise ValueError("continuing return requires a feasible rollout and positive hold")
    return _return_takeover(rollout, hold_steps, "continuing_return")


def certify_policy_block(
    case: SyntheticMap,
    state: FlightState,
    energy_available: float,
    desired_velocity_xy: object,
    *,
    router: PlanarRouter,
    current_backup: ReturnRollout | None = None,
    energy_model: SyntheticEnergy | None = None,
    hold_steps: int | None = None,
) -> CertifiedBlock:
    if not isfinite(energy_available) or energy_available < 0:
        raise ValueError("available energy must be finite and nonnegative")
    velocity = np.asarray(desired_velocity_xy, dtype=float)
    if velocity.shape != (2,) or not np.isfinite(velocity).all():
        raise ValueError("desired velocity must be a finite 2-vector")
    if np.linalg.norm(velocity) > case.config.max_speed_mps + 1e-9:
        raise ValueError("desired velocity exceeds the model speed limit")
    count = case.config.policy_hold_steps if hold_steps is None else hold_steps
    if not isinstance(count, int) or count <= 0:
        raise ValueError("hold_steps must be positive")
    power = energy_model or SyntheticEnergy()
    backup = current_backup or return_rollout(
        case, state, energy_available, router=router, energy_model=power
    )
    if not backup.feasible:
        return CertifiedBlock(False, "no_verified_start_backup", (), (), (), (), None, 0, False)
    if not np.allclose(backup.positions_xy[0], state.position[:2], atol=1e-7) or not np.allclose(
        backup.velocities_xy[0], state.velocity[:2], atol=1e-7
    ) or backup.energy_needed > energy_available + 1e-9:
        raise ValueError("cached backup does not match the current state or energy")
    controller = _controller(case)
    positions = [tuple(float(x) for x in state.position[:2])]
    velocities = [tuple(float(x) for x in state.velocity[:2])]
    cumulative = [0.0]
    actions: list[np.ndarray] = []
    interventions = 0
    desired = np.array((velocity[0], velocity[1], 0.0))
    current = state
    for _ in range(count):
        control = controller.action(current.position, current.velocity, desired)
        if not control.feasible or abs(control.acceleration[2]) > 1e-8:
            return _return_takeover(backup, count, "collision_filter")
        interventions += int(control.intervened)
        cost = power.flight_cost(current.velocity[:2], control.acceleration[:2], case.config.physics_dt_s)
        if cumulative[-1] + cost > energy_available + 1e-10:
            return _return_takeover(backup, count, "energy_filter")
        next_state, event = step_flight(case.world, current, control.acceleration, controller.config)
        if event.unified_contact:
            return _return_takeover(backup, count, "collision_filter")
        cumulative.append(cumulative[-1] + cost)
        actions.append(control.acceleration.copy())
        positions.append(tuple(float(x) for x in next_state.position[:2]))
        velocities.append(tuple(float(x) for x in next_state.velocity[:2]))
        current = next_state
    endpoint_backup = return_rollout(
        case, current, energy_available - cumulative[-1], router=router, energy_model=power
    )
    if not endpoint_backup.feasible:
        return _return_takeover(backup, count, "energy_filter" if endpoint_backup.reason == "insufficient_energy" else "return_filter")
    return CertifiedBlock(
        True, "policy_block_certified", tuple(actions), tuple(positions),
        tuple(velocities), tuple(cumulative), endpoint_backup, interventions, False,
    )


def execute_certified_block(
    case: SyntheticMap,
    state: FlightState,
    energy_available: float,
    block: CertifiedBlock,
) -> tuple[FlightState, float]:
    """Execute a prevalidated block; any trace mismatch is an explicit failure."""
    if not block.actions or len(block.positions_xy) != len(block.actions) + 1:
        raise ValueError("block has no executable actions or malformed trace")
    if block.endpoint_backup is None:
        raise ValueError("block has no endpoint return continuation")
    cfg = _controller(case).config
    if not np.allclose(state.position[:2], block.positions_xy[0], atol=1e-7) or not np.allclose(
        state.velocity[:2], block.velocities_xy[0], atol=1e-7
    ):
        raise RuntimeError("certified block does not start at current flight state")
    current = state
    for i, action in enumerate(block.actions, start=1):
        current, event = step_flight(case.world, current, action, cfg)
        if event.unified_contact or not np.allclose(current.position[:2], block.positions_xy[i], atol=1e-7) or not np.allclose(
            current.velocity[:2], block.velocities_xy[i], atol=1e-7
        ):
            raise RuntimeError("executed state diverged from the validated block")
        if energy_available - block.cumulative_energy[i] < -1e-9:
            raise RuntimeError("executed block depleted available energy")
        suffix_energy = (
            block.cumulative_energy[-1] - block.cumulative_energy[i]
            + block.endpoint_backup.energy_needed
        )
        if energy_available - block.cumulative_energy[i] + 1e-8 < suffix_energy:
            raise RuntimeError("executed step lost its verified return continuation")
    remaining = energy_available - block.cumulative_energy[-1]
    if block.endpoint_backup.energy_needed > remaining + 1e-8:
        raise RuntimeError("executed block lost its return certificate")
    return current, remaining
