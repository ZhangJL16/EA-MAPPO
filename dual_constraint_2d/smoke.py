"""Short integration smoke: serve, verified return, partial charge, serve.

This hand-coded route follower is a wiring check, not a trained policy, strong
planning baseline, or formal evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nav3d.simulation import FlightState

from .calibration import DEFAULT_OUTPUT
from .routes import PlanarRouter
from .shield import certify_policy_block, continue_verified_return, execute_certified_block
from .synthetic import SyntheticChargeState, SyntheticCharger, make_synthetic_map
from .targets import select_next_target


def _move_to_goal(case, router, state, energy, backup, goal_xy, max_blocks=150):
    route = router.plan(state.position[:2], goal_xy)
    waypoint_index = min(1, len(route.waypoints_xy) - 1)
    blocks = 0
    flight_seconds = 0.0
    for _ in range(max_blocks):
        while waypoint_index < len(route.waypoints_xy) - 1 and np.linalg.norm(
            state.position[:2] - route.waypoints_xy[waypoint_index]
        ) <= 0.8:
            waypoint_index += 1
        delta = route.waypoints_xy[waypoint_index] - state.position[:2]
        distance = float(np.linalg.norm(delta))
        desired = delta / distance * min(5.0, 0.8 * distance) if distance > 1e-10 else np.zeros(2)
        block = certify_policy_block(case, state, energy, desired, router=router, current_backup=backup)
        if not block.accepted_policy:
            raise RuntimeError(f"smoke route follower was filtered before target: {block.reason}")
        state, energy = execute_certified_block(case, state, energy, block)
        backup = block.endpoint_backup
        blocks += 1
        flight_seconds += len(block.actions) * case.config.physics_dt_s
        if np.linalg.norm(state.position[:2] - goal_xy) <= 1.5 and np.linalg.norm(state.velocity[:2]) <= 0.1:
            return state, energy, backup, flight_seconds, blocks
    raise RuntimeError("smoke target was not reached within block limit")


def run_smoke(calibration_path: Path = DEFAULT_OUTPUT / "calibration.json") -> dict:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    capacity = float(calibration["capacity_synthetic_energy"])
    charger = SyntheticCharger(capacity, float(calibration["full_charge_seconds"]))
    case = make_synthetic_map(0)
    router = PlanarRouter(case)
    first = select_next_target(case, router, capacity)
    state = FlightState(case.xyz(case.station_xy), np.zeros(3))
    energy = capacity
    backup = None
    state, energy, backup, first_seconds, first_blocks = _move_to_goal(
        case, router, state, energy, backup, first.position_xy
    )
    second = select_next_target(
        case, router, capacity, next_candidate_id=first.next_candidate_id
    )
    return_seconds = 0.0
    return_blocks = 0
    while backup.steps:
        block = continue_verified_return(backup, hold_steps=case.config.policy_hold_steps)
        state, energy = execute_certified_block(case, state, energy, block)
        backup = block.endpoint_backup
        return_seconds += len(block.actions) * case.config.physics_dt_s
        return_blocks += 1
    if np.linalg.norm(state.position[:2] - case.station_xy) > 1.5 or np.linalg.norm(state.velocity[:2]) > 0.1:
        raise RuntimeError("verified return did not finish at station")
    before_charge = energy
    charged = charger.charge_to_fraction(
        SyntheticChargeState(energy, first_seconds + return_seconds), 0.95
    )
    charge_seconds = charged.elapsed_s - first_seconds - return_seconds
    state, energy, backup, second_seconds, second_blocks = _move_to_goal(
        case, router, state, charged.energy, backup, second.position_xy
    )
    return {
        "completed_targets": 2,
        "first_target_candidate_id": first.accepted_candidate_id,
        "second_target_candidate_id": second.accepted_candidate_id,
        "first_service_seconds": first_seconds,
        "return_seconds": return_seconds,
        "charge_seconds": charge_seconds,
        "second_service_seconds": second_seconds,
        "first_service_blocks": first_blocks,
        "return_blocks": return_blocks,
        "second_service_blocks": second_blocks,
        "energy_before_charge": before_charge,
        "energy_after_partial_charge": charged.energy,
        "energy_after_second_service": energy,
        "collision_count": state.collision_count,
        "simulated_seconds": first_seconds + return_seconds + charge_seconds + second_seconds,
        "second_service_still_has_return": bool(backup.feasible and backup.energy_needed <= energy + 1e-8),
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), sort_keys=True))
