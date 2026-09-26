"""Short public-interface smoke for the continuing 2D task state machine."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibration import DEFAULT_OUTPUT
from .environment import DualConstraintEnv, EnvironmentAction


def _follow(env: DualConstraintEnv, destination_xy: tuple[float, float], max_decisions: int = 160, *, allow_partial: bool = False):
    route = env.router.plan(env.state.position[:2], destination_xy)
    waypoint_index = min(1, len(route.waypoints_xy) - 1)
    events: list[str] = []
    for _ in range(max_decisions):
        while waypoint_index < len(route.waypoints_xy) - 1 and np.linalg.norm(
            env.state.position[:2] - route.waypoints_xy[waypoint_index]
        ) <= 0.8:
            waypoint_index += 1
        delta = route.waypoints_xy[waypoint_index] - env.state.position[:2]
        distance = float(np.linalg.norm(delta))
        desired = delta / distance * min(5.0, 0.8 * distance) if distance > 1e-10 else np.zeros(2)
        _, _, done, info = env.step(EnvironmentAction(tuple(float(x) for x in desired)))
        events.append(info["event"])
        if done or info["event"] in ("target_completed", "docked", "return_takeover", "safety_failure"):
            return events
    if allow_partial:
        return events
    raise RuntimeError("route follower exceeded smoke decision limit")


def run_smoke(calibration_path: Path = DEFAULT_OUTPUT / "calibration.json") -> dict:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    env = DualConstraintEnv(
        0, calibration["capacity_synthetic_energy"], calibration["full_charge_seconds"],
    )
    first_target = env.target.position_xy
    service_events = _follow(env, first_target)
    if service_events[-1] != "target_completed" or env.completed_targets != 1:
        raise RuntimeError(f"first target was not completed: {service_events[-1]}")
    second_target = env.target.position_xy
    if second_target == first_target:
        raise RuntimeError("goal stream did not advance after completion")
    return_events = _follow(env, env.case.station_xy)
    if return_events[-1] == "return_takeover":
        while env.mode == "return" and not env.done:
            _, _, _, info = env.step()
            return_events.append(info["event"])
    if env.mode != "docked":
        raise RuntimeError(f"did not safely dock after service: {return_events[-1]}")
    before_charge = env.energy
    time_before_charge = env.time_s
    _, _, _, info = env.step(EnvironmentAction(charge_target_fraction=0.95))
    if info["event"] != "charged" or env.energy <= before_charge or env.target.position_xy != second_target:
        raise RuntimeError("partial charging changed the target or failed to add energy")
    departure_events = _follow(env, second_target, max_decisions=2, allow_partial=True)
    if env.mode != "flight" or env.done:
        raise RuntimeError(f"could not resume service after charge: {departure_events[-1]}")
    return {
        "completed_targets": env.completed_targets,
        "first_target": first_target,
        "second_target": second_target,
        "service_decisions": len(service_events),
        "return_decisions": len(return_events),
        "post_charge_decisions": len(departure_events),
        "energy_before_charge": before_charge,
        "energy_after_charge_and_departure": env.energy,
        "charge_seconds": env.time_s - time_before_charge - len(departure_events) * env.case.config.policy_dt_s,
        "simulated_seconds": env.time_s,
        "collision_count": env.state.collision_count,
        "mode": env.mode,
        "backup_feasible": bool(env.backup and env.backup.feasible and env.backup.energy_needed <= env.energy),
        "return_takeovers": env.return_takeovers,
        "charge_events": env.charge_events,
    }


if __name__ == "__main__":
    print(json.dumps(run_smoke(), sort_keys=True))
