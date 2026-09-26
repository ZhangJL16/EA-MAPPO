"""Return audit with a frozen, backup-only CBF recovery setting."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from nav3d.simulation import FlightState, step_flight
from nav3d_v2.controller import SegmentTrackingController, WAYPOINT_RADIUS

from .navigation import DeliveryRoutePlanner
from .scenario import M100_1KM
from . import return_audit_v2


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m100_1km_retrospective_executable_return_v3_backup_cbf_1p5"
CBF_GAIN = 1.5
SOURCE_FILES = return_audit_v2.SOURCE_FILES + ("delivery_1km/return_audit_v3.py",)


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0")
        digest.update((ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def build_manifest(navigation: Path, job_id: str) -> dict:
    manifest = return_audit_v2.build_manifest(navigation, job_id)
    manifest["protocol"] = PROTOCOL
    manifest["source_sha256"] = source_hash()
    manifest["backup_cbf_gains"] = {"k1": CBF_GAIN, "k2": CBF_GAIN}
    manifest["parameter_selection_maps"] = [112, 113, 114]
    return manifest


def simulate_return(world, position: np.ndarray, velocity: np.ndarray, station: object):
    cfg = replace(M100_1KM.navigation_config(), k1=CBF_GAIN, k2=CBF_GAIN)
    target = np.asarray(station, dtype=float)
    planner = DeliveryRoutePlanner(world, body_radius=cfg.body_radius)
    route = planner.plan(position, target)
    controller = SegmentTrackingController(world, cfg)
    state = FlightState(position, velocity)
    positions = [state.position.copy()]
    velocities = [state.velocity.copy()]
    accelerations = []
    waypoint_index = min(1, len(route.waypoints) - 1)
    infeasible = 0
    interventions = 0
    if (np.linalg.norm(state.position - target) <= 0.8
            and np.linalg.norm(state.velocity) <= 0.5):
        return {"arrived": True, "steps": 0, "collisions": 0, "controller_infeasible_steps": 0,
                "interventions": 0, "route_length_m": route.geometric_length,
                "positions": np.stack(positions), "velocities": np.stack(velocities),
                "accelerations": np.zeros((0, 3))}
    for step in range(1, M100_1KM.navigation_guard_steps + 1):
        while waypoint_index < len(route.waypoints) - 1 and np.linalg.norm(state.position - route.waypoints[waypoint_index]) <= WAYPOINT_RADIUS:
            waypoint_index += 1
        action = controller.action(state.position, state.velocity, route.waypoints[waypoint_index])
        infeasible += int(not action.feasible)
        interventions += int(action.intervened)
        state, _ = step_flight(world, state, action.acceleration, cfg)
        positions.append(state.position.copy())
        accelerations.append(action.acceleration.copy())
        velocities.append(state.velocity.copy())
        if (np.linalg.norm(state.position - target) <= 0.8
                and np.linalg.norm(state.velocity) <= 0.5):
            velocities[-1] = np.zeros(3)
            return {"arrived": True, "steps": step, "collisions": state.collision_count,
                    "controller_infeasible_steps": infeasible, "interventions": interventions,
                    "route_length_m": route.geometric_length,
                    "positions": np.stack(positions), "velocities": np.stack(velocities),
                    "accelerations": np.stack(accelerations)}
    return {"arrived": False, "steps": M100_1KM.navigation_guard_steps,
            "collisions": state.collision_count, "controller_infeasible_steps": infeasible,
            "interventions": interventions, "route_length_m": route.geometric_length,
            "positions": np.stack(positions), "velocities": np.stack(velocities),
            "accelerations": np.stack(accelerations)}


def audit_state(navigation: Path, output: Path, manifest: dict, step_index: int) -> dict:
    # Each process pool worker runs one audit at a time; restore the module
    # binding so this wrapper cannot affect another engine used in that worker.
    original = return_audit_v2.simulate_return
    try:
        return_audit_v2.simulate_return = simulate_return
        return return_audit_v2.audit_state(navigation, output, manifest, step_index)
    finally:
        return_audit_v2.simulate_return = original
