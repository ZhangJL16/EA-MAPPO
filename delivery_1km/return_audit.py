"""Resumable, retrospective return-to-station audit of frozen flight states."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from nav3d.planner import NoRouteError
from nav3d.simulation import FlightState, step_flight
from nav3d_v2.controller import SegmentTrackingController, WAYPOINT_RADIUS

from .energy import account_flight_trace
from .energy_profiles import ENERGY_PROFILES
from .navigation import DeliveryRoutePlanner
from .scenario import M100_1KM, make_map


ROOT = Path(__file__).resolve().parents[1]
NAVIGATION = ROOT / "artifacts/delivery_1km_nav_arm_20260926"
PROTOCOL = "m100_1km_retrospective_executable_return_v1"
RETURN_GOAL_RADIUS_M = 0.8
RETURN_SPEED_TOLERANCE_MPS = 0.5
SOURCE_FILES = (
    "nav3d/geometry.py", "nav3d/controller.py", "nav3d/simulation.py",
    "nav3d_v2/controller.py", "delivery_1km/scenario.py",
    "delivery_1km/navigation.py", "delivery_1km/energy.py",
    "delivery_1km/energy_profiles.py", "delivery_1km/return_audit.py",
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update((ROOT / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def simulate_return(world, position: np.ndarray, velocity: np.ndarray, station: object):
    """Execute the frozen planner/controller/plant from a moving flight state."""
    cfg = M100_1KM.navigation_config()
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
    if (np.linalg.norm(state.position - target) <= RETURN_GOAL_RADIUS_M
            and np.linalg.norm(state.velocity) <= RETURN_SPEED_TOLERANCE_MPS):
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
        if (np.linalg.norm(state.position - target) <= RETURN_GOAL_RADIUS_M
                and np.linalg.norm(state.velocity) <= RETURN_SPEED_TOLERANCE_MPS):
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


def build_manifest(navigation: Path, job_id: str) -> dict:
    nav_manifest_path = navigation / "manifest.json"
    nav_manifest = json.loads(nav_manifest_path.read_text(encoding="utf-8"))
    nav_checkpoint = json.loads((navigation / "checkpoint.json").read_text(encoding="utf-8"))
    if nav_checkpoint["completed"] != len(nav_manifest["jobs"]):
        raise RuntimeError("navigation qualification is incomplete")
    job = next((item for item in nav_manifest["jobs"] if item["job_id"] == job_id), None)
    if job is None or job["stratum"] not in ("station_to_ground", "station_to_roof"):
        raise ValueError("choose an outbound station-to-site job")
    result_path = navigation / "results" / f"{job_id}.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["outcome"] != "arrived" or result["collision_count"]:
        raise RuntimeError("source flight must arrive without contact")
    trace_path = navigation / result["trace"]
    with np.load(trace_path, allow_pickle=False) as trace:
        positions, velocities, accelerations = (trace[key] for key in ("positions", "velocities", "accelerations"))
    if len(positions) != len(velocities) or len(positions) != len(accelerations) + 1:
        raise RuntimeError("invalid navigation trace lengths")
    case = make_map(job["map_id"])
    if not np.allclose(case.station_m, job["station_m"]) or not np.allclose(positions[0], job["start_m"]):
        raise RuntimeError("navigation map or starting state differs")
    distances = np.linalg.norm(positions[1:] - np.asarray(case.station_m), axis=1)
    moving = np.linalg.norm(velocities[1:], axis=1) > RETURN_SPEED_TOLERANCE_MPS
    candidates = [index for index, eligible in enumerate(
        (distances > M100_1KM.goal_radius_m) & moving, start=1,
    ) if eligible]
    if len(candidates) < 2:
        raise RuntimeError("source flight lacks two moving states outside station tolerance")
    first = candidates[:2]
    state_indices = first + [index for index in range(1, len(accelerations) + 1) if index not in first]
    return {
        "protocol": PROTOCOL, "source_sha256": source_hash(),
        "navigation_manifest_sha256": _hash(nav_manifest_path),
        "navigation_result_sha256": _hash(result_path),
        "navigation_trace_sha256": _hash(trace_path),
        "job": job, "trace": result["trace"],
        "initial_soc": 1.0, "payloads_kg": list(M100_1KM.cargo_masses_kg),
        "profiles": [asdict(profile) for profile in ENERGY_PROFILES],
        "states": len(accelerations), "state_indices": state_indices, "checkpoint_states": 2,
        "diagnostic_only": "executed state and return trace are retrospective labels, not preflight prediction",
    }


def audit_state(navigation: Path, output: Path, manifest: dict, step_index: int) -> dict:
    job = manifest["job"]
    trace_path = navigation / manifest["trace"]
    result_path = navigation / "results" / f"{job['job_id']}.json"
    if _hash(trace_path) != manifest["navigation_trace_sha256"] or _hash(result_path) != manifest["navigation_result_sha256"]:
        raise RuntimeError("frozen navigation input changed")
    with np.load(trace_path, allow_pickle=False) as trace:
        positions, velocities, accelerations = (trace[key] for key in ("positions", "velocities", "accelerations"))
    if not 1 <= step_index <= len(accelerations):
        raise ValueError("step index is outside the source flight")
    case = make_map(job["map_id"])
    started = perf_counter()
    try:
        backup = simulate_return(case.world, positions[step_index], velocities[step_index], case.station_m)
        outcome = "arrived" if backup["arrived"] else "navigation_timeout"
    except NoRouteError as error:
        backup, outcome = None, "no_route"
        route_error = str(error)
    samples = []
    for payload in manifest["payloads_kg"]:
        for profile in ENERGY_PROFILES:
            outbound = account_flight_trace(velocities[:step_index + 1], accelerations[:step_index], payload, profile)
            returned = (account_flight_trace(backup["velocities"], backup["accelerations"], payload, profile)
                        if backup is not None else None)
            capacity = M100_1KM.nominal_battery_wh * profile.usable_energy_fraction
            margin = capacity - outbound.energy_wh - returned.energy_wh if returned is not None else None
            valid = (backup is not None and backup["arrived"] and backup["collisions"] == 0
                     and backup["controller_infeasible_steps"] == 0 and outbound.infeasible_steps == 0
                     and returned.infeasible_steps == 0 and margin >= 0)
            samples.append({"payload_kg": payload, "profile": profile.name,
                            "outbound_wh": outbound.energy_wh,
                            "return_wh": returned.energy_wh if returned is not None else None,
                            "usable_capacity_wh": capacity, "margin_wh": margin,
                            "outbound_capability_violations": outbound.infeasible_steps,
                            "return_capability_violations": returned.infeasible_steps if returned is not None else None,
                            "return_evidence_valid": valid})
    row = {"job_id": job["job_id"], "map_id": job["map_id"], "step_index": step_index,
           "position_m": positions[step_index].tolist(), "velocity_mps": velocities[step_index].tolist(),
           "elapsed_seconds": step_index * M100_1KM.physics_dt_s,
           "outcome": outcome, "samples": samples, "query_wall_seconds": perf_counter() - started}
    if backup is not None:
        trace_name = f"step_{step_index:05d}.npz"
        saved_trace = output / "traces" / trace_name
        temporary = saved_trace.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, positions=backup["positions"],
                                velocities=backup["velocities"], accelerations=backup["accelerations"])
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(saved_trace)
        row.update({"return_steps": backup["steps"], "return_collisions": backup["collisions"],
                    "return_controller_infeasible_steps": backup["controller_infeasible_steps"],
                    "return_interventions": backup["interventions"],
                    "return_route_length_m": backup["route_length_m"],
                    "return_terminal_position_m": backup["positions"][-1].tolist(),
                    "return_trace": f"traces/{trace_name}", "return_trace_sha256": _hash(saved_trace)})
    else:
        row["error"] = route_error
    return row


def run(output: Path, *, navigation: Path = NAVIGATION,
        job_id: str = "m100_station_to_ground_00", resume: bool = False,
        stop_after_checkpoint: bool = False) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_hash():
            raise RuntimeError("return audit source changed")
        if manifest["navigation_manifest_sha256"] != _hash(navigation / "manifest.json"):
            raise RuntimeError("navigation manifest changed")
        if manifest["profiles"] != [asdict(profile) for profile in ENERGY_PROFILES]:
            raise RuntimeError("energy profiles changed")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        (output / "traces").mkdir()
        manifest = build_manifest(navigation, job_id)
        _atomic_json(output / "manifest.json", manifest)
    if (len(manifest["state_indices"]) != manifest["states"]
            or set(manifest["state_indices"]) != set(range(1, manifest["states"] + 1))):
        raise RuntimeError("audit state schedule is incomplete")
    previous_completed = (json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))["completed"]
                          if resume and (output / "checkpoint.json").exists() else 0)
    for completed, index in enumerate(manifest["state_indices"], start=1):
        path = output / "results" / f"step_{index:05d}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["step_index"] != index or len(saved["samples"]) != len(manifest["profiles"]) * len(manifest["payloads_kg"]):
                raise RuntimeError("saved return audit result is invalid")
            if "return_trace" in saved and _hash(output / saved["return_trace"]) != saved["return_trace_sha256"]:
                raise RuntimeError("saved return trace changed")
        else:
            _atomic_json(path, audit_state(navigation, output, manifest, index))
        if completed > previous_completed and (completed % manifest["checkpoint_states"] == 0
                                             or completed == manifest["states"]):
            checkpoint = {"protocol": PROTOCOL, "source_sha256": manifest["source_sha256"],
                          "completed": completed, "total": manifest["states"],
                          "last_step_index": index}
            _atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    return json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--navigation", type=Path, default=NAVIGATION)
    parser.add_argument("--job-id", default="m100_station_to_ground_00")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    run(args.output, navigation=args.navigation, job_id=args.job_id,
        resume=args.resume, stop_after_checkpoint=args.stop_after_checkpoint)


if __name__ == "__main__":
    main()
