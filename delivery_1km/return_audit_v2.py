"""Vectorized accounting for the frozen executable return audit."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from nav3d.planner import NoRouteError

from .energy_profiles import ENERGY_PROFILES
from .energy_vectorized import account_flight_trace_vectorized
from .return_audit import (ROOT, M100_1KM, _hash,
                           build_manifest as scalar_build_manifest, make_map, simulate_return)


PROTOCOL = "m100_1km_retrospective_executable_return_v2_vectorized"
SOURCE_FILES = ("delivery_1km/return_audit.py", "delivery_1km/return_audit_v2.py",
                "delivery_1km/energy_vectorized.py", "delivery_1km/energy.py",
                "delivery_1km/energy_profiles.py", "delivery_1km/scenario.py",
                "delivery_1km/navigation.py", "nav3d/simulation.py",
                "nav3d_v2/controller.py")


def source_hash() -> str:
    digest = sha256()
    for relative in SOURCE_FILES:
        digest.update(relative.encode() + b"\0")
        digest.update((ROOT / relative).read_bytes() + b"\0")
    return digest.hexdigest()


def build_manifest(navigation: Path, job_id: str) -> dict:
    manifest = scalar_build_manifest(navigation, job_id)
    manifest["protocol"] = PROTOCOL
    manifest["source_sha256"] = source_hash()
    manifest["energy_accounting"] = "vectorized_v1; scalar_crosscheck_required"
    manifest["checkpoint_states"] = 100
    return manifest


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
            outbound = account_flight_trace_vectorized(velocities[:step_index + 1], accelerations[:step_index], payload, profile)
            returned = (account_flight_trace_vectorized(backup["velocities"], backup["accelerations"], payload, profile)
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
