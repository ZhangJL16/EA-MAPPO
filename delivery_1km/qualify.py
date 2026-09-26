"""Resumable qualification on unseen 1 km ground/roof routes."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from nav3d.planner import NoRouteError
from nav3d_v2.qualify import source_hash as frozen_navigation_hash

from .navigation import fly_delivery_route
from .scenario import M100_1KM, make_map
from .sites import sample_site


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "m100_1km_finite_cylinders_navigation_v1_overflight_fallback"
STRATA = ("station_to_ground", "station_to_roof", "ground_to_roof", "roof_to_station")


def source_hash() -> str:
    digest = sha256()
    digest.update(frozen_navigation_hash().encode())
    for path in (ROOT / "delivery_1km/scenario.py", ROOT / "delivery_1km/sites.py",
                 ROOT / "delivery_1km/navigation.py", ROOT / "delivery_1km/qualify.py"):
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest(map_count: int = 12, per_stratum: int = 2) -> dict:
    if map_count <= 0 or per_stratum <= 0:
        raise ValueError("map_count and per_stratum must be positive")
    jobs = []
    for map_id in range(100, 100 + map_count):
        for repeat in range(per_stratum):
            for stratum in STRATA:
                case, start, goal = _endpoints({"map_id": map_id, "repeat": repeat, "stratum": stratum})
                jobs.append({
                    "job_id": f"m{map_id:03d}_{stratum}_{repeat:02d}",
                    "map_id": map_id,
                    "stratum": stratum,
                    "repeat": repeat,
                    "station_m": case.station_m,
                    "start_m": start,
                    "goal_m": goal,
                })
    return {
        "protocol": PROTOCOL,
        "source_sha256": source_hash(),
        "frozen_navigation_sha256": frozen_navigation_hash(),
        "scenario_name": M100_1KM.name,
        "map_count": map_count,
        "per_stratum": per_stratum,
        "max_steps": M100_1KM.navigation_guard_steps,
        "goal_radius_m": M100_1KM.goal_radius_m,
        "planner": "direct_or_verified_overflight_v1",
        "checkpoint_jobs": 2,
        "jobs": jobs,
    }


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _quantiles(values: np.ndarray) -> dict:
    return {f"p{level}": float(np.quantile(values, level / 100)) if values.size else 0.0 for level in (50, 95, 99)}


def _endpoints(job: dict) -> tuple[object, object, object]:
    case = make_map(job["map_id"])
    repeat = job["repeat"]
    stratum = job["stratum"]
    ground = sample_site(case, 10_000 + 17 * repeat, "ground") if "ground" in stratum else None
    roof = sample_site(case, 20_000 + 17 * repeat, "roof") if "roof" in stratum else None
    if stratum == "station_to_ground":
        return case, case.station_m, ground.position_m
    if stratum == "station_to_roof":
        return case, case.station_m, roof.position_m
    if stratum == "ground_to_roof":
        return case, ground.position_m, roof.position_m
    if stratum == "roof_to_station":
        return case, roof.position_m, case.station_m
    raise ValueError("unknown stratum")


def execute_job(output: Path, manifest: dict, job: dict) -> dict:
    started = perf_counter()
    row = {"job_id": job["job_id"], "map_id": job["map_id"], "stratum": job["stratum"]}
    case = make_map(job["map_id"])
    start, goal = job["start_m"], job["goal_m"]
    if not np.allclose(case.station_m, job["station_m"]):
        raise RuntimeError("saved station does not match generated map")
    row.update({"station_m": case.station_m, "start_m": start, "goal_m": goal})
    try:
        flight = fly_delivery_route(
            case.world, start, goal, config=M100_1KM.navigation_config(),
            max_steps=manifest["max_steps"], goal_radius=manifest["goal_radius_m"],
        )
    except NoRouteError as error:
        row.update({"outcome": "no_route", "error": str(error), "wall_seconds": perf_counter() - started})
        return row
    trace_name = f"{job['job_id']}.npz"
    trace = output / "traces" / trace_name
    temporary = trace.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream, positions=flight.positions, velocities=flight.velocities,
            accelerations=flight.accelerations, control_seconds=flight.control_seconds,
            qp_seconds=flight.qp_seconds,
        )
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(trace)
    row.update({
        "outcome": "arrived" if flight.arrived else "navigation_timeout",
        "trace": f"traces/{trace_name}",
        "seconds": flight.elapsed_seconds,
        "steps": flight.steps,
        "collision_count": flight.final_state.collision_count,
        "controller_infeasible_steps": flight.controller_infeasible_steps,
        "intervention_steps": flight.intervention_steps,
        "route_length_m": flight.route.geometric_length,
        "actual_length_m": float(np.linalg.norm(np.diff(flight.positions, axis=0), axis=1).sum()),
        "route_planning_seconds": flight.route.planning_seconds,
        "control_seconds": _quantiles(flight.control_seconds),
        "qp_seconds": _quantiles(flight.qp_seconds),
        "wall_seconds": perf_counter() - started,
    })
    return row


def run(output: Path, *, resume: bool = False, stop_after_checkpoint: bool = False) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_hash() or manifest["frozen_navigation_sha256"] != frozen_navigation_hash():
            raise RuntimeError("qualification source differs from the manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        (output / "traces").mkdir()
        manifest = build_manifest()
        _atomic_json(output / "manifest.json", manifest)
    completed = 0
    for job in manifest["jobs"]:
        result_path = output / "results" / f"{job['job_id']}.json"
        if result_path.exists():
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved["job_id"] != job["job_id"] or ("trace" in saved and not (output / saved["trace"]).exists()):
                raise RuntimeError("saved result or trace does not match its job")
            completed += 1
            continue
        _atomic_json(result_path, execute_job(output, manifest, job))
        completed += 1
        if completed % manifest["checkpoint_jobs"] == 0 or completed == len(manifest["jobs"]):
            checkpoint = {
                "protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"],
                "completed": completed, "total": len(manifest["jobs"]),
                "last_job_id": job["job_id"],
            }
            _atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    return json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    arguments = parser.parse_args()
    run(arguments.output, resume=arguments.resume, stop_after_checkpoint=arguments.stop_after_checkpoint)


if __name__ == "__main__":
    main()
