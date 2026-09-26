"""Frozen-map, resumable qualification for the independent analytic navigator."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .controller import CBFConfig
from .geometry import BoxObstacle, CylinderObstacle, World3D
from .planner import NoRouteError
from .simulation import simulate_flight


DEFAULT_MAPS = 12
DEFAULT_PER_STRATUM = 4
CHECKPOINT_JOBS = 2
PROTOCOL = "nav3d_qualification_20260923_v1"


def config() -> CBFConfig:
    return CBFConfig(
        dt=0.05, body_radius=0.4, clearance_margin=0.2,
        max_horizontal_speed=7, max_vertical_speed=4,
        max_horizontal_acceleration=4, max_vertical_acceleration=3,
        activation_distance=15,
    )


def source_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "nav3d").glob("*.py")) + [root / "pyproject.toml", root / "uv.lock", root / "NAV3D_QUALIFICATION_PROTOCOL.md"]
    digest = sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def make_world(map_id: int) -> tuple[World3D, list[dict]]:
    rng = np.random.default_rng(20260923 + 1009 * map_id)
    wall_height = float(rng.uniform(9.0, 15.0))
    boxes = [
        BoxObstacle(np.array([27.0, 0.0, 0.0]), np.array([33.0, 60.0, wall_height]), "cross_wall"),
        BoxObstacle(np.array([11.0, float(rng.uniform(12, 28)), 0.0]), np.array([17.0, float(rng.uniform(35, 48)), float(rng.uniform(6, 14))]), "left_block"),
        BoxObstacle(np.array([43.0, float(rng.uniform(10, 25)), 0.0]), np.array([49.0, float(rng.uniform(34, 49)), float(rng.uniform(6, 14))]), "right_block"),
    ]
    cylinders = [
        CylinderObstacle(np.array([21.0, float(rng.uniform(7, 53))]), float(rng.uniform(1.5, 3.4)), 0.0, float(rng.uniform(7, 16)), "left_tower"),
        CylinderObstacle(np.array([39.0, float(rng.uniform(7, 53))]), float(rng.uniform(1.5, 3.4)), 0.0, float(rng.uniform(7, 16)), "right_tower"),
    ]
    obstacles = boxes + cylinders
    encoded = []
    for obstacle in obstacles:
        if isinstance(obstacle, BoxObstacle):
            encoded.append({"type": "box", "name": obstacle.name, "low": obstacle.low.tolist(), "high": obstacle.high.tolist()})
        else:
            encoded.append({"type": "cylinder", "name": obstacle.name, "center_xy": obstacle.center_xy.tolist(), "radius": obstacle.radius, "z_low": obstacle.z_low, "z_high": obstacle.z_high})
    return World3D([60.0, 60.0, 24.0], tuple(obstacles)), encoded


def decode_world(spec: dict) -> World3D:
    obstacles = []
    for item in spec["obstacles"]:
        if item["type"] == "box":
            obstacles.append(BoxObstacle(np.asarray(item["low"]), np.asarray(item["high"]), item["name"]))
        else:
            obstacles.append(CylinderObstacle(np.asarray(item["center_xy"]), item["radius"], item["z_low"], item["z_high"], item["name"]))
    return World3D(spec["size"], tuple(obstacles))


def build_manifest(map_count: int = DEFAULT_MAPS, per_stratum: int = DEFAULT_PER_STRATUM) -> dict:
    if map_count <= 0 or per_stratum <= 0:
        raise ValueError("sample counts must be positive")
    cfg = config()
    worlds, jobs = [], []
    for map_id in range(map_count):
        world, obstacles = make_world(map_id)
        worlds.append({"map_id": map_id, "size": world.size.tolist(), "obstacles": obstacles})
        groups: dict[str, list[dict]] = {}
        for blocked_index, label in enumerate(("blocked", "clear")):
            rng = np.random.default_rng(6042026 + 7919 * map_id + 17 * blocked_index)
            selected = []
            for _ in range(20000):
                start = np.array([rng.uniform(3, 9), rng.uniform(5, 55), rng.uniform(2, 20)])
                goal = np.array([rng.uniform(51, 57), rng.uniform(5, 55), rng.uniform(2, 20)])
                if min(world.clearance(start, cfg.inflation), world.clearance(goal, cfg.inflation)) <= 1.0:
                    continue
                direct_clear = world.segment_clear(start, goal, cfg.inflation)
                if direct_clear != (label == "clear"):
                    continue
                selected.append({"map_id": map_id, "stratum": label, "start": start.tolist(), "goal": goal.tolist()})
                if len(selected) == per_stratum:
                    break
            if len(selected) != per_stratum:
                raise RuntimeError(f"could not generate {label} routes for map {map_id}")
            groups[label] = selected
        for index in range(per_stratum):
            for label in ("blocked", "clear"):
                job = groups[label][index]
                job["job_id"] = f"m{map_id:02d}_{label}_{index:02d}"
                jobs.append(job)
    return {
        "protocol": PROTOCOL, "source_sha256": source_sha256(),
        "configuration": asdict(cfg), "max_steps": 1200, "goal_radius": 0.8,
        "checkpoint_jobs": CHECKPOINT_JOBS, "worlds": worlds, "jobs": jobs,
    }


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _timing(values: np.ndarray) -> dict:
    return {f"p{p}": float(np.quantile(values, p / 100)) if values.size else 0.0 for p in (50, 95, 99)}


def execute_job(output: Path, manifest: dict, job: dict) -> dict:
    world = decode_world(manifest["worlds"][job["map_id"]])
    cfg = CBFConfig(**manifest["configuration"])
    begun = perf_counter()
    result = {"job_id": job["job_id"], "map_id": job["map_id"], "stratum": job["stratum"]}
    try:
        flight = simulate_flight(world, job["start"], job["goal"], config=cfg, max_steps=manifest["max_steps"], goal_radius=manifest["goal_radius"])
    except NoRouteError as error:
        result.update({"outcome": "no_route", "error": str(error), "wall_seconds": perf_counter() - begun})
        return result
    trace_name = f"{job['job_id']}.npz"
    temporary = output / "traces" / (trace_name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, positions=flight.positions, velocities=flight.velocities,
                            accelerations=flight.accelerations, control_seconds=flight.control_seconds,
                            qp_seconds=flight.qp_seconds)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output / "traces" / trace_name)
    result.update({
        "outcome": "arrived" if flight.arrived else "navigation_timeout",
        "trace": f"traces/{trace_name}", "flight_seconds": flight.elapsed_seconds,
        "steps": flight.steps, "collision_count": flight.final_state.collision_count,
        "safety_cost": flight.final_state.safety_cost,
        "controller_infeasible_steps": flight.controller_infeasible_steps,
        "intervention_steps": flight.intervention_steps,
        "route_length": flight.route.geometric_length,
        "route_planning_seconds": flight.route.planning_seconds,
        "route_nodes": flight.route.candidate_nodes,
        "route_checked_edges": flight.route.checked_edges,
        "control_seconds": _timing(flight.control_seconds),
        "qp_seconds": _timing(flight.qp_seconds),
        "actual_path_length": float(np.linalg.norm(np.diff(flight.positions, axis=0), axis=1).sum()),
        "wall_seconds": perf_counter() - begun,
    })
    return result


def run(output: Path, *, resume: bool = False, stop_after_checkpoint: bool = False,
        manifest_override: dict | None = None) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_sha256():
            raise RuntimeError("source hash differs from frozen qualification manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        (output / "traces").mkdir()
        manifest = build_manifest() if manifest_override is None else manifest_override
        _atomic_json(output / "manifest.json", manifest)
    completed = 0
    for job in manifest["jobs"]:
        path = output / "results" / f"{job['job_id']}.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["job_id"] != job["job_id"]:
                raise RuntimeError("result job ID mismatch")
            if "trace" in saved and not (output / saved["trace"]).exists():
                raise RuntimeError("result trace missing")
            completed += 1
            continue
        record = execute_job(output, manifest, job)
        _atomic_json(path, record)
        completed += 1
        if completed % manifest["checkpoint_jobs"] == 0 or completed == len(manifest["jobs"]):
            checkpoint = {"protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"],
                          "completed": completed, "total": len(manifest["jobs"]),
                          "last_job_id": job["job_id"]}
            _atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint, ensure_ascii=False), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    checkpoint = {"protocol": manifest["protocol"], "source_sha256": manifest["source_sha256"],
                  "completed": completed, "total": len(manifest["jobs"]), "last_job_id": manifest["jobs"][-1]["job_id"]}
    _atomic_json(output / "checkpoint.json", checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, resume=args.resume, stop_after_checkpoint=args.stop_after_checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
