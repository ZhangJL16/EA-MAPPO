"""Resumable paired and held-out qualification of the path-tracking navigator."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from nav3d.controller import CBFConfig
from nav3d.planner import NoRouteError
from nav3d.qualify import config, decode_world, make_world
from .controller import PLANNING_CLEARANCE_MARGIN, WAYPOINT_RADIUS, fly_segment_route


ROOT = Path(__file__).resolve().parents[1]
V1_MANIFEST = ROOT / "artifacts/nav3d_qualification_20260923/manifest.json"
PROTOCOL = "nav3d_v2_segment_tracking_20260923"


def source_hash() -> str:
    files = sorted((ROOT / "nav3d").glob("*.py")) + sorted((ROOT / "nav3d_v2").glob("*.py"))
    files += [ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "NAV3D_QUALIFICATION_PROTOCOL.md", ROOT / "NAV3D_V2_PROTOCOL_20260923.md"]
    digest = sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _encoded_world(world) -> dict:
    encoded = []
    for obstacle in world.obstacles:
        if hasattr(obstacle, "low"):
            encoded.append({"type": "box", "name": obstacle.name, "low": obstacle.low.tolist(), "high": obstacle.high.tolist()})
        else:
            encoded.append({"type": "cylinder", "name": obstacle.name, "center_xy": obstacle.center_xy.tolist(), "radius": obstacle.radius, "z_low": obstacle.z_low, "z_high": obstacle.z_high})
    return {"size": world.size.tolist(), "obstacles": encoded}


def build_manifest(suite: str, *, map_count: int = 12, per_stratum: int = 4) -> dict:
    cfg = config()
    if suite == "paired":
        original = json.loads(V1_MANIFEST.read_text())
        from nav3d.qualify import source_sha256
        if original["source_sha256"] != source_sha256():
            raise RuntimeError("frozen v1 source differs from original manifest")
        worlds, jobs = original["worlds"], original["jobs"]
    elif suite == "holdout":
        if map_count <= 0 or per_stratum <= 0:
            raise ValueError("sample counts must be positive")
        worlds, jobs = [], []
        for map_id in range(map_count):
            seed_map_id = 100 + map_id
            world, _ = make_world(seed_map_id)
            worlds.append({"map_id": map_id, "seed_map_id": seed_map_id, **_encoded_world(world)})
            groups = {}
            for label_index, label in enumerate(("blocked", "clear")):
                rng = np.random.default_rng(6042026 + 7919 * seed_map_id + 17 * label_index)
                selected = []
                for _ in range(20000):
                    start = np.array([rng.uniform(3, 9), rng.uniform(5, 55), rng.uniform(2, 20)])
                    goal = np.array([rng.uniform(51, 57), rng.uniform(5, 55), rng.uniform(2, 20)])
                    if min(world.clearance(start, cfg.inflation), world.clearance(goal, cfg.inflation)) <= 1.0:
                        continue
                    if world.segment_clear(start, goal, cfg.inflation) != (label == "clear"):
                        continue
                    selected.append({"map_id": map_id, "stratum": label, "start": start.tolist(), "goal": goal.tolist()})
                    if len(selected) == per_stratum:
                        break
                if len(selected) != per_stratum:
                    raise RuntimeError(f"missing {label} routes on holdout map {map_id}")
                groups[label] = selected
            for index in range(per_stratum):
                for label in ("blocked", "clear"):
                    job = groups[label][index]
                    job["job_id"] = f"h{map_id:02d}_{label}_{index:02d}"
                    jobs.append(job)
    else:
        raise ValueError("suite must be paired or holdout")
    return {"protocol": PROTOCOL, "suite": suite, "source_sha256": source_hash(),
            "configuration": cfg.__dict__, "planning_clearance_margin": PLANNING_CLEARANCE_MARGIN,
            "waypoint_radius": WAYPOINT_RADIUS, "goal_radius": 0.8, "max_steps": 1200,
            "checkpoint_jobs": 2, "worlds": worlds, "jobs": jobs,
            "paired_parent_source_sha256": json.loads(V1_MANIFEST.read_text())["source_sha256"]}


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _quantiles(values: np.ndarray) -> dict:
    return {f"p{p}": float(np.quantile(values, p / 100)) if values.size else 0.0 for p in (50, 95, 99)}


def execute_job(output: Path, manifest: dict, job: dict) -> dict:
    world = decode_world(manifest["worlds"][job["map_id"]])
    cfg = CBFConfig(**manifest["configuration"])
    begun = perf_counter()
    row = {"job_id": job["job_id"], "map_id": job["map_id"], "stratum": job["stratum"]}
    try:
        flight = fly_segment_route(world, job["start"], job["goal"], config=cfg,
                                   max_steps=manifest["max_steps"], goal_radius=manifest["goal_radius"])
    except NoRouteError as error:
        row.update({"outcome": "no_route", "error": str(error), "wall_seconds": perf_counter() - begun})
        return row
    trace = output / "traces" / f"{job['job_id']}.npz"
    temporary = trace.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, positions=flight.positions, velocities=flight.velocities,
                            accelerations=flight.accelerations, control_seconds=flight.control_seconds,
                            qp_seconds=flight.qp_seconds)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(trace)
    row.update({"outcome": "arrived" if flight.arrived else "navigation_timeout",
                "trace": f"traces/{trace.name}", "flight_seconds": flight.elapsed_seconds,
                "steps": flight.steps, "collision_count": flight.final_state.collision_count,
                "safety_cost": flight.final_state.safety_cost,
                "controller_infeasible_steps": flight.controller_infeasible_steps,
                "intervention_steps": flight.intervention_steps,
                "route_length": flight.route.geometric_length,
                "route_planning_seconds": flight.route.planning_seconds,
                "route_nodes": flight.route.candidate_nodes,
                "route_checked_edges": flight.route.checked_edges,
                "control_seconds": _quantiles(flight.control_seconds),
                "qp_seconds": _quantiles(flight.qp_seconds),
                "actual_path_length": float(np.linalg.norm(np.diff(flight.positions, axis=0), axis=1).sum()),
                "wall_seconds": perf_counter() - begun})
    return row


def run(output: Path, *, suite: str, resume: bool = False,
        stop_after_checkpoint: bool = False, manifest_override: dict | None = None,
        execute=execute_job) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text())
        if manifest["suite"] != suite or manifest["source_sha256"] != source_hash():
            raise RuntimeError("suite/source differs from frozen v2 manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        (output / "traces").mkdir()
        manifest = build_manifest(suite) if manifest_override is None else manifest_override
        atomic_json(output / "manifest.json", manifest)
    count = 0
    for job in manifest["jobs"]:
        path = output / "results" / f"{job['job_id']}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row["job_id"] != job["job_id"] or ("trace" in row and not (output / row["trace"]).exists()):
                raise RuntimeError("saved result/trace mismatch")
            count += 1
            continue
        row = execute(output, manifest, job)
        atomic_json(path, row)
        count += 1
        if count % manifest["checkpoint_jobs"] == 0 or count == len(manifest["jobs"]):
            checkpoint = {"protocol": manifest["protocol"], "suite": suite,
                          "source_sha256": manifest["source_sha256"], "completed": count,
                          "total": len(manifest["jobs"]), "last_job_id": job["job_id"]}
            atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    checkpoint = {"protocol": manifest["protocol"], "suite": suite,
                  "source_sha256": manifest["source_sha256"], "completed": count,
                  "total": len(manifest["jobs"]), "last_job_id": manifest["jobs"][-1]["job_id"]}
    atomic_json(output / "checkpoint.json", checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--suite", required=True, choices=("paired", "holdout"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, suite=args.suite, resume=args.resume,
                         stop_after_checkpoint=args.stop_after_checkpoint), ensure_ascii=False))


if __name__ == "__main__":
    main()
