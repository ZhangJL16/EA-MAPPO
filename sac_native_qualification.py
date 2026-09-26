"""Read-only frozen SAC + original HOCBF check on fresh native cylinder maps."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

OLD_ROOT = Path("/home/zjl/mappo")
sys.path[:0] = [str(OLD_ROOT / "persistent_uav_throughput_v1"), str(OLD_ROOT)]

from persistent_uav.navigation import FrozenNavigator, MODEL, MODEL_SHA  # noqa: E402
from persistent_uav.streams import legal_position  # noqa: E402

MAP_SEEDS = tuple(720260921 + index for index in range(4))
PAIRS_PER_MAP = 4
CHECKPOINT_JOBS = 2
PROTOCOL = "frozen_sac_native_new_maps_20260923_v1"


def source_hash() -> str:
    paths = [
        Path(__file__),
        OLD_ROOT / "persistent_uav_throughput_v1/persistent_uav/navigation.py",
        OLD_ROOT / "persistent_uav_throughput_v1/persistent_uav/streams.py",
        OLD_ROOT / "experiments/directional_navigation/recovery.py",
        OLD_ROOT / "experiments/directional_navigation/features.py",
        OLD_ROOT / "experiments/directional_navigation/correction_supervision.py",
        OLD_ROOT / "envs/UAVEnergyDeliverySAC.py",
    ]
    digest = sha256()
    for path in paths:
        digest.update(str(path.resolve()).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def build_manifest() -> dict:
    if sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA:
        raise RuntimeError("frozen SAC checkpoint checksum mismatch")
    maps = []
    for map_index, seed in enumerate(MAP_SEEDS):
        nav = FrozenNavigator(1e9, seed=seed)
        try:
            layout = nav.layout
        finally:
            nav.close()
        rng = np.random.default_rng(2026092300 + map_index)
        pairs = [{"start": list(legal_position(rng, layout)), "goal": list(legal_position(rng, layout))}
                 for _ in range(PAIRS_PER_MAP)]
        maps.append({"map_id": map_index, "seed": seed, "layout": layout, "pairs": pairs})
    jobs = []
    for pair_index in range(PAIRS_PER_MAP):
        for item in maps:
            jobs.append({"job_id": f"m{item['map_id']:02d}_p{pair_index:02d}",
                         "map_id": item["map_id"], "pair_index": pair_index})
    return {"protocol": PROTOCOL, "source_sha256": source_hash(), "model_sha256": MODEL_SHA,
            "domain": [4000, 4000, 400], "obstacle_family": "24 full-height cylinders",
            "goal_radius": 5.0, "option_step_limit": 4000,
            "model": str(MODEL), "checkpoint_jobs": CHECKPOINT_JOBS,
            "maps": maps, "jobs": jobs, "neural_updates": 0}


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def execute_job(manifest: dict, job: dict) -> dict:
    item = manifest["maps"][job["map_id"]]
    pair = item["pairs"][job["pair_index"]]
    begun = perf_counter()
    nav = FrozenNavigator(1e9, layout=item["layout"], start=pair["start"], seed=item["seed"])
    energy_used = 0.0
    try:
        nav.start_leg(pair["goal"])
        reached = nav.reached()
        timeout = False
        while not (reached or timeout):
            step = nav.advance_flight(.2)
            energy_used += step.energy_used
            reached, timeout = step.reached, step.timeout
            if step.depleted:
                raise RuntimeError("virtual pilot battery unexpectedly depleted")
        return {"job_id": job["job_id"], "map_id": job["map_id"],
                "outcome": "arrived" if reached else "navigation_timeout",
                "duration_seconds": nav.time, "energy_synthetic_units": energy_used,
                "policy_steps": nav.policy_steps, "collision_count": nav.contacts,
                "hocbf_fallback_steps": nav.hocbf_fallback_steps,
                "final_position": nav.position.tolist(),
                "goal_distance": float(np.linalg.norm(nav.position - pair["goal"])),
                "wall_seconds": perf_counter() - begun}
    finally:
        nav.close()


def run(output: Path, *, resume: bool = False, stop_after_checkpoint: bool = False,
        manifest_override: dict | None = None, execute=execute_job) -> dict:
    if resume:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest["source_sha256"] != source_hash() or manifest["model_sha256"] != sha256(MODEL.read_bytes()).hexdigest():
            raise RuntimeError("source or model differs from frozen SAC manifest")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "results").mkdir()
        manifest = build_manifest() if manifest_override is None else manifest_override
        atomic_json(output / "manifest.json", manifest)
    count = 0
    for job in manifest["jobs"]:
        result_path = output / "results" / f"{job['job_id']}.json"
        if result_path.exists():
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if row["job_id"] != job["job_id"]:
                raise RuntimeError("saved job ID mismatch")
            count += 1
            continue
        row = execute(manifest, job)
        atomic_json(result_path, row)
        count += 1
        if count % manifest["checkpoint_jobs"] == 0 or count == len(manifest["jobs"]):
            checkpoint = {"protocol": manifest["protocol"], "completed": count,
                          "total": len(manifest["jobs"]), "last_job_id": job["job_id"],
                          "source_sha256": manifest["source_sha256"], "model_sha256": manifest["model_sha256"]}
            atomic_json(output / "checkpoint.json", checkpoint)
            print(json.dumps(checkpoint), flush=True)
            if stop_after_checkpoint:
                return checkpoint
    checkpoint = {"protocol": manifest["protocol"], "completed": count,
                  "total": len(manifest["jobs"]), "last_job_id": manifest["jobs"][-1]["job_id"],
                  "source_sha256": manifest["source_sha256"], "model_sha256": manifest["model_sha256"]}
    atomic_json(output / "checkpoint.json", checkpoint)
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-checkpoint", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, resume=args.resume, stop_after_checkpoint=args.stop_after_checkpoint)))


if __name__ == "__main__":
    main()
