#!/usr/bin/env python3
"""Collect paired stochastic return trajectories for a defective resource law."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.stochastic_anchor_return import (
    StochasticAnchorReturnRecovery,
)
from experiments.directional_navigation.standard_baselines import atomic_json


PROTOCOL = "STOCHASTIC_DEFECTIVE_RETURN_COLLECTION_V1"
STOP_REQUESTED = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_anchors(path: Path, smoke_anchors: int | None) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    anchors = [dict(row) for row in document["anchors"] if int(row["leg"]) == 0]
    if smoke_anchors is None and len(anchors) != 310:
        raise RuntimeError(f"formal collection requires 310 task-leg anchors, got {len(anchors)}")
    if smoke_anchors is not None:
        anchors = anchors[:smoke_anchors]
    return anchors


def disturbance_seed(base: int, anchor_index: int, replicate: int) -> int:
    return int(base + 1009 * int(anchor_index) + int(replicate))


def trajectory_path(root: Path, arm: str, anchor_id: str, replicate: int) -> Path:
    return root / arm / "rows" / anchor_id / f"replicate_{replicate:02d}.json"


def save_row(root: Path, arm: str, row: dict[str, Any]) -> None:
    path = trajectory_path(root, arm, str(row["anchor_id"]), int(row["replicate"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, row)


def collect_arm(
    *,
    root: Path,
    arm: str,
    anchors: list[dict[str, object]],
    replicates: int,
    model: SAC,
    horizon: int,
    obstacles: int,
    num_envs: int,
    world_seed: int,
    disturbance_seed_base: int,
    sigma: float,
    rho: float,
    clip_sigma: float,
    resume: bool,
) -> bool:
    jobs = [(anchor, replicate) for anchor in anchors for replicate in range(replicates)]
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    for anchor, replicate in jobs:
        key = (str(anchor["anchor_id"]), replicate)
        path = trajectory_path(root, arm, *key)
        if resume and path.is_file():
            completed[key] = json.loads(path.read_text(encoding="utf-8"))
    pending = deque((anchor, replicate) for anchor, replicate in jobs if (str(anchor["anchor_id"]), replicate) not in completed)
    if not pending:
        return True
    worker_count = min(num_envs, len(pending))
    vec = SubprocVecEnv(
        [
            partial(
                StochasticAnchorReturnRecovery,
                horizon=horizon,
                obstacles=obstacles,
                hocbf=arm == "hocbf",
                execution_error_sigma=sigma,
                execution_error_rho=rho,
                execution_error_clip_sigma=clip_sigma,
            )
            for _ in range(worker_count)
        ],
        start_method="forkserver",
    )
    active: dict[int, tuple[dict[str, object], int, int]] = {}
    observations = np.zeros((worker_count, 2056), dtype=np.float32)

    def assign(worker_ids: list[int]) -> None:
        for worker_id in worker_ids:
            if not pending or STOP_REQUESTED:
                vec.env_method("park", indices=[worker_id])
                active.pop(worker_id, None)
                continue
            anchor, replicate = pending.popleft()
            seed = disturbance_seed(disturbance_seed_base, int(anchor["anchor_index"]), replicate)
            observation, _ = vec.env_method(
                "reset_from_anchor", anchor, world_seed, seed, indices=[worker_id]
            )[0]
            if np.asarray(observation).shape != (2056,) or not np.isfinite(observation).all():
                raise RuntimeError("invalid stochastic anchor observation")
            observations[worker_id] = observation
            active[worker_id] = (anchor, replicate, seed)

    try:
        assign(list(range(worker_count)))
        while active and not STOP_REQUESTED:
            actions, _ = model.predict(observations, deterministic=True)
            next_observations, _, dones, infos = vec.step(actions)
            observations[:] = next_observations
            finished = []
            for worker_id in sorted(active):
                if not dones[worker_id]:
                    continue
                anchor, replicate, seed = active[worker_id]
                episode = dict(infos[worker_id]["navigation_episode"])
                episode.pop("outcome", None)
                episode.update(
                    {
                        "protocol": PROTOCOL,
                        "arm": arm,
                        "anchor_id": str(anchor["anchor_id"]),
                        "anchor_index": int(anchor["anchor_index"]),
                        "scene_index": int(anchor["scene_index"]),
                        "source_transition_index": int(anchor["source_transition_index"]),
                        "distance_bucket": str(anchor["distance_bucket"]),
                        "replicate": replicate,
                        "disturbance_seed": seed,
                        "return_start_distance": float(
                            np.linalg.norm(
                                np.asarray(anchor["charger_goal"], dtype=np.float32)
                                - np.asarray(anchor["position"], dtype=np.float32)
                            )
                        ),
                    }
                )
                save_row(root, arm, episode)
                completed[(str(anchor["anchor_id"]), replicate)] = episode
                finished.append(worker_id)
            if finished:
                assign(finished)
                atomic_json(
                    root / "status.json",
                    {
                        "status": "COLLECTING",
                        "arm": arm,
                        "completed": len(completed),
                        "target": len(jobs),
                        "active": len(active),
                        "pid": os.getpid(),
                        "updated_unix": time.time(),
                    },
                )
    finally:
        vec.close()
    if STOP_REQUESTED:
        return False
    rows = [
        completed[(str(anchor["anchor_id"]), replicate)]
        for anchor, replicate in jobs
    ]
    atomic_json(
        root / arm / "summary.json",
        {
            "trajectories": len(rows),
            "anchors": len(anchors),
            "replicates": replicates,
            "goal_reached": sum(row["goal_reached"] for row in rows),
            "safe_goal": sum(row["safe_goal"] for row in rows),
            "timeouts": sum(row["termination"] == "deadline" for row in rows),
            "tasks_with_contact": sum(row["collision_count"] > 0 for row in rows),
            "mean_unified_contacts": float(np.mean([row["collision_count"] for row in rows])),
            "mean_energy": float(np.mean([row["energy"] for row in rows])),
            "collision_statistic": "one unified count per policy step",
            "contact_terminal": False,
        },
    )
    atomic_json(root / arm / "COMPLETE.json", {"trajectories": len(rows), "finished_unix": time.time()})
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--anchors",
        type=Path,
        default=Path("artifacts/rcps_meet_confirmation_fork_150scenes_20260904_v1/anchors.json"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/recovery_sac_ppo_scratch_20260906_v1/sac/checkpoint_000524288/model.zip"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/STOCHASTIC_DEFECTIVE_RETURN_COLLECTION_PROTOCOL.md"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4000)
    parser.add_argument("--obstacles", type=int, default=24)
    parser.add_argument("--world-seed", type=int, default=620001)
    parser.add_argument("--disturbance-seed-base", type=int, default=730001)
    parser.add_argument("--execution-error-sigma", type=float, default=0.04)
    parser.add_argument("--execution-error-rho", type=float, default=0.95)
    parser.add_argument("--execution-error-clip-sigma", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-anchors", type=int)
    args = parser.parse_args()
    if min(args.num_envs, args.replicates, args.horizon, args.obstacles) <= 0:
        raise ValueError("invalid collection size")
    for name in ("output_dir", "anchors", "model", "protocol"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    for path in (args.anchors, args.model, args.protocol):
        if not path.is_file():
            raise FileNotFoundError(path)
    anchors = load_anchors(args.anchors, args.smoke_anchors)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=args.resume)
    sources = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/stochastic_anchor_return.py",
        ROOT / "experiments/directional_navigation/anchor_return.py",
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
    ]
    contract = {
        "protocol": PROTOCOL,
        "arguments": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_dir", "anchors", "model", "protocol", "resume"}
        },
        "anchors_sha256": sha256_file(args.anchors),
        "model_sha256": sha256_file(args.model),
        "protocol_sha256": sha256_file(args.protocol),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "arms": ["raw", "hocbf"],
        "collision_contract": "nonterminal_repair_zero_velocity_unified_count_v1",
    }
    manifest = root / "manifest.json"
    if args.resume:
        if json.loads(manifest.read_text(encoding="utf-8"))["contract"] != contract:
            raise RuntimeError("resume contract mismatch")
    else:
        atomic_json(
            manifest,
            {
                "contract": contract,
                "command": [sys.executable, *sys.argv],
                "anchors": len(anchors),
                "trajectories_per_arm": len(anchors) * args.replicates,
                "formal_evidence": False,
                "development_feasibility_collection": args.smoke_anchors is None,
                "started_unix": time.time(),
            },
        )
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    model = SAC.load(args.model, device=args.device, print_system_info=False)
    try:
        for arm in ("raw", "hocbf"):
            complete = collect_arm(
                root=root,
                arm=arm,
                anchors=anchors,
                replicates=args.replicates,
                model=model,
                horizon=args.horizon,
                obstacles=args.obstacles,
                num_envs=args.num_envs,
                world_seed=args.world_seed,
                disturbance_seed_base=args.disturbance_seed_base,
                sigma=args.execution_error_sigma,
                rho=args.execution_error_rho,
                clip_sigma=args.execution_error_clip_sigma,
                resume=args.resume,
            )
            if not complete:
                atomic_json(root / "PAUSED.json", {"status": "PAUSED", "arm": arm, "pid": os.getpid()})
                return 130
        status = "SMOKE_COMPLETE_NOT_EVIDENCE" if args.smoke_anchors is not None else "COMPLETE"
        atomic_json(root / "RESULT.json", {"status": status, "finished_unix": time.time()})
        atomic_json(root / "COMPLETE.json", {"status": status, "result": str(root / "RESULT.json")})
        atomic_json(root / "status.json", {"status": status})
        (root / "PAUSED.json").unlink(missing_ok=True)
        return 0
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc(), "failed_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
