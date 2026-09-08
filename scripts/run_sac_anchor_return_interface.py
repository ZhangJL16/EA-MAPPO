#!/usr/bin/env python3
"""Evaluate scratch SAC return behavior on preserved anchors, raw and HOCBF."""

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

from experiments.directional_navigation.anchor_return import AnchorReturnRecovery
from experiments.directional_navigation.standard_baselines import atomic_json


PROTOCOL = "SAC_ANCHOR_RETURN_EXECUTED_INTERFACE_V1"
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
        raise RuntimeError(f"formal evaluation requires 310 task-leg anchors, got {len(anchors)}")
    if smoke_anchors is not None:
        anchors = anchors[:smoke_anchors]
    if len({str(row["anchor_id"]) for row in anchors}) != len(anchors):
        raise RuntimeError("anchor IDs are not unique")
    return anchors


def row_path(root: Path, arm: str, anchor_id: str) -> Path:
    return root / arm / "rows" / f"{anchor_id}.json"


def save_row(root: Path, arm: str, row: dict[str, Any]) -> None:
    path = row_path(root, arm, str(row["anchor_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, row)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    goals = [row for row in rows if row["goal_reached"]]
    steps = sum(int(row["steps"]) for row in rows)
    contacts = sum(int(row["collision_count"]) for row in rows)
    return {
        "tasks": len(rows),
        "goal_reached": sum(bool(row["goal_reached"]) for row in rows),
        "safe_goal": sum(bool(row["safe_goal"]) for row in rows),
        "timeouts": sum(row["termination"] == "deadline" for row in rows),
        "tasks_with_contact": sum(int(row["collision_count"]) > 0 for row in rows),
        "mean_collision_count": float(np.mean([row["collision_count"] for row in rows])),
        "median_collision_count": float(np.median([row["collision_count"] for row in rows])),
        "contact_step_rate": float(contacts / steps),
        "mean_energy_to_charger": float(np.mean([row["energy"] for row in rows])),
        "mean_success_path_ratio": (
            float(np.mean([row["success_path_ratio"] for row in goals])) if goals else None
        ),
        "hocbf_step_intervention_rate": float(
            sum(int(row["hocbf_step_interventions"]) for row in rows) / steps
        ),
        "collision_statistic": "one unified count per policy step",
        "contact_terminal": False,
    }


def collect_arm(
    *,
    root: Path,
    arm: str,
    anchors: list[dict[str, object]],
    model: SAC,
    horizon: int,
    obstacles: int,
    num_envs: int,
    world_seed: int,
    resume: bool,
) -> bool:
    completed: dict[str, dict[str, Any]] = {}
    for anchor in anchors:
        path = row_path(root, arm, str(anchor["anchor_id"]))
        if resume and path.is_file():
            completed[str(anchor["anchor_id"])] = json.loads(path.read_text(encoding="utf-8"))
    pending = deque(anchor for anchor in anchors if str(anchor["anchor_id"]) not in completed)
    if not pending:
        rows = [completed[str(anchor["anchor_id"])] for anchor in anchors]
        atomic_json(root / arm / "evaluation.json", rows)
        atomic_json(root / arm / "summary.json", summarize(rows))
        return True
    count = min(num_envs, len(pending))
    vec = SubprocVecEnv(
        [partial(AnchorReturnRecovery, horizon=horizon, obstacles=obstacles, hocbf=arm == "hocbf")
         for _ in range(count)],
        start_method="forkserver",
    )
    active: dict[int, dict[str, object]] = {}
    observations = np.zeros((count, 2056), dtype=np.float32)

    def assign(worker_ids: list[int]) -> None:
        for worker_id in worker_ids:
            if not pending or STOP_REQUESTED:
                vec.env_method("park", indices=[worker_id])
                active.pop(worker_id, None)
                continue
            anchor = pending.popleft()
            observation, _ = vec.env_method(
                "reset_from_anchor", anchor, world_seed, indices=[worker_id]
            )[0]
            if np.asarray(observation).shape != (2056,) or not np.isfinite(observation).all():
                raise RuntimeError("invalid anchor-return observation")
            observations[worker_id] = observation
            active[worker_id] = anchor

    try:
        assign(list(range(count)))
        while active and not STOP_REQUESTED:
            actions, _ = model.predict(observations, deterministic=True)
            next_observations, _, dones, infos = vec.step(actions)
            observations[:] = next_observations
            finished: list[int] = []
            for worker_id in sorted(active):
                if not dones[worker_id]:
                    continue
                anchor = active[worker_id]
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
                        "return_start_distance": float(
                            np.linalg.norm(
                                np.asarray(anchor["charger_goal"], dtype=np.float32)
                                - np.asarray(anchor["position"], dtype=np.float32)
                            )
                        ),
                    }
                )
                save_row(root, arm, episode)
                completed[str(anchor["anchor_id"])] = episode
                finished.append(worker_id)
            if finished:
                assign(finished)
                atomic_json(
                    root / "status.json",
                    {
                        "status": "EVALUATING",
                        "arm": arm,
                        "completed": len(completed),
                        "target": len(anchors),
                        "active": len(active),
                        "pid": os.getpid(),
                        "updated_unix": time.time(),
                    },
                )
    finally:
        vec.close()
    if STOP_REQUESTED:
        return False
    rows = [completed[str(anchor["anchor_id"])] for anchor in anchors]
    atomic_json(root / arm / "evaluation.json", rows)
    atomic_json(root / arm / "summary.json", summarize(rows))
    atomic_json(root / arm / "COMPLETE.json", {"tasks": len(rows), "finished_unix": time.time()})
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
        default=Path(
            "artifacts/recovery_sac_ppo_scratch_20260906_v1/sac/"
            "checkpoint_000524288/model.zip"
        ),
    )
    parser.add_argument("--protocol", type=Path, default=Path("docs/SAC_ANCHOR_RETURN_INTERFACE_PROTOCOL.md"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4000)
    parser.add_argument("--obstacles", type=int, default=24)
    parser.add_argument("--world-seed", type=int, default=620001)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-anchors", type=int)
    args = parser.parse_args()
    if min(args.num_envs, args.horizon, args.obstacles) <= 0:
        raise ValueError("invalid execution size")
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
        ROOT / "experiments/directional_navigation/anchor_return.py",
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
    ]
    contract = {
        "protocol": PROTOCOL,
        "arguments": {
            "device": args.device,
            "num_envs": args.num_envs,
            "horizon": args.horizon,
            "obstacles": args.obstacles,
            "world_seed": args.world_seed,
            "smoke_anchors": args.smoke_anchors,
        },
        "anchors_sha256": sha256_file(args.anchors),
        "model_sha256": sha256_file(args.model),
        "protocol_sha256": sha256_file(args.protocol),
        "source_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in sources},
        "arms": ["raw", "hocbf"],
        "collision_contract": "nonterminal_repair_zero_velocity_unified_count_v1",
    }
    manifest_path = root / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8"))["contract"] != contract:
            raise RuntimeError("resume contract mismatch")
    else:
        atomic_json(
            manifest_path,
            {
                "contract": contract,
                "command": [sys.executable, *sys.argv],
                "anchors": len(anchors),
                "formal_evidence": False,
                "development_mechanism_diagnostic": args.smoke_anchors is None,
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
            if not collect_arm(
                root=root,
                arm=arm,
                anchors=anchors,
                model=model,
                horizon=args.horizon,
                obstacles=args.obstacles,
                num_envs=args.num_envs,
                world_seed=args.world_seed,
                resume=args.resume,
            ):
                atomic_json(root / "PAUSED.json", {"status": "PAUSED", "arm": arm, "pid": os.getpid()})
                return 130
        final = {
            "status": "SMOKE_COMPLETE_NOT_EVIDENCE" if args.smoke_anchors is not None else "COMPLETE",
            "raw": json.loads((root / "raw" / "summary.json").read_text(encoding="utf-8")),
            "hocbf": json.loads((root / "hocbf" / "summary.json").read_text(encoding="utf-8")),
            "finished_unix": time.time(),
        }
        atomic_json(root / "RESULT.json", final)
        atomic_json(root / "COMPLETE.json", {"status": final["status"], "result": str(root / "RESULT.json")})
        atomic_json(root / "status.json", {"status": final["status"]})
        (root / "PAUSED.json").unlink(missing_ok=True)
        return 0
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc(), "failed_unix": time.time()})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
