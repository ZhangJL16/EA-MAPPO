"""Resumable frozen-policy recovery audit; no training or automatic promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from experiments.directional_navigation.recovery import RecoveryCohort
from scripts import run_directional_lagrangian_pair as storage
from scripts.train_directional_ppo import atomic_json


def summarize(rows):
    return {
        "tasks": len(rows),
        "goal_reached": sum(r["goal_reached"] for r in rows),
        "safe_goal": sum(r["safe_goal"] for r in rows),
        "goal_after_collision": sum(
            r["goal_reached"] and r["collision_count"] > 0 for r in rows
        ),
        "timeout": sum(r["termination"] == "deadline" for r in rows),
        "total_collision_count": sum(r["collision_count"] for r in rows),
        "mean_collision_count": float(np.mean([r["collision_count"] for r in rows]))
        if rows
        else None,
        "physical_steps": sum(r["steps"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.tasks <= 0:
        raise ValueError("positive task count required")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    sources = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
        ROOT / "envs/UAVEnergyDelivery.py",
        ROOT / "experiments/directional_navigation/environment.py",
        ROOT / "experiments/directional_navigation/lagrangian.py",
        ROOT / "experiments/directional_navigation/cohort.py",
        ROOT / "experiments/directional_navigation/features.py",
    ]
    checkpoints = {
        arm: args.source / arm / "checkpoint_0064/state.pt"
        for arm in ("lagrangian", "control")
    }
    contract = {
        "protocol": "user_locked_collision_recovery_audit_v1",
        "tasks_per_mode": args.tasks,
        "num_envs": 8,
        "horizon": 4000,
        "obstacles": 24,
        "training": False,
        "formal_500": False,
        "counting": "one unified collision per contacted policy step",
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "checkpoint_hashes": {
            k: hashlib.sha256(p.read_bytes()).hexdigest()
            for k, p in checkpoints.items()
        },
    }
    if args.resume:
        if json.loads((root / "manifest.json").read_text()) != contract:
            raise ValueError("resume contract mismatch")
    else:
        atomic_json(root / "manifest.json", contract)
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, storage.pause)
    signal.signal(signal.SIGINT, storage.pause)
    started = time.perf_counter()
    atomic_json(root / "status.json", {"status": "STARTING", "pid": os.getpid()})
    vec = SubprocVecEnv(
        [partial(RecoveryCohort) for _ in range(8)], start_method="forkserver"
    )
    results = {}
    try:
        for arm, checkpoint in checkpoints.items():
            set_random_seed(193500001, using_cuda=True)
            policy = TwoValuePolicy(
                vec.observation_space,
                vec.action_space,
                lambda _: 3e-4,
                features_extractor_class=DirectionalLidarExtractor,
                features_extractor_kwargs={"remaining_time": True},
                share_features_extractor=False,
                net_arch={"pi": [256, 256], "vf": [256, 256]},
                log_std_init=float(np.log(0.5)),
            ).to("cuda")
            policy.load_state_dict(
                torch.load(checkpoint, map_location="cuda", weights_only=False)["model"]
            )
            before = storage.state_hash(policy)
            for mode in ("deterministic", "stochastic"):
                label = f"{arm}_{mode}"
                directory = root / label
                directory.mkdir(exist_ok=True)
                for start in range(0, args.tasks, 8):
                    path = directory / f"cohort_{start // 8:04d}.json"
                    seeds = list(
                        range(593800001 + start, 593800001 + min(start + 8, args.tasks))
                    )
                    if path.exists():
                        saved = json.loads(path.read_text())
                        if sorted(r["seed"] for r in saved) != seeds:
                            raise ValueError("saved seed cohort mismatch")
                        continue
                    if storage.PAUSED or (root / "PAUSE").exists():
                        atomic_json(
                            root / "status.json",
                            {"status": "PAUSED", "arm": arm, "mode": mode},
                        )
                        return
                    set_random_seed(693800001 + seeds[0], using_cuda=True)
                    _, rows, _ = collect_cohort(
                        policy,
                        vec,
                        seeds,
                        deterministic=mode == "deterministic",
                        training=False,
                    )
                    if storage.state_hash(policy) != before:
                        raise RuntimeError("frozen policy changed")
                    for row in rows:
                        row.pop(
                            "outcome"
                        )  # Do not conflate contact with task termination.
                        if not 0 <= row["collision_count"] <= row["steps"]:
                            raise RuntimeError("invalid unified collision count")
                        if not np.isfinite(
                            [
                                row["raw_return"],
                                row["energy"],
                                row["remaining_distance"],
                            ]
                        ).all():
                            raise FloatingPointError("nonfinite evaluation record")
                    atomic_json(path, sorted(rows, key=lambda r: r["seed"]))
                    status = {
                        "status": "EVALUATING",
                        "pid": os.getpid(),
                        "arm": arm,
                        "mode": mode,
                        "completed_in_mode": start + len(rows),
                        "weights_unchanged": True,
                        "session_seconds": time.perf_counter() - started,
                    }
                    atomic_json(root / "status.json", status)
                    print(json.dumps(status), flush=True)
                all_rows = [
                    r
                    for p in sorted(directory.glob("cohort_*.json"))
                    for r in json.loads(p.read_text())
                ]
                results[label] = summarize(all_rows)
                atomic_json(directory / "RESULT.json", results[label])
        atomic_json(
            root / "RESULT.json",
            {
                "status": "COMPLETE",
                "results": results,
                "weights_unchanged": True,
                "training": False,
                "formal_500": False,
                "session_seconds": time.perf_counter() - started,
            },
        )
        atomic_json(root / "status.json", {"status": "COMPLETE"})
    except BaseException as exc:
        atomic_json(
            root / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    main()
