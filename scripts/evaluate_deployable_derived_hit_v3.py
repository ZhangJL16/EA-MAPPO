#!/usr/bin/env python3
"""Frozen derived-hit v3 evaluation; unchanged v2 fixed500 task/resume logic."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.deployable_observation import DIM, DeployableRecovery
from experiments.directional_navigation.standard_baselines import atomic_json
from scripts.resume_recovery_sac_ppo_stratified import BUCKETS, TASKS, TASK_SHA256, file_hash, load_tasks

STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


class ParkedDeployableEvaluation(DeployableRecovery):
    """Park a completed worker without sampling another training episode."""

    def reset(self, *, seed=None, options=None):
        if seed is None and self.park_after_terminal:
            self.parked = True
            return self.parked_observation.copy(), {}
        return super().reset(seed=seed, options=options)


def summarize(rows: list[dict]) -> dict:
    goals = [r for r in rows if r["goal_reached"]]
    return {
        "tasks": len(rows), "goal_reached": len(goals),
        "safe_goal": sum(r["safe_goal"] for r in rows),
        "timeouts": sum(r["termination"] == "deadline" for r in rows),
        "mean_collision_count": float(np.mean([r["collision_count"] for r in rows])) if rows else None,
        "median_collision_count": float(np.median([r["collision_count"] for r in rows])) if rows else None,
        "mean_success_path_ratio": float(np.mean([r["success_path_ratio"] for r in goals])) if goals else None,
        "mean_energy": float(np.mean([r["energy"] for r in rows])) if rows else None,
    }


def validate_prefix(rows: list[dict], tasks: list[dict], base_seed: int) -> None:
    if len(rows) > len(tasks):
        raise ValueError("saved prefix longer than task set")
    for index, row in enumerate(rows):
        task = tasks[index]
        expected = (index, int(task["source_task_index"]), task["distance_bucket"],
                    base_seed + int(task["source_task_index"]))
        actual = (row["evaluation_index"], row["source_task_index"], row["distance_bucket"], row["seed"])
        if actual != expected:
            raise ValueError("saved evaluation prefix does not match immutable tasks")
        if row["safe_goal"] != (bool(row["goal_reached"]) and row["collision_count"] == 0):
            raise ValueError("saved collision-free arrival is inconsistent")


def evaluate(model, root: Path, args: argparse.Namespace, tasks: list[dict], base_seed: int) -> bool:
    path = root / "evaluation_stratified.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    validate_prefix(rows, tasks, base_seed)
    if len(rows) == len(tasks):
        return True
    workers = min(args.num_envs, len(tasks) - len(rows))
    vec = SubprocVecEnv([
        partial(ParkedDeployableEvaluation, battery_capacity=args.battery_capacity,
                initial_soc=args.initial_soc, horizon=args.horizon, obstacles=args.obstacles)
        for _ in range(workers)
    ], start_method="forkserver")
    def stopped() -> bool:
        return STOP or (root / "PAUSE").exists()
    try:
        while len(rows) < len(tasks) and not stopped():
            count = min(workers, len(tasks) - len(rows))
            observations, group = [], []
            for worker in range(workers):
                index = len(rows) + worker
                task = tasks[index] if worker < count else tasks[0]
                observation, _ = vec.env_method(
                    "reset", seed=base_seed + int(task["source_task_index"]),
                    options={"start_position": task["start_position"],
                             "start_velocity": task["initial_velocity"],
                             "task_point": task["goal_position"]}, indices=[worker],
                )[0]
                if worker >= count:
                    vec.env_method("park", indices=[worker])
                observations.append(observation)
            obs = np.stack(observations)
            active = np.arange(workers) < count
            if obs.shape != (workers, DIM) or not np.isfinite(obs).all():
                raise ValueError("invalid v2 reset observation")
            for _ in range(args.horizon):
                action, _ = model.predict(obs, deterministic=True)
                if not np.isfinite(action).all():
                    raise FloatingPointError("nonfinite evaluation action")
                obs, _, dones, infos = vec.step(action)
                if not np.isfinite(obs).all():
                    raise FloatingPointError("nonfinite evaluation observation")
                for worker in np.flatnonzero(active & dones):
                    index = len(rows) + int(worker)
                    task = tasks[index]
                    row = dict(infos[worker]["navigation_episode"])
                    row.pop("outcome", None)
                    row.update(evaluation_index=index, source_task_index=int(task["source_task_index"]),
                               distance_bucket=task["distance_bucket"],
                               straight_line_distance=float(task["straight_line_distance"]))
                    group.append(row)
                    active[worker] = False
                if not active.any():
                    break
                if stopped():
                    return False  # Only the uncommitted batch is replayed.
            if active.any():
                raise RuntimeError("evaluation task exceeded registered horizon")
            rows.extend(sorted(group, key=lambda r: r["evaluation_index"]))
            validate_prefix(rows, tasks, base_seed)
            atomic_json(path, rows)
            atomic_json(root / "status.json", {
                "status": "EVALUATING", "pid": os.getpid(), "completed": len(rows),
                "target": len(tasks), "updated_unix": time.time(),
            })
    finally:
        vec.close()
    return len(rows) == len(tasks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Only five tasks x two steps; not formal evidence")
    args = parser.parse_args()
    if args.num_envs <= 0:
        raise ValueError("positive worker count required")
    source, root = args.source.resolve(), args.output_dir.resolve()
    if root == source or source in root.parents:
        raise ValueError("evaluation must use a separate output directory")
    task_rows, base_seed = load_tasks()
    checkpoint = source / "sac" / json.loads((source / "sac/latest.json").read_text())["checkpoint"]
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    training = metadata["contract"]
    settings = training["arguments"]
    if training["protocol"] != "deployable_distance_derived_hit_v3":
        raise ValueError("requires a derived-hit v3 training contract")
    if training["observation_dim"] != DIM or metadata["transitions"] != settings["target_steps"]:
        raise ValueError("requires the completed v3 checkpoint")
    if any(file_hash(ROOT / p) != h for p, h in training["source_hashes"].items()):
        raise ValueError("training source changed; explicitly audit before evaluation")
    if (checkpoint / "model.zip").stat().st_size != metadata["files"]["model.zip"]:
        raise ValueError("incomplete model checkpoint")
    args.battery_capacity = settings["battery_capacity"]
    args.initial_soc = settings["initial_soc"]
    args.horizon = 2 if args.smoke else settings["horizon"]
    args.obstacles = settings["obstacles"]
    tasks = task_rows[:5] if args.smoke else task_rows
    contract = {
        "protocol": "deployable_derived_hit_v3_fixed500_v1",
        "training_protocol": training["protocol"],
        "internal_lidar_shape": training["lidar_shape"], "formal": not args.smoke,
        "task_source": str(TASKS), "task_sha256": TASK_SHA256,
        "task_seed": base_seed, "task_count": len(tasks), "horizon": args.horizon,
        "obstacles": args.obstacles, "battery_capacity": args.battery_capacity,
        "initial_soc": args.initial_soc, "checkpoint": str(checkpoint),
        "model_sha256": file_hash(checkpoint / "model.zip"),
        "metadata_sha256": file_hash(checkpoint / "metadata.json"),
        "source_sha256": file_hash(Path(__file__)),
        "task_loader_sha256": file_hash(ROOT / "scripts/resume_recovery_sac_ppo_stratified.py"),
        "device": args.device, "num_envs": args.num_envs,
        "deterministic_policy": True, "train_updates": False,
        "collision_contract": "locked_nonterminal_repair_unified_policy_step_count",
        "claim_scope": "navigation_only_not_energy_sustainability",
    }
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("evaluation resume contract mismatch")
    else:
        root.mkdir(parents=True, exist_ok=False)
        atomic_json(root / "manifest.json", {"contract": contract, "started_unix": time.time()})
    if (root / "PAUSE").exists():
        raise ValueError("PAUSE request still present")
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        model = SAC.load(checkpoint / "model.zip", device=args.device)
        if model.observation_space.shape != (DIM,):
            raise ValueError("loaded model has the wrong observation contract")
        model.policy.set_training_mode(False)
        atomic_json(root / "status.json", {"status": "STARTING", "pid": os.getpid(), "target": len(tasks)})
        complete = evaluate(model, root, args, tasks, base_seed)
        if complete:
            rows = json.loads((root / "evaluation_stratified.json").read_text())
            summary = {**summarize(rows), "contract": contract,
                       "distance_buckets": {b: summarize([r for r in rows if r["distance_bucket"] == b])
                                            for b in BUCKETS}}
            atomic_json(root / "summary.json", summary)
        atomic_json(root / "status.json", {
            "status": "COMPLETE" if complete else "PAUSED", "pid": os.getpid(),
            "updated_unix": time.time(), "automatic_next_experiment": False,
        })
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
