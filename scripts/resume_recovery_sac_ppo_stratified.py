#!/usr/bin/env python3
"""Resume the scratch pair with the legacy immutable five-distance evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.standard_baselines import atomic_json
from scripts import run_recovery_sac_ppo_scratch as base

TASKS = (
    ROOT / "artifacts/jseb_navigation_repair_r3r4_uv_seed0_20260828_193904"
    / "R3/eval_navigation_tasks.json"
)
TASK_SHA256 = "6169ef6ec56e38d9c95260a1351bd40cfc8af0e62b6672dac93013494e00692d"
BUCKETS = ("100-500", "500-1500", "1500-2500", "2500-4000", ">4000")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_tasks(path: Path = TASKS) -> tuple[list[dict[str, object]], int]:
    if file_hash(path) != TASK_SHA256:
        raise ValueError("immutable stratified task file hash mismatch")
    payload = json.loads(path.read_text())
    by_bucket = {name: [] for name in BUCKETS}
    for source_index, raw in enumerate(payload["tasks"]):
        row = dict(raw)
        name = str(row["distance_bucket"])
        if name not in by_bucket:
            raise ValueError(f"unknown distance bucket: {name}")
        row["source_task_index"] = source_index
        by_bucket[name].append(row)
    if any(len(by_bucket[name]) != 100 for name in BUCKETS):
        raise ValueError("formal task set must contain exactly100 tasks per bucket")
    # Interleaving changes only execution order. World seeds remain tied to the
    # immutable source index, while every partial batch stays distance-balanced.
    tasks = [by_bucket[name][index] for index in range(100) for name in BUCKETS]
    return tasks, int(payload["seed"])


def stratified_evaluate(model, arm: Path, root: Path, args) -> bool:
    tasks, base_seed = load_tasks()
    if args.eval_tasks != len(tasks):
        raise ValueError("formal stratified evaluation requires exactly500 tasks")
    path = arm / "evaluation_stratified.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    if any(int(row["evaluation_index"]) != i for i, row in enumerate(rows)):
        raise ValueError("saved evaluation prefix is not canonical")
    vec = SubprocVecEnv(
        [partial(RecoveryCohort, horizon=args.horizon, obstacles=args.obstacles)
         for _ in range(args.num_envs)], start_method="forkserver"
    )
    try:
        while len(rows) < len(tasks) and not base.paused(root):
            count = min(args.num_envs, len(tasks) - len(rows))
            group: list[dict[str, object]] = []
            observations = []
            for worker in range(args.num_envs):
                evaluation_index = len(rows) + worker
                if worker < count:
                    task = tasks[evaluation_index]
                    observation, _ = vec.env_method(
                        "reset",
                        seed=base_seed + int(task["source_task_index"]),
                        options={
                            "start_position": task["start_position"],
                            "start_velocity": task["initial_velocity"],
                            "task_point": task["goal_position"],
                        },
                        indices=[worker],
                    )[0]
                else:
                    observation, _ = vec.env_method("reset", seed=base_seed, indices=[worker])[0]
                    vec.env_method("park", indices=[worker])
                observations.append(observation)
            obs = np.stack(observations)
            active = np.arange(args.num_envs) < count
            for _ in range(args.horizon):
                actions, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec.step(actions)
                for worker in np.flatnonzero(active & dones):
                    evaluation_index = len(rows) + int(worker)
                    task = tasks[evaluation_index]
                    row = dict(infos[worker]["navigation_episode"])
                    row.pop("outcome", None)
                    row.update(
                        evaluation_index=evaluation_index,
                        source_task_index=int(task["source_task_index"]),
                        distance_bucket=str(task["distance_bucket"]),
                        straight_line_distance=float(task["straight_line_distance"]),
                    )
                    group.append(row)
                    active[worker] = False
                if not active.any():
                    break
                if base.paused(root):
                    return False
            if active.any():
                raise RuntimeError("stratified task exceeded finite horizon")
            rows.extend(sorted(group, key=lambda row: int(row["evaluation_index"])))
            atomic_json(path, rows)
            atomic_json(root / "status.json", {
                "status": "EVALUATING_STRATIFIED", "arm": arm.name,
                "completed": len(rows), "target": len(tasks), "pid": os.getpid(),
            })
    finally:
        vec.close()
    if len(rows) != len(tasks):
        return False

    def summarize(subset: list[dict[str, object]]) -> dict[str, object]:
        successes = [row for row in subset if bool(row["goal_reached"])]
        return {
            "tasks": len(subset),
            "goal_reached": len(successes),
            "safe_goal": sum(bool(row["safe_goal"]) for row in subset),
            "timeouts": sum(row["termination"] == "deadline" for row in subset),
            "mean_collision_count": float(
                np.mean([int(row["collision_count"]) for row in subset])
            ),
            "mean_success_path_ratio": (
                float(np.mean([float(row["success_path_ratio"]) for row in successes]))
                if successes else None
            ),
            "mean_energy": float(np.mean([float(row["energy"]) for row in subset])),
        }

    summary = summarize(rows)
    summary.update({
        "protocol": "locked_recovery_five_distance_buckets_v1",
        "task_source": str(TASKS), "task_source_sha256": TASK_SHA256,
        "task_seed": base_seed, "deterministic_policy": True,
        "distance_buckets": {
            name: summarize([row for row in rows if row["distance_bucket"] == name])
            for name in BUCKETS
        },
        "collision_statistic": "one unified count per policy step",
    })
    atomic_json(arm / "summary.json", summary)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    contract = manifest["contract"]
    run_args = argparse.Namespace(**contract["arguments"])
    if run_args.eval_tasks != 500:
        raise ValueError("original run was not registered for500 tasks")
    if (root / "sac/evaluation.json").exists() or (root / "ppo/evaluation.json").exists():
        raise ValueError("legacy random evaluation already started; preserve it separately")
    signal.signal(signal.SIGTERM, base.request_pause)
    signal.signal(signal.SIGINT, base.request_pause)
    base.evaluate = stratified_evaluate
    atomic_json(root / "EVALUATION_PROTOCOL_MIGRATION.json", {
        "reason": "restore the prior immutable five-distance-bucket formal protocol",
        "training_contract_unchanged": True,
        "old_random_evaluation_started": False,
        "task_source": str(TASKS), "task_source_sha256": TASK_SHA256,
        "buckets": list(BUCKETS), "tasks_per_bucket": 100,
        "collision_protocol": "locked nonterminal recovery; unified contact count",
        "resume_script": str(Path(__file__).relative_to(ROOT)),
        "resume_script_sha256": file_hash(Path(__file__)),
        "migrated_unix": time.time(),
    })
    try:
        for algorithm in ("sac", "ppo"):
            if not base.train_arm(algorithm, root, run_args, contract):
                atomic_json(root / "status.json", {
                    "status": "PAUSED", "pid": os.getpid(), "arm": algorithm,
                })
                return
        atomic_json(root / "COMPLETE.json", {"completed_unix": time.time()})
        atomic_json(root / "status.json", {"status": "COMPLETE"})
    except Exception:
        atomic_json(root / "ERROR_STRATIFIED_RESUME.json", {
            "error": traceback.format_exc(), "time": time.time(),
        })
        raise


if __name__ == "__main__":
    main()
