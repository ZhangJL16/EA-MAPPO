#!/usr/bin/env python3
"""Frozen correction-SAC evaluation: paired raw/HOCBF, immutable fixed500."""
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

from experiments.directional_navigation.correction_supervision import (
    CorrectionRecovery, OBSERVATION_DIM, VERSION,
)
from experiments.directional_navigation.standard_baselines import atomic_json
from scripts.evaluate_deployable_observation_v2 import summarize, validate_prefix
from scripts.resume_recovery_sac_ppo_stratified import (
    BUCKETS, TASKS, TASK_SHA256, file_hash, load_tasks,
)

STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


class ParkedCorrectionEvaluation(CorrectionRecovery):
    def reset(self, *, seed=None, options=None):
        if seed is None and self.park_after_terminal:
            self.parked = True
            return self.parked_observation.copy(), {}
        return super().reset(seed=seed, options=options)


def evaluate(model, root, args, tasks, base_seed, hocbf):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evaluation_stratified.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    validate_prefix(rows, tasks, base_seed)
    if len(rows) == len(tasks):
        return True
    workers = min(args.num_envs, len(tasks) - len(rows))
    vec = SubprocVecEnv([
        partial(ParkedCorrectionEvaluation, seed_start=1, seed_stride=1,
                horizon=args.horizon, obstacles=args.obstacles, hocbf=hocbf)
        for _ in range(workers)
    ], start_method="forkserver")

    def stopped():
        return STOP or (root / "PAUSE").exists() or (root.parent / "PAUSE").exists()

    try:
        while len(rows) < len(tasks) and not stopped():
            count = min(workers, len(tasks) - len(rows))
            observations, group = [], []
            for worker in range(workers):
                task = tasks[len(rows) + worker] if worker < count else tasks[0]
                obs, _ = vec.env_method(
                    "reset", seed=base_seed + int(task["source_task_index"]),
                    options={"start_position": task["start_position"],
                             "start_velocity": task["initial_velocity"],
                             "task_point": task["goal_position"]}, indices=[worker],
                )[0]
                if worker >= count:
                    vec.env_method("park", indices=[worker])
                observations.append(obs)
            obs = np.stack(observations)
            active = np.arange(workers) < count
            if obs.shape != (workers, OBSERVATION_DIM) or not np.isfinite(obs).all():
                raise ValueError("invalid legacy2056 evaluation observation")
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
                    row.update(evaluation_index=index,
                               source_task_index=int(task["source_task_index"]),
                               distance_bucket=task["distance_bucket"],
                               straight_line_distance=float(task["straight_line_distance"]),
                               hocbf=bool(hocbf))
                    group.append(row)
                    active[worker] = False
                if not active.any():
                    break
                if stopped():
                    return False  # Replay only the uncommitted batch on resume.
            if active.any():
                raise RuntimeError("evaluation exceeded fixed horizon")
            rows.extend(sorted(group, key=lambda row: row["evaluation_index"]))
            validate_prefix(rows, tasks, base_seed)
            atomic_json(path, rows)
            atomic_json(root / "status.json", {
                "status": "EVALUATING", "pid": os.getpid(), "completed": len(rows),
                "target": len(tasks), "updated_unix": time.time(),
            })
    finally:
        vec.close()
    return len(rows) == len(tasks)


def validated_checkpoint(source, steps, smoke=False):
    manifest = json.loads((source / "manifest.json").read_text())
    checkpoint = source / "sac" / f"checkpoint_{steps:09d}"
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    training = metadata["contract"]
    if (training != manifest["contract"] or training["protocol"] != VERSION
            or training["observation_dim"] != OBSERVATION_DIM
            or metadata["transitions"] != steps):
        raise ValueError("wrong correction-SAC checkpoint contract")
    if not smoke:
        status = json.loads((source / "status.json").read_text())
        if (status["status"] != "TRAINING_COMPLETE_AWAITING_USER"
                or status["checkpoint_transitions"] != steps
                or status["adaptation_transitions"] != steps
                or manifest["planned_additional_steps"] != steps):
            raise ValueError("formal evaluation requires the completed planned checkpoint")
    if any(file_hash(ROOT / p) != h for p, h in training["source_hashes"].items()):
        raise ValueError("training source changed; explicit audit required")
    if (checkpoint / "model.zip").stat().st_size != metadata["files"]["model.zip"]:
        raise ValueError("incomplete model checkpoint")
    return checkpoint, training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-transitions", type=int, default=131072)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="5 tasks x 2 steps per arm; NOT evidence")
    args = parser.parse_args()
    if args.num_envs <= 0 or args.checkpoint_transitions <= 0:
        raise ValueError("positive worker count/checkpoint required")
    source, root = args.source.resolve(), args.output_dir.resolve()
    if root == source or source in root.parents:
        raise ValueError("evaluation output must be separate from training")
    checkpoint, training = validated_checkpoint(source, args.checkpoint_transitions, args.smoke)
    tasks, base_seed = load_tasks()
    if args.smoke:
        tasks = tasks[:5]
    args.horizon = 2 if args.smoke else training["arguments"]["horizon"]
    args.obstacles = training["arguments"]["obstacles"]
    contract = {
        "protocol": "hocbf_correction_paired_fixed500_v1", "formal": not args.smoke,
        "task_source": str(TASKS), "task_sha256": TASK_SHA256, "task_seed": base_seed,
        "unique_tasks": len(tasks), "arms": ["raw", "hocbf"], "total_rollouts": 2 * len(tasks),
        "checkpoint": str(checkpoint), "model_sha256": file_hash(checkpoint / "model.zip"),
        "metadata_sha256": file_hash(checkpoint / "metadata.json"),
        "source_hashes": {p: file_hash(ROOT / p) for p in (
            "scripts/evaluate_hocbf_correction_fixed500.py",
            "scripts/evaluate_deployable_observation_v2.py",
            "scripts/resume_recovery_sac_ppo_stratified.py")},
        "observation_dim": OBSERVATION_DIM, "horizon": args.horizon,
        "obstacles": args.obstacles, "device": args.device, "num_envs": args.num_envs,
        "deterministic_policy": True, "train_updates": False,
        "collision_contract": training["collision_contract"],
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
        # Inference only: ordinary SAC loads the same policy tensors; no teacher/replay needed.
        model = SAC.load(checkpoint / "model.zip", device=args.device)
        if model.observation_space.shape != (OBSERVATION_DIM,):
            raise ValueError("wrong model observation shape")
        model.policy.set_training_mode(False)
        summaries = {}
        for arm in contract["arms"]:
            atomic_json(root / "status.json", {"status": "EVALUATING", "pid": os.getpid(), "arm": arm})
            complete = evaluate(model, root / arm, args, tasks, base_seed, hocbf=arm == "hocbf")
            if not complete:
                atomic_json(root / "status.json", {"status": "PAUSED", "arm": arm})
                return
            rows = json.loads((root / arm / "evaluation_stratified.json").read_text())
            summary = {**summarize(rows),
                       "distance_buckets": {b: summarize([r for r in rows if r["distance_bucket"] == b])
                                            for b in BUCKETS},
                       "hocbf_intervention_steps": sum(r["hocbf_intervention_steps"] for r in rows),
                       "hocbf_fallback_steps": sum(r["hocbf_fallback_steps"] for r in rows),
                       "hocbf_emergency_steps": sum(r["hocbf_emergency_steps"] for r in rows)}
            atomic_json(root / arm / "summary.json", summary)
            atomic_json(root / arm / "status.json", {"status": "COMPLETE", "completed": len(rows)})
            summaries[arm] = summary
        atomic_json(root / "summary.json", {"contract": contract, "arms": summaries})
        atomic_json(root / "status.json", {"status": "COMPLETE", "updated_unix": time.time(),
                                           "automatic_energy_training": False})
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
