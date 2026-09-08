#!/usr/bin/env python3
"""Preplanned native SAC/PPO comparison; no performance-triggered promotion."""

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
import stable_baselines3
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.standard_baselines import (
    ContinuousRecovery,
    atomic_json,
    load_checkpoint,
    save_checkpoint,
)

PAUSED = False


def request_pause(signum, frame):
    global PAUSED
    PAUSED = True


def paused(root: Path) -> bool:
    return PAUSED or (root / "PAUSE").exists()


class Records(BaseCallback):
    def __init__(self, records: dict):
        super().__init__()
        self.records = records
        self.collect_seconds = 0.0

    def _on_rollout_start(self):
        self.started = time.perf_counter()

    def _on_rollout_end(self):
        self.collect_seconds += time.perf_counter() - self.started

    def _on_step(self) -> bool:
        for key in ("actions", "rewards", "new_obs"):
            if not np.isfinite(self.locals[key]).all():
                raise FloatingPointError(f"nonfinite {key}")
        for info in self.locals["infos"]:
            if "navigation_episode" in info:
                row = dict(info["navigation_episode"])
                row.pop("outcome", None)
                row["transitions"] = self.num_timesteps
                self.records["episodes"].append(row)
        return True


def count_updates(model, records: dict) -> list:
    handles = []
    if isinstance(model, SAC):
        pairs = [(model.actor.optimizer, "actor_updates"),
                 (model.critic.optimizer, "critic_updates")]
    else:
        pairs = [(model.policy.optimizer, "joint_updates")]
    for optimizer, key in pairs:
        def counted(_optimizer, _args, _kwargs, key=key):
            records[key] = records.get(key, 0) + 1
        handles.append(optimizer.register_step_post_hook(counted))
    return handles


def make_model(algorithm: str, vec, args):
    policy_kwargs = {
        "features_extractor_class": DirectionalLidarExtractor,
        "features_extractor_kwargs": {"remaining_time": True},
        "share_features_extractor": False,
    }
    common = dict(policy="MlpPolicy", env=vec, learning_rate=3e-4,
                  gamma=0.99, batch_size=args.batch_size, seed=0,
                  device=args.device, verbose=0, policy_kwargs=policy_kwargs)
    if algorithm == "sac":
        policy_kwargs["net_arch"] = {"pi": [256, 256], "qf": [256, 256]}
        return SAC(**common, buffer_size=args.buffer_size,
                   learning_starts=args.learning_starts, tau=0.005,
                   train_freq=1, gradient_steps=-1, ent_coef="auto_0.01",
                   target_entropy=-3.0)
    policy_kwargs.update(net_arch={"pi": [256, 256], "vf": [256, 256]},
                         log_std_init=float(np.log(0.5)))
    return PPO(**common, n_steps=args.rollout_steps, n_epochs=10,
               gae_lambda=0.95, clip_range=0.2, ent_coef=0.0, vf_coef=0.5,
               max_grad_norm=0.5, target_kl=0.02)


def evaluate(model, arm: Path, root: Path, args) -> bool:
    path = arm / "evaluation.json"
    rows = json.loads(path.read_text()) if path.exists() else []
    vec = SubprocVecEnv([partial(RecoveryCohort, horizon=args.horizon,
                                  obstacles=args.obstacles)
                        for _ in range(args.num_envs)], start_method="forkserver")
    try:
        while len(rows) < args.eval_tasks and not paused(root):
            count = min(args.num_envs, args.eval_tasks - len(rows))
            observations = []
            for i in range(args.num_envs):
                observation, _ = vec.env_method(
                    "reset", seed=993800001 + len(rows) + i, indices=[i])[0]
                observations.append(observation)
                if i >= count:
                    vec.env_method("park", indices=[i])
            obs = np.stack(observations)
            active = np.arange(args.num_envs) < count
            group = []
            for _ in range(args.horizon):
                actions, _ = model.predict(obs, deterministic=True)
                obs, _, dones, infos = vec.step(actions)
                for i in np.flatnonzero(active & dones):
                    row = dict(infos[i]["navigation_episode"])
                    row.pop("outcome", None)
                    group.append(row)
                    active[i] = False
                if not active.any():
                    break
                if paused(root):
                    return False  # Uncommitted group is replayed on resume.
            if active.any():
                raise RuntimeError("evaluation exceeded finite task horizon")
            rows.extend(sorted(group, key=lambda r: r["seed"]))
            atomic_json(path, rows)
            atomic_json(root / "status.json", dict(status="EVALUATING", arm=arm.name,
                        completed=len(rows), target=args.eval_tasks, pid=os.getpid()))
    finally:
        vec.close()
    if len(rows) != args.eval_tasks:
        return False
    successes = [r for r in rows if r["goal_reached"]]
    atomic_json(arm / "summary.json", {
        "tasks": len(rows), "goal_reached": len(successes),
        "safe_goal": sum(r["safe_goal"] for r in rows),
        "timeouts": sum(r["termination"] == "deadline" for r in rows),
        "mean_collision_count": float(np.mean([r["collision_count"] for r in rows])) if rows else None,
        "mean_success_path_ratio": float(np.mean([r["success_path_ratio"] for r in successes])) if successes else None,
        "mean_energy": float(np.mean([r["energy"] for r in rows])) if rows else None,
    })
    return True


def train_arm(algorithm: str, root: Path, args, contract: dict) -> bool:
    arm = root / algorithm
    arm.mkdir(exist_ok=True)
    if (arm / "COMPLETE.json").exists():
        return True
    vec = SubprocVecEnv([
        partial(ContinuousRecovery, seed_start=993600001+i,
                seed_stride=args.num_envs, horizon=args.horizon,
                obstacles=args.obstacles) for i in range(args.num_envs)
    ], start_method="forkserver")
    try:
        if (arm / "latest.json").exists():
            model, records = load_checkpoint(algorithm, vec, arm, contract, args.device)
        else:
            model = make_model(algorithm, vec, args)
            vec._reset_seeds()
            records = {"episodes": [], "updates": [], "training_seconds": 0.0}
        # Rebuild logs from the committed state, discarding crash-only rows.
        atomic_json(arm / "training_records.json", records)
        hooks = count_updates(model, records)
        block = args.num_envs * args.rollout_steps
        callback = Records(records)
        while model.num_timesteps < args.target_steps and not paused(root):
            atomic_json(root / "status.json", dict(
                status="TRAINING", arm=algorithm, transitions=model.num_timesteps,
                target=args.target_steps, pid=os.getpid(), updated_unix=time.time()))
            begin = time.perf_counter()
            callback.collect_seconds = 0.0
            model.learn(total_timesteps=block, reset_num_timesteps=False,
                        callback=callback, log_interval=None)
            seconds = time.perf_counter() - begin
            metrics = {k: float(v) for k, v in model.logger.name_to_value.items()
                       if k.startswith("train/") and isinstance(v, (int, float, np.number))}
            if not all(np.isfinite(v) for v in metrics.values()) or not all(
                torch.isfinite(p).all().item() for p in model.policy.parameters()
            ):
                raise FloatingPointError("nonfinite training state")
            records["training_seconds"] += seconds
            record = dict(transitions=model.num_timesteps, block_seconds=seconds,
                          collection_seconds=callback.collect_seconds,
                          optimization_and_setup_seconds=seconds-callback.collect_seconds,
                          metrics=metrics,
                          optimizer_steps={k: v for k, v in records.items() if k.endswith("_updates")})
            records["updates"].append(record)
            atomic_json(arm / "training_records.json", records)
            print(json.dumps({"arm": algorithm, **record}), flush=True)
            if (model.num_timesteps == min(2*block, args.target_steps)
                or model.num_timesteps % args.checkpoint_steps == 0
                or model.num_timesteps >= args.target_steps or paused(root)):
                save_checkpoint(model, vec, arm, contract, records)
            atomic_json(root / "status.json", dict(
                status="TRAINING", arm=algorithm, pid=os.getpid(), **record,
                updated_unix=time.time()))
        # A pause between blocks also needs the most recent completed update.
        last = json.loads((arm / "latest.json").read_text())["checkpoint"] if (arm / "latest.json").exists() else None
        if last != f"checkpoint_{model.num_timesteps:09d}":
            save_checkpoint(model, vec, arm, contract, records)
        for hook in hooks:
            hook.remove()
    finally:
        vec.close()
    if paused(root) or not evaluate(model, arm, root, args):
        return False
    atomic_json(arm / "COMPLETE.json", dict(transitions=model.num_timesteps,
                training_seconds=records["training_seconds"], completed_unix=time.time()))
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-steps", type=int, default=524288)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--checkpoint-steps", type=int, default=32768)
    parser.add_argument("--horizon", type=int, default=4000)
    parser.add_argument("--obstacles", type=int, default=24)
    parser.add_argument("--eval-tasks", type=int, default=500)
    args = parser.parse_args()
    block = args.num_envs * args.rollout_steps
    if (min(block, args.batch_size, args.target_steps, args.horizon) <= 0
        or args.target_steps % block or args.checkpoint_steps % block
        or block % args.batch_size or args.eval_tasks < 0):
        raise ValueError("invalid size or incomplete rollout boundary")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    paths = [Path(__file__),
             ROOT / "experiments/directional_navigation/standard_baselines.py",
             ROOT / "experiments/directional_navigation/recovery.py",
             ROOT / "experiments/directional_navigation/environment.py",
             ROOT / "experiments/directional_navigation/features.py",
             ROOT / "experiments/jacobian_energy_bridge/features.py",
             ROOT / "envs/UAVEnergyDeliverySAC.py"]
    contract = {
        "protocol": "recovery_sac_ppo_scratch_v1",
        "arguments": {k: v for k, v in vars(args).items() if k not in {"resume", "output_dir"}},
        "sb3": stable_baselines3.__version__, "torch": torch.__version__,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("resume source/argument contract mismatch")
    else:
        atomic_json(root / "manifest.json", dict(contract=contract, command=sys.argv,
                    started_unix=time.time(), order=["sac", "ppo"],
                    protocol="docs/RECOVERY_SAC_PPO_SCRATCH_PROTOCOL.md"))
    try:
        for algorithm in ("sac", "ppo"):
            if not train_arm(algorithm, root, args, contract):
                atomic_json(root / "status.json", dict(status="PAUSED", pid=os.getpid(), arm=algorithm))
                return
        atomic_json(root / "COMPLETE.json", dict(completed_unix=time.time()))
        atomic_json(root / "status.json", dict(status="COMPLETE"))
    except Exception:
        atomic_json(root / "ERROR.json", dict(error=traceback.format_exc(), time=time.time()))
        raise


if __name__ == "__main__":
    main()
