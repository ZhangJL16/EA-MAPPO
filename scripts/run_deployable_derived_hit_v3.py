#!/usr/bin/env python3
"""Versioned derived-hit ablation; same training/checkpoint loop as v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from functools import partial

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.derived_hit_observation import (
    DIM, FIELDS, VERSION, DeployableRecovery, policy_kwargs,
)
from experiments.directional_navigation.standard_baselines import (
    atomic_json, load_checkpoint, save_checkpoint,
)
from scripts.run_recovery_sac_ppo_scratch import Records, count_updates

STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument("--start", action="store_true", help="Actually train; omitted means preparation only")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--battery-capacity", required=True, type=float)
    result.add_argument("--capacity-source", required=True)
    result.add_argument("--initial-soc", type=float, default=1.0)
    result.add_argument("--device", default="cuda")
    result.add_argument("--target-steps", type=int, default=524288)
    result.add_argument("--num-envs", type=int, default=8)
    result.add_argument("--rollout-steps", type=int, default=512)
    result.add_argument("--batch-size", type=int, default=256)
    result.add_argument("--buffer-size", type=int, default=200000)
    result.add_argument("--learning-starts", type=int, default=5000)
    result.add_argument("--checkpoint-steps", type=int, default=32768)
    result.add_argument("--horizon", type=int, default=4000)
    result.add_argument("--obstacles", type=int, default=24)
    return result


def validate(args: argparse.Namespace) -> None:
    block = args.num_envs * args.rollout_steps
    if min(args.num_envs, args.rollout_steps, args.batch_size, args.buffer_size,
           args.target_steps, args.checkpoint_steps, args.horizon) <= 0:
        raise ValueError("positive training sizes required")
    if args.target_steps % block or args.checkpoint_steps % block:
        raise ValueError("target/checkpoints must align with full training blocks")
    if args.obstacles < 0 or args.learning_starts < 0:
        raise ValueError("negative obstacle count or warmup")
    if not np.isfinite(args.battery_capacity) or args.battery_capacity <= 0:
        raise ValueError("explicit positive finite battery capacity required")
    if not np.isfinite(args.initial_soc) or not 0 <= args.initial_soc <= 1:
        raise ValueError("SOC must lie in [0,1]")
    if not args.capacity_source.strip():
        raise ValueError("battery capacity provenance required")
    if args.resume and not args.start:
        raise ValueError("resume also requires explicit --start")


def contract_for(args: argparse.Namespace) -> dict:
    sources = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/deployable_observation.py",
        ROOT / "experiments/directional_navigation/derived_hit_observation.py",
        ROOT / "experiments/directional_navigation/standard_baselines.py",
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "experiments/directional_navigation/environment.py",
        ROOT / "experiments/jacobian_energy_bridge/features.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
        ROOT / "scripts/run_recovery_sac_ppo_scratch.py",
    ]
    return {
        "protocol": VERSION,
        "arguments": {k: v for k, v in vars(args).items()
                      if k not in {"output_dir", "start", "resume"}},
        "observation_dim": DIM, "fields": FIELDS,
        "lidar_shape": [2, 8, 128],
        "ablation": "internal_hit_from_ranges_lt_one; all_v2_context_retained",
        "initialization": "v2_range_weights_preserved_new_hit_weights_zero_rng_preserved", "privileged_obstacle_input": False,
        "battery_source": "initial_budget_minus_realized_telemetry",
        "energy_unit": "synthetic_simulation_energy_units",
        "energy_exhaustion_terminates": False,
        "reward_and_physics": "unchanged_locked_recovery",
        "experiment_scope": "observation_only_navigation_not_energy_sustainability",
        "mission_sampling": "task_only; explicit return-mode reset supported but not trained here",
        "sb3": stable_baselines3.__version__, "torch": torch.__version__,
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sources},
    }


def make_model(vec, args: argparse.Namespace) -> SAC:
    return SAC(
        "MlpPolicy", vec, learning_rate=3e-4, gamma=0.99,
        batch_size=args.batch_size, seed=0, device=args.device, verbose=0,
        policy_kwargs=policy_kwargs(), buffer_size=args.buffer_size,
        learning_starts=args.learning_starts, tau=0.005,
        train_freq=1, gradient_steps=-1, ent_coef="auto_0.01", target_entropy=-3.0,
    )


def train(args: argparse.Namespace, root: Path, contract: dict) -> None:
    global STOP
    STOP = False
    arm = root / "sac"
    arm.mkdir(exist_ok=True)
    if (arm / "latest.json").exists() != args.resume:
        raise ValueError("use --resume exactly when a committed checkpoint exists")
    if (root / "PAUSE").exists():
        raise ValueError("PAUSE request still present; no training started")
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    vec = SubprocVecEnv([
        partial(DeployableRecovery, battery_capacity=args.battery_capacity,
                initial_soc=args.initial_soc, seed_start=993600001+i,
                seed_stride=args.num_envs, horizon=args.horizon, obstacles=args.obstacles)
        for i in range(args.num_envs)
    ], start_method="forkserver")
    hooks = []
    try:
        if args.resume:
            model, records = load_checkpoint("sac", vec, arm, contract, args.device)
        else:
            model = make_model(vec, args)
            vec._reset_seeds()
            records = {"episodes": [], "updates": [], "training_seconds": 0.0}
        atomic_json(arm / "training_records.json", records)
        callback = Records(records)
        hooks = count_updates(model, records)
        block = args.num_envs * args.rollout_steps
        latest = model.num_timesteps if args.resume else -1
        while model.num_timesteps < args.target_steps and not STOP and not (root / "PAUSE").exists():
            atomic_json(root / "status.json", {"status": "TRAINING", "pid": os.getpid(),
                        "transitions": model.num_timesteps, "target": args.target_steps})
            started = time.perf_counter()
            callback.collect_seconds = 0.0
            model.learn(block, reset_num_timesteps=False, callback=callback, log_interval=None)
            seconds = time.perf_counter() - started
            metrics = {k: float(v) for k, v in model.logger.name_to_value.items()
                       if k.startswith("train/") and isinstance(v, (int, float, np.number))}
            if not all(np.isfinite(v) for v in metrics.values()) or not all(
                torch.isfinite(p).all().item() for p in model.policy.parameters()
            ):
                raise FloatingPointError("nonfinite training state")
            records["training_seconds"] += seconds
            records["updates"].append({"transitions": model.num_timesteps,
                "block_seconds": seconds, "collection_seconds": callback.collect_seconds,
                "metrics": metrics})
            atomic_json(arm / "training_records.json", records)
            should_pause = STOP or (root / "PAUSE").exists()
            if (latest < 0 or model.num_timesteps == min(2 * block, args.target_steps)
                    or model.num_timesteps % args.checkpoint_steps == 0
                    or model.num_timesteps >= args.target_steps or should_pause):
                save_checkpoint(model, vec, arm, contract, records)
                latest = model.num_timesteps
            atomic_json(root / "status.json", {"status": "TRAINING", "pid": os.getpid(),
                "transitions": model.num_timesteps, "checkpoint_transitions": latest,
                "training_seconds": records["training_seconds"]})
        if model.num_timesteps != latest:
            save_checkpoint(model, vec, arm, contract, records)
        atomic_json(root / "status.json", {
            "status": "TRAINING_COMPLETE_AWAITING_USER" if model.num_timesteps >= args.target_steps else "PAUSED",
            "transitions": model.num_timesteps, "automatic_evaluation": False,
        })
    finally:
        for hook in hooks:
            hook.remove()
        vec.close()


def main() -> None:
    args = parser().parse_args()
    validate(args)
    root = args.output_dir.resolve()
    contract = contract_for(args)
    if root.exists():
        manifest = root / "manifest.json"
        if not manifest.exists() or json.loads(manifest.read_text())["contract"] != contract:
            raise ValueError("output directory is unrelated or prepared source/config changed")
    else:
        if args.resume:
            raise ValueError("cannot resume a missing run")
        root.mkdir(parents=True)
        atomic_json(root / "manifest.json", {"contract": contract, "prepared_unix": time.time()})
        atomic_json(root / "status.json", {"status": "READY_NOT_STARTED", "transitions": 0})
    if not args.start:
        print(json.dumps({"status": "PREPARATION_ONLY", "output_dir": str(root),
                          "training_started": False, "observation_dim": DIM}))
        return
    try:
        train(args, root, contract)
    except Exception:
        atomic_json(root / "ERROR.json", {"error": traceback.format_exc(), "time": time.time()})
        raise


if __name__ == "__main__":
    main()
