"""Resumable plain-PPO learnability pilot; not constrained RL or energy training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import cast

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.environment import FirstContactNavigation
from experiments.directional_navigation.features import DirectionalLidarExtractor

PAUSED = False


def request_pause(signum, frame):
    global PAUSED
    PAUSED = True


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value) + "\n")


class EpisodeLog(BaseCallback):
    def __init__(self, output: Path):
        super().__init__()
        self.output = output
        self.collect_seconds = 0.0

    def _on_rollout_start(self):
        self.started = time.perf_counter()

    def _on_rollout_end(self):
        self.collect_seconds = time.perf_counter() - self.started

    def _on_step(self) -> bool:
        if (
            not np.isfinite(self.locals["rewards"]).all()
            or not np.isfinite(self.locals["actions"]).all()
        ):
            raise FloatingPointError("nonfinite rollout")
        for info in self.locals["infos"]:
            if "navigation_episode" in info:
                append_json(
                    self.output / "training_episodes.jsonl",
                    dict(transitions=self.num_timesteps, **info["navigation_episode"]),
                )
        return True  # Pause only after a complete PPO update, never discard a rollout.


def save_checkpoint(
    model: PPO, vec: SubprocVecEnv, output: Path, contract: dict
) -> Path:
    destination = output / f"checkpoint_{model.num_timesteps:09d}"
    if destination.exists():
        return destination  # Only called again for the same completed update.
    temporary = output / f".checkpoint_{model.num_timesteps:09d}.tmp"
    temporary.mkdir(exist_ok=False)
    model.save(temporary / "model.zip")  # SB3 includes policy optimizer state.
    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary / "rng.pt",
    )
    atomic_json(
        temporary / "state.json",
        {
            "contract": contract,
            "transitions": model.num_timesteps,
            "next_episode_indices": vec.env_method("episode_cursor"),
            "resume_semantics": "optimizer/RNG restored; unfinished tasks abandoned, next keyed task starts; not bitwise simulation resume",
        },
    )
    os.replace(temporary, destination)
    atomic_json(
        output / "latest.json",
        {"checkpoint": destination.name, "transitions": model.num_timesteps},
    )
    return destination


def evaluate(model: PPO, vec: SubprocVecEnv, output: Path, count: int) -> dict:
    path = output / f"evaluation_{model.num_timesteps:09d}.jsonl"
    records = (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )
    seen = {row["seed"] for row in records}
    pending = [293800001 + i for i in range(count) if 293800001 + i not in seen]
    active = {}
    obs = cast(np.ndarray, vec.reset())

    def assign(worker: int) -> None:
        if pending:
            seed = pending.pop(0)
            obs[worker] = vec.env_method("reset", seed=seed, indices=[worker])[0][0]
            active[worker] = seed

    for worker in range(vec.num_envs):
        assign(worker)
    while active and not PAUSED and not (output / "PAUSE").exists():
        actions, _ = model.predict(obs, deterministic=True)
        next_obs, _, dones, infos = vec.step(actions)
        assert isinstance(next_obs, np.ndarray)
        obs = next_obs
        for worker in list(active):
            if dones[worker]:
                row = dict(
                    infos[worker]["navigation_episode"], transitions=model.num_timesteps
                )
                if row["seed"] != active[worker]:
                    raise AssertionError("evaluation seed mismatch")
                records.append(row)
                append_json(path, row)
                del active[worker]
                assign(worker)
    successful = [
        r["success_path_ratio"] for r in records if r["outcome"] == "safe_goal"
    ]
    return {
        "completed": len(records),
        "requested": count,
        "counts": {
            k: sum(r["outcome"] == k for r in records)
            for k in ("safe_goal", "contact", "timeout")
        },
        "success_path_ratio_mean": float(np.mean(successful)) if successful else None,
        "success_path_ratio_count": len(successful),
        "formal_500": False,
        "complete": len(records) == count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--target-steps", type=int, default=262144)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--rollout-steps", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=4000)
    parser.add_argument("--obstacles", type=int, default=24)
    parser.add_argument("--evaluation-tasks", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    chunk = args.num_envs * args.rollout_steps
    if (
        min(
            args.num_envs,
            args.rollout_steps,
            args.epochs,
            args.batch_size,
            args.target_steps,
        )
        <= 0
    ):
        raise ValueError("training sizes must be positive")
    if args.target_steps % chunk or chunk % args.batch_size or args.batch_size < 2:
        raise ValueError("target and minibatches must align with full rollouts")
    if args.evaluation_tasks < 0:
        raise ValueError("negative evaluation count")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, request_pause)
    signal.signal(signal.SIGINT, request_pause)
    sources = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/features.py",
        ROOT / "experiments/directional_navigation/environment.py",
        ROOT / "experiments/jacobian_energy_bridge/features.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
    ]
    contract = {
        "protocol": "directional_plain_ppo_first_contact_v1",
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "obstacles": args.obstacles,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "learning_rate": 0.0003,
        "reward_scale": 0.01,
        "seed": 193500001,
        "train_seed_start": 193600001,
        "eval_seed_start": 293800001,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
        "action_distribution": "SB3 latent Gaussian; clip then physical radial bounds; likelihood uses pre-clip sample",
        "safety_constraint": False,
        "energy_training": False,
        "shield": False,
    }
    env_fns: list[Callable[[], gym.Env]] = [
        partial(
            FirstContactNavigation,
            horizon=args.horizon,
            obstacles=args.obstacles,
            seed_start=193600001 + i,
            seed_stride=args.num_envs,
        )
        for i in range(args.num_envs)
    ]
    vec = SubprocVecEnv(env_fns, start_method="forkserver")
    started = time.perf_counter()
    try:
        if args.resume:
            latest = json.loads((output / "latest.json").read_text())
            checkpoint = output / latest["checkpoint"]
            state = json.loads((checkpoint / "state.json").read_text())
            if state["contract"] != contract:
                raise ValueError(
                    "resume contract/source mismatch; use a separately versioned run"
                )
            model = PPO.load(checkpoint / "model.zip", env=vec, device=args.device)
            for worker, cursor in enumerate(state["next_episode_indices"]):
                vec.env_method("restore_episode_cursor", cursor, indices=[worker])
            rng = torch.load(
                checkpoint / "rng.pt", map_location="cpu", weights_only=False
            )
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"])
            if args.device.startswith("cuda") and rng["cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            model = PPO(
                "MlpPolicy",
                vec,
                device=args.device,
                seed=contract["seed"],
                n_steps=args.rollout_steps,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
                gamma=1.0,
                gae_lambda=0.95,
                learning_rate=0.0003,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.02,
                policy_kwargs={
                    "features_extractor_class": DirectionalLidarExtractor,
                    "features_extractor_kwargs": {"remaining_time": True},
                    "share_features_extractor": False,
                    "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                    "log_std_init": float(np.log(0.5)),
                },
                verbose=0,
            )
            atomic_json(
                output / "manifest.json",
                {
                    "contract": contract,
                    "command": sys.argv,
                    "parameters": sum(p.numel() for p in model.policy.parameters()),
                    "extractor": cast(
                        DirectionalLidarExtractor, model.policy.pi_features_extractor
                    ).architecture_audit(),
                    "note": "Navigation sanity only. No automatic safety-algorithm or 500k promotion.",
                },
            )
        # SB3 seeds VecEnv during model setup/load. Our wrapper owns the keyed
        # task schedule, including after resume; do not replay SB3's first seed.
        vec._reset_seeds()
        callback = EpisodeLog(output)
        if not args.resume:
            save_checkpoint(model, vec, output, contract)
        session_start_steps = model.num_timesteps
        atomic_json(
            output / "status.json",
            {
                "status": "TRAINING",
                "pid": os.getpid(),
                "target_steps": args.target_steps,
                "transitions": model.num_timesteps,
                "started_unix": time.time(),
            },
        )
        while (
            model.num_timesteps < args.target_steps
            and not PAUSED
            and not (output / "PAUSE").exists()
        ):
            begin = time.perf_counter()
            model.learn(
                total_timesteps=chunk, reset_num_timesteps=False, callback=callback
            )
            elapsed = time.perf_counter() - begin
            if not all(
                torch.isfinite(p).all().item() for p in model.policy.parameters()
            ):
                raise FloatingPointError("nonfinite updated model")
            metrics = {
                k: float(v)
                for k, v in model.logger.name_to_value.items()
                if k.startswith("train/")
                and isinstance(v, (int, float, np.integer, np.floating))
            }
            if not all(np.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("nonfinite training metrics")
            record = dict(
                transitions=model.num_timesteps,
                chunk_seconds=elapsed,
                collection_seconds=callback.collect_seconds,
                update_and_setup_seconds=elapsed - callback.collect_seconds,
                session_seconds=time.perf_counter() - started,
                **metrics,
            )
            append_json(output / "updates.jsonl", record)
            print(json.dumps(record), flush=True)
            if model.num_timesteps % (4 * chunk) == 0:
                save_checkpoint(model, vec, output, contract)
            atomic_json(
                output / "status.json",
                dict(
                    status="TRAINING",
                    pid=os.getpid(),
                    target_steps=args.target_steps,
                    **record,
                ),
            )
        save_checkpoint(model, vec, output, contract)
        result = {
            "transitions": model.num_timesteps,
            "target_steps": args.target_steps,
            "session_transitions": model.num_timesteps - session_start_steps,
            "session_seconds": time.perf_counter() - started,
        }
        if PAUSED or (output / "PAUSE").exists():
            result["status"] = "PAUSED"
        else:
            atomic_json(
                output / "status.json",
                dict(status="EVALUATING", pid=os.getpid(), **result),
            )
            result["evaluation"] = evaluate(model, vec, output, args.evaluation_tasks)
            result["status"] = (
                "COMPLETE" if result["evaluation"]["complete"] else "EVALUATION_PAUSED"
            )
        result["session_seconds"] = time.perf_counter() - started
        atomic_json(output / "status.json", result)
        atomic_json(output / "RESULT.json", result)
    except BaseException as exc:
        atomic_json(
            output / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    main()
