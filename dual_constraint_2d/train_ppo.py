"""Checkpointed PPO training on raw trial-and-error synthetic episodes.

The program uses the existing PyTorch/SB3 runtime and stores its exact package
versions in the manifest.  Interrupted continuation restores model and
optimizer but starts a new episode; it is resumable, not bitwise identical.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import platform

import gymnasium
import numpy
import stable_baselines3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import torch

from .calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from .gym_adapter import DualConstraintGym


ROOT = Path(__file__).resolve().parents[1]
TRAIN_MAP_IDS = tuple(range(8, 32))
VALIDATION_MAP_IDS = tuple(range(32, 40))
TEST_MAP_IDS = tuple(range(40, 56))
TRAIN_SEEDS = (101, 202, 303)
DEFAULT_TIMESTEPS = 51200
DEFAULT_CHECKPOINT_STEPS = 5120
SOURCE_FILES = (
    "dual_constraint_2d/action_adapter.py",
    "dual_constraint_2d/backup.py",
    "dual_constraint_2d/calibration.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/gym_adapter.py",
    "dual_constraint_2d/raw_training_env.py",
    "dual_constraint_2d/reference.py",
    "dual_constraint_2d/routes.py",
    "dual_constraint_2d/shield.py",
    "dual_constraint_2d/synthetic.py",
    "dual_constraint_2d/targets.py",
    "dual_constraint_2d/train_ppo.py",
    "nav3d/controller.py",
    "nav3d/geometry.py",
    "nav3d/simulation.py",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def training_manifest(seed: int, total_timesteps: int, n_steps: int, checkpoint_steps: int) -> dict:
    calibration_path = CALIBRATION_OUTPUT / "calibration.json"
    return {
        "protocol": "dual_constraint_2d_static_fullmap_ppo_v0",
        "seed": seed,
        "train_map_ids": list(TRAIN_MAP_IDS),
        "validation_map_ids": list(VALIDATION_MAP_IDS),
        "test_map_ids": list(TEST_MAP_IDS),
        "horizon_s": 600.0,
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "checkpoint_steps": checkpoint_steps,
        "ppo": {
            "policy": "MlpPolicy",
            "net_arch": [128, 128],
            "batch_size": min(128, n_steps),
            "n_epochs": 4,
            "learning_rate": 0.0003,
            "gamma": 0.995,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "device": "cpu",
        },
        "python": platform.python_version(),
        "packages": {
            "numpy": numpy.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
            "torch": torch.__version__,
        },
        "calibration_sha256": sha256(calibration_path.read_bytes()).hexdigest(),
        "source_sha256": {
            file: sha256((ROOT / file).read_bytes()).hexdigest() for file in SOURCE_FILES
        },
    }


class SaveProgress(BaseCallback):
    def __init__(self, output: Path, *, initial_episodes: int = 0) -> None:
        super().__init__()
        self.output = output
        self.episodes = initial_episodes

    def save_after_update(self) -> dict:
        count = self.model.num_timesteps
        path = self.output / f"model_{count:09d}.zip"
        temp = self.output / f"model_{count:09d}.tmp.zip"
        self.model.save(str(temp))
        os.replace(temp, path)
        status = {
            "timesteps": count,
            "episodes": self.episodes,
            "latest_model": path.name,
            "complete": False,
        }
        _atomic_json(self.output / "status.json", status)
        return status

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = self.locals.get("dones", ())
        for info, done in zip(infos, dones):
            if done:
                self.episodes += 1
                row = {
                    "episode": self.episodes,
                    "timesteps": self.model.num_timesteps,
                    "map_id": info.get("map_id"),
                    "completed_targets": info.get("completed_targets"),
                    "collision_count": info.get("collision_count"),
                    "safety_cost": info.get("safety_cost"),
                    "failure_reason": info.get("failure_reason"),
                    "charge_events": info.get("charge_events"),
                    "simulated_seconds": info.get("time_s"),
                }
                with (self.output / "episodes.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
        return True


def train(
    output: Path,
    *,
    seed: int,
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_steps: int = 512,
    checkpoint_steps: int = DEFAULT_CHECKPOINT_STEPS,
    max_new_timesteps: int | None = None,
) -> dict:
    if seed < 0 or total_timesteps <= 0 or n_steps <= 0 or checkpoint_steps <= 0:
        raise ValueError("training budget and seed must be valid")
    if total_timesteps % n_steps or checkpoint_steps % n_steps:
        raise ValueError("training and checkpoint budgets must be multiples of n_steps")
    if max_new_timesteps is not None and (
        max_new_timesteps <= 0 or max_new_timesteps % n_steps
    ):
        raise ValueError("max_new_timesteps must be a positive multiple of n_steps")
    output.mkdir(parents=True, exist_ok=True)
    manifest = training_manifest(seed, total_timesteps, n_steps, checkpoint_steps)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("training manifest changed; refusing mixed results")
    else:
        _atomic_json(manifest_path, manifest)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else None
    if status and status.get("complete"):
        return status
    calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text())
    env = DualConstraintGym(
        TRAIN_MAP_IDS,
        calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"],
        shielded=False,
        map_seed=seed,
    )
    if status is None:
        model = PPO(
            "MlpPolicy", env, seed=seed, verbose=0, device="cpu",
            policy_kwargs={"net_arch": [128, 128]},
            n_steps=n_steps, batch_size=min(128, n_steps), n_epochs=4,
            learning_rate=0.0003, gamma=0.995, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01,
        )
        episodes = 0
    else:
        model = PPO.load(str(output / status["latest_model"]), env=env, device="cpu")
        episodes = int(status["episodes"])
        if model.num_timesteps != status["timesteps"]:
            raise RuntimeError("saved training model and status disagree")
    remaining = total_timesteps - model.num_timesteps
    if remaining <= 0:
        result = {
            "timesteps": model.num_timesteps, "episodes": episodes,
            "latest_model": status["latest_model"], "complete": True,
        }
        _atomic_json(status_path, result)
        return result
    callback = SaveProgress(output, initial_episodes=episodes)
    call_limit = total_timesteps if max_new_timesteps is None else min(
        total_timesteps, model.num_timesteps + max_new_timesteps
    )
    while model.num_timesteps < call_limit:
        chunk = min(checkpoint_steps, call_limit - model.num_timesteps)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, callback=callback)
        callback.model = model
        callback.save_after_update()
    result = {
        "timesteps": model.num_timesteps,
        "episodes": callback.episodes,
        "latest_model": f"model_{model.num_timesteps:09d}.zip",
        "complete": model.num_timesteps >= total_timesteps,
    }
    _atomic_json(status_path, result)
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--checkpoint-steps", type=int, default=DEFAULT_CHECKPOINT_STEPS)
    parser.add_argument("--max-new-timesteps", type=int, default=None)
    args = parser.parse_args()
    print(json.dumps(train(
        args.output,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        checkpoint_steps=args.checkpoint_steps,
        max_new_timesteps=args.max_new_timesteps,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
