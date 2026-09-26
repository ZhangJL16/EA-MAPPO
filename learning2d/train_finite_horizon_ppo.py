"""Resumable PPO for the physical 600-second task horizon.

Two predeclared discounts share one implementation: gamma=1 is the primary
undiscounted finite-window objective; gamma=0.995 isolates the horizon-terminal
repair while retaining the earlier per-decision discount.  Neither condition
changes the physical plant, collision semantics, maps, or action mask.
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
import torch

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.train_ppo import (
    ROOT,
    SOURCE_FILES,
    TRAIN_MAP_IDS,
    VALIDATION_MAP_IDS,
    SaveProgress,
)
from .finite_horizon_gym import FiniteHorizonGym
from .mode_policy import ModeMaskedPolicy


DEFAULT_TIMESTEPS = 51200
DEFAULT_CHECKPOINT_STEPS = 512
SOURCE_SET = SOURCE_FILES + (
    "learning2d/finite_horizon_gym.py",
    "learning2d/mode_policy.py",
    "learning2d/train_finite_horizon_ppo.py",
)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _selected_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu, or cuda")
    return device


def training_manifest(seed: int, total_timesteps: int, n_steps: int,
                      checkpoint_steps: int, gamma: float, device: str) -> dict:
    calibration_path = CALIBRATION_OUTPUT / "calibration.json"
    return {
        "protocol": "dual_constraint_2d_mode_masked_finite_horizon_v3",
        "seed": seed,
        "train_map_ids": list(TRAIN_MAP_IDS),
        "validation_map_ids": list(VALIDATION_MAP_IDS),
        "holdout_map_ids": list(range(56, 72)),
        "horizon_s": 600.0,
        "horizon_endpoint": "terminated_no_bootstrap",
        "total_timesteps": total_timesteps,
        "n_steps": n_steps,
        "checkpoint_steps": checkpoint_steps,
        "policy": "ModeMaskedPolicy",
        "action_support": "flight 0-9; dock 4 or strictly increasing charge target",
        "ppo": {
            "net_arch": [128, 128],
            "batch_size": min(128, n_steps),
            "n_epochs": 4,
            "learning_rate": 0.0003,
            "gamma": gamma,
            "discount_basis": "per_decision; gamma=1 avoids variable-duration discount mismatch",
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "device": device,
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
            file: sha256((ROOT / file).read_bytes()).hexdigest() for file in SOURCE_SET
        },
    }


def train(output: Path, *, seed: int, gamma: float = 1.0,
          total_timesteps: int = DEFAULT_TIMESTEPS, n_steps: int = 512,
          checkpoint_steps: int = DEFAULT_CHECKPOINT_STEPS,
          max_new_timesteps: int | None = None, device: str = "auto") -> dict:
    if gamma not in (1.0, 0.995):
        raise ValueError("gamma must be one of the two predeclared conditions")
    if seed < 0 or min(total_timesteps, n_steps, checkpoint_steps) <= 0:
        raise ValueError("seed and budgets must be valid")
    if total_timesteps % n_steps or checkpoint_steps % n_steps:
        raise ValueError("training/checkpoint budgets must be multiples of n_steps")
    if max_new_timesteps is not None and (
        max_new_timesteps <= 0 or max_new_timesteps % n_steps
    ):
        raise ValueError("max_new_timesteps must be a positive multiple of n_steps")
    chosen_device = _selected_device(device)
    output.mkdir(parents=True, exist_ok=True)
    manifest = training_manifest(seed, total_timesteps, n_steps,
                                 checkpoint_steps, gamma, chosen_device)
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
    env = FiniteHorizonGym(
        TRAIN_MAP_IDS,
        calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"],
        shielded=False,
        map_seed=seed,
    )
    try:
        if status is None:
            model = PPO(
                ModeMaskedPolicy, env, seed=seed, verbose=0, device=chosen_device,
                policy_kwargs={"net_arch": [128, 128]},
                n_steps=n_steps, batch_size=min(128, n_steps), n_epochs=4,
                learning_rate=0.0003, gamma=gamma, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.01,
            )
            episodes = 0
        else:
            model = PPO.load(str(output / status["latest_model"]), env=env,
                             device=chosen_device)
            episodes = int(status["episodes"])
            if model.num_timesteps != status["timesteps"] or model.gamma != gamma:
                raise RuntimeError("saved model disagrees with status or discount")
        if model.num_timesteps >= total_timesteps:
            result = {**status, "complete": True}
            _atomic_json(status_path, result)
            return result
        callback = SaveProgress(output, initial_episodes=episodes)
        limit = total_timesteps if max_new_timesteps is None else min(
            total_timesteps, model.num_timesteps + max_new_timesteps
        )
        while model.num_timesteps < limit:
            chunk = min(checkpoint_steps, limit - model.num_timesteps)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False,
                        callback=callback)
            callback.model = model
            callback.save_after_update()
        result = {
            "timesteps": model.num_timesteps,
            "episodes": callback.episodes,
            "latest_model": f"model_{model.num_timesteps:09d}.zip",
            "complete": model.num_timesteps >= total_timesteps,
        }
        _atomic_json(status_path, result)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gamma", type=float, choices=(1.0, 0.995), default=1.0)
    parser.add_argument("--total-timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--checkpoint-steps", type=int, default=DEFAULT_CHECKPOINT_STEPS)
    parser.add_argument("--max-new-timesteps", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    print(json.dumps(train(
        args.output, seed=args.seed, gamma=args.gamma,
        total_timesteps=args.total_timesteps, n_steps=args.n_steps,
        checkpoint_steps=args.checkpoint_steps,
        max_new_timesteps=args.max_new_timesteps, device=args.device,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
