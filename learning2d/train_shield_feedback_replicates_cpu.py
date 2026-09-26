"""Independent single-environment paired CPU seeds run alongside primary PPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
import torch

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.train_ppo import ROOT, SaveProgress, TRAIN_MAP_IDS
from .mode_policy import ModeMaskedPolicy  # noqa: F401; frozen parent policy
from .shield_feedback_gym import ShieldFeedbackGym
from .train_shield_feedback_cpu import (
    CONDITIONS, PARENT_MODEL, SOURCE_FILES as BASE_SOURCE_FILES,
    TOTAL_TIMESTEPS, _atomic_json, _hash, check_arm_cpu_runtime,
    training_manifest as base_training_manifest, verify_parent,
)


PROTOCOL = "dual_constraint_2d_shield_feedback_cpu_independent_seeds_v1"
SEEDS = tuple(range(102, 107))
CHECKPOINT_STEPS = 512
SOURCE_FILES = tuple(dict.fromkeys(BASE_SOURCE_FILES + (
    "learning2d/train_shield_feedback_replicates_cpu.py",
    "learning2d/evaluate_shield_feedback_replicates_cpu.py",
    "PAIRED_2D_CPU_REPLICATION_PROTOCOL_20260926.md",
    "scripts/run_2d_shield_feedback_cpu_replicates.sh",
)))


def training_manifest(condition: str, seed: int) -> dict:
    if condition not in CONDITIONS or seed not in SEEDS:
        raise ValueError("invalid condition or frozen replication seed")
    manifest = base_training_manifest(condition)
    manifest["protocol"] = PROTOCOL
    manifest["seed"] = seed
    manifest["parent_policy_seed"] = 101
    manifest["panel_seeds"] = list(SEEDS)
    manifest["n_envs"] = 1
    manifest["n_steps"] = CHECKPOINT_STEPS
    manifest["comparison_boundary"] = "single-environment replication; not pooled with vectorized run"
    manifest["source_sha256"] = {file: _hash(ROOT / file) for file in SOURCE_FILES}
    return manifest


def _fresh_model(env: ShieldFeedbackGym, *, seed: int) -> PPO:
    parent = PPO.load(str(PARENT_MODEL), device="cpu")
    model = PPO(
        ModeMaskedPolicy, env, seed=seed, verbose=0, device="cpu",
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=CHECKPOINT_STEPS, batch_size=128, n_epochs=4,
        learning_rate=0.0003, gamma=1.0, gae_lambda=1.0,
        clip_range=0.2, ent_coef=0.01,
    )
    model.policy.load_state_dict(parent.policy.state_dict(), strict=True)
    for name, parameter in parent.policy.state_dict().items():
        if not torch.equal(parameter, model.policy.state_dict()[name]):
            raise RuntimeError(f"parent parameter transfer failed: {name}")
    if model.policy.optimizer.state or model.num_timesteps or str(model.device) != "cpu":
        raise RuntimeError("replication did not start with a fresh CPU optimizer")
    return model


def train(output: Path, *, condition: str, seed: int,
          max_new_timesteps: int | None = None) -> dict:
    if max_new_timesteps is not None and (
            max_new_timesteps <= 0 or max_new_timesteps % CHECKPOINT_STEPS):
        raise ValueError("max-new-timesteps must be a positive multiple of 512")
    check_arm_cpu_runtime()
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("replication requires a CPU-only PyTorch wheel")
    verify_parent()
    manifest = training_manifest(condition, seed)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("replication manifest changed; refusing mixed results")
    else:
        _atomic_json(manifest_path, manifest)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    if status and status.get("complete"):
        return status
    calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text(encoding="utf-8"))
    env = ShieldFeedbackGym(
        TRAIN_MAP_IDS, calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"], shielded=True,
        feedback=condition == "feedback", map_seed=seed,
    )
    try:
        if status is None:
            model = _fresh_model(env, seed=seed)
            episodes = 0
        else:
            model = PPO.load(str(output / status["latest_model"]), env=env, device="cpu")
            episodes = int(status["episodes"])
            if (model.num_timesteps != status["timesteps"] or model.n_envs != 1
                    or model.n_steps != CHECKPOINT_STEPS or str(model.device) != "cpu"
                    or model.gamma != 1.0 or model.gae_lambda != 1.0):
                raise RuntimeError("saved CPU replication model disagrees with protocol")
        if model.num_timesteps >= TOTAL_TIMESTEPS:
            result = {**status, "complete": True}
            _atomic_json(status_path, result)
            return result
        callback = SaveProgress(output, initial_episodes=episodes)
        limit = (TOTAL_TIMESTEPS if max_new_timesteps is None else
                 min(TOTAL_TIMESTEPS, model.num_timesteps + max_new_timesteps))
        while model.num_timesteps < limit:
            model.learn(total_timesteps=CHECKPOINT_STEPS, reset_num_timesteps=False,
                        callback=callback)
            callback.model = model
            callback.save_after_update()
        result = {"timesteps": model.num_timesteps, "episodes": callback.episodes,
                  "latest_model": f"model_{model.num_timesteps:09d}.zip",
                  "complete": model.num_timesteps >= TOTAL_TIMESTEPS}
        _atomic_json(status_path, result)
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--max-new-timesteps", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.output, condition=args.condition, seed=args.seed,
                           max_new_timesteps=args.max_new_timesteps), sort_keys=True))


if __name__ == "__main__":
    main()
