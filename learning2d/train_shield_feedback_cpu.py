"""ARM64/CPU paired PPO fine-tuning from the frozen v4 policy weights.

Both conditions use the same shielded plant and a fresh optimizer. Only the
extra learning penalty for takeover and rejected departure differs.
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
from dual_constraint_2d.train_ppo import ROOT, TRAIN_MAP_IDS, VALIDATION_MAP_IDS, SaveProgress
from .mode_policy import ModeMaskedPolicy  # noqa: F401; needed when loading v4
from .shield_feedback_gym import ShieldFeedbackGym


PROTOCOL = "dual_constraint_2d_shield_feedback_paired_arm_cpu_v1"
CONDITIONS = ("control", "feedback")
SEED = 101
TOTAL_TIMESTEPS = 25600
N_STEPS = 512
CHECKPOINT_STEPS = 512
PARENT_MODEL = ROOT / "evidence/mode_ppo_v4_lambda1_seed101_20260926/training/model_000051200.zip"
PARENT_MANIFEST = PARENT_MODEL.parent / "manifest.json"
PARENT_SHA256 = "71ca7c3382f4fb994f90a1334b70fcece5ea5fdf1cb98b3319e069b696858429"
SOURCE_FILES = (
    "dual_constraint_2d/action_adapter.py",
    "dual_constraint_2d/backup.py",
    "dual_constraint_2d/calibration.py",
    "dual_constraint_2d/environment.py",
    "dual_constraint_2d/gym_adapter.py",
    "dual_constraint_2d/reference.py",
    "dual_constraint_2d/routes.py",
    "dual_constraint_2d/shield.py",
    "dual_constraint_2d/synthetic.py",
    "dual_constraint_2d/targets.py",
    "dual_constraint_2d/train_ppo.py",
    "learning2d/finite_horizon_gym.py",
    "learning2d/mode_policy.py",
    "learning2d/shield_feedback_gym.py",
    "learning2d/train_shield_feedback_cpu.py",
    "nav3d/controller.py",
    "nav3d/geometry.py",
    "nav3d/simulation.py",
    "SERVER_2D_RESEARCH_HANDOFF_20260926.md",
    "ARM_CPU_2D_EXECUTION_20260926.md",
    "scripts/run_2d_shield_feedback_arm_cpu.sh",
)


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def check_arm_cpu_runtime() -> None:
    if platform.machine().lower() not in ("aarch64", "arm64"):
        raise RuntimeError("this launch is restricted to ARM64")
    torch.set_num_threads(1)
    if torch.get_num_threads() != 1:
        raise RuntimeError("PPO training must use one CPU thread per process")


def verify_parent() -> dict:
    if _hash(PARENT_MODEL) != PARENT_SHA256:
        raise RuntimeError("frozen v4 parent model hash changed")
    parent = json.loads(PARENT_MANIFEST.read_text(encoding="utf-8"))
    if (parent["protocol"] != "dual_constraint_2d_mode_masked_long_credit_v4"
            or parent["seed"] != SEED or parent["ppo"]["gamma"] != 1.0
            or parent["ppo"]["gae_lambda"] != 1.0):
        raise RuntimeError("unexpected v4 parent manifest")
    for relative, digest in parent["source_sha256"].items():
        if _hash(ROOT / relative) != digest:
            raise RuntimeError(f"parent source changed: {relative}")
    return parent


def training_manifest(condition: str) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("condition must be control or feedback")
    calibration_path = CALIBRATION_OUTPUT / "calibration.json"
    return {
        "protocol": PROTOCOL,
        "condition": condition,
        "architecture": platform.machine(),
        "device": "cpu",
        "torch_cpu_threads": 1,
        "seed": SEED,
        "train_map_ids": list(TRAIN_MAP_IDS),
        "validation_map_ids": list(VALIDATION_MAP_IDS),
        "horizon_s": 600.0,
        "horizon_endpoint": "terminated_no_bootstrap",
        "total_timesteps": TOTAL_TIMESTEPS,
        "n_steps": N_STEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "policy": "ModeMaskedPolicy",
        "plant": "shielded_evaluation_plant",
        "initialization": "v4_policy_weights_only; fresh_optimizer; new timestep count",
        "parent_model_sha256": PARENT_SHA256,
        "parent_manifest_sha256": _hash(PARENT_MANIFEST),
        "calibration_sha256": _hash(calibration_path),
        "reward": {
            "parent_takeover_penalty": -0.2,
            "extra_takeover_penalty": -1.8 if condition == "feedback" else 0.0,
            "rejected_departure_penalty": -2.0 if condition == "feedback" else 0.0,
            "ordinary_qp_intervention_extra_penalty": 0.0,
        },
        "ppo": {
            "net_arch": [128, 128], "batch_size": 128, "n_epochs": 4,
            "learning_rate": 0.0003, "gamma": 1.0, "gae_lambda": 1.0,
            "clip_range": 0.2, "ent_coef": 0.01, "device": "cpu",
        },
        "python": platform.python_version(),
        "packages": {
            "numpy": numpy.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__, "torch": torch.__version__,
        },
        "source_sha256": {file: _hash(ROOT / file) for file in SOURCE_FILES},
    }


def _fresh_model(env: ShieldFeedbackGym) -> PPO:
    parent = PPO.load(str(PARENT_MODEL), device="cpu")
    model = PPO(
        ModeMaskedPolicy, env, seed=SEED, verbose=0, device="cpu",
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=N_STEPS, batch_size=128, n_epochs=4,
        learning_rate=0.0003, gamma=1.0, gae_lambda=1.0,
        clip_range=0.2, ent_coef=0.01,
    )
    model.policy.load_state_dict(parent.policy.state_dict(), strict=True)
    for name, parameter in parent.policy.state_dict().items():
        if not torch.equal(parameter, model.policy.state_dict()[name]):
            raise RuntimeError(f"parent parameter transfer failed: {name}")
    if model.policy.optimizer.state or model.num_timesteps != 0 or str(model.device) != "cpu":
        raise RuntimeError("fine-tuning did not start with fresh CPU optimizer state")
    return model


def train(output: Path, *, condition: str, max_new_timesteps: int | None = None) -> dict:
    if max_new_timesteps is not None and (
            max_new_timesteps <= 0 or max_new_timesteps % N_STEPS):
        raise ValueError("max-new-timesteps must be a positive multiple of 512")
    check_arm_cpu_runtime()
    verify_parent()
    manifest = training_manifest(condition)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("training manifest changed; refusing mixed results")
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
        feedback=condition == "feedback", map_seed=SEED,
    )
    try:
        if status is None:
            model = _fresh_model(env)
            episodes = 0
        else:
            model = PPO.load(str(output / status["latest_model"]), env=env, device="cpu")
            episodes = int(status["episodes"])
            if (model.num_timesteps != status["timesteps"] or str(model.device) != "cpu"
                    or model.gamma != 1.0 or model.gae_lambda != 1.0):
                raise RuntimeError("saved CPU model disagrees with frozen protocol")
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
    parser.add_argument("--max-new-timesteps", type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.output, condition=args.condition,
                           max_new_timesteps=args.max_new_timesteps), sort_keys=True))


if __name__ == "__main__":
    main()
