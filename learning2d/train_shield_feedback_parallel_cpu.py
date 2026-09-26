"""Paired ARM CPU PPO fine-tuning with 16 environment workers in total."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.train_ppo import ROOT, SaveProgress
from .mode_policy import ModeMaskedPolicy  # noqa: F401; frozen parent policy
from .shield_feedback_gym import ShieldFeedbackGym
from .train_shield_feedback_cpu import (
    CONDITIONS, PARENT_MODEL, SEED, SOURCE_FILES as BASE_SOURCE_FILES,
    TOTAL_TIMESTEPS, _atomic_json, _hash, check_arm_cpu_runtime,
    training_manifest as base_training_manifest, verify_parent,
)


PROTOCOL = "dual_constraint_2d_shield_feedback_paired_arm_cpu_16workers_v1"
WORKERS_PER_CONDITION = 8
N_STEPS_PER_WORKER = 64
CHECKPOINT_STEPS = WORKERS_PER_CONDITION * N_STEPS_PER_WORKER
SOURCE_FILES = tuple(dict.fromkeys(BASE_SOURCE_FILES + (
    "learning2d/train_shield_feedback_parallel_cpu.py",
    "learning2d/evaluate_shield_feedback_parallel_cpu.py",
    "PARALLEL_2D_SHIELD_FEEDBACK_PROTOCOL_20260926.md",
    "scripts/run_2d_shield_feedback_arm_cpu_16workers.sh",
)))


def _make_worker(condition: str, rank: int, capacity: float, charge_seconds: float):
    if condition not in CONDITIONS or not 0 <= rank < WORKERS_PER_CONDITION:
        raise ValueError("invalid condition or worker rank")
    return ShieldFeedbackGym(
        tuple(range(8, 32)), capacity, charge_seconds,
        shielded=True, feedback=condition == "feedback", map_seed=SEED + rank,
    )


def training_manifest(condition: str) -> dict:
    manifest = base_training_manifest(condition)
    manifest["protocol"] = PROTOCOL
    manifest["worker_processes_per_condition"] = WORKERS_PER_CONDITION
    manifest["simultaneous_conditions"] = list(CONDITIONS)
    manifest["total_environment_workers"] = 2 * WORKERS_PER_CONDITION
    manifest["worker_seeds"] = list(range(SEED, SEED + WORKERS_PER_CONDITION))
    manifest["subprocess_start_method"] = "forkserver"
    manifest["n_steps_per_worker"] = N_STEPS_PER_WORKER
    manifest["n_steps"] = CHECKPOINT_STEPS
    manifest["checkpoint_steps"] = CHECKPOINT_STEPS
    manifest["sampling_change_from_single_env"] = (
        "8 workers x 64 decisions rather than 1 worker x 512; new protocol"
    )
    manifest["source_sha256"] = {file: _hash(ROOT / file) for file in SOURCE_FILES}
    return manifest


def _fresh_model(env: SubprocVecEnv) -> PPO:
    parent = PPO.load(str(PARENT_MODEL), device="cpu")
    model = PPO(
        ModeMaskedPolicy, env, seed=SEED, verbose=0, device="cpu",
        policy_kwargs={"net_arch": [128, 128]},
        n_steps=N_STEPS_PER_WORKER, batch_size=128, n_epochs=4,
        learning_rate=0.0003, gamma=1.0, gae_lambda=1.0,
        clip_range=0.2, ent_coef=0.01,
    )
    model.policy.load_state_dict(parent.policy.state_dict(), strict=True)
    for name, parameter in parent.policy.state_dict().items():
        if not torch.equal(parameter, model.policy.state_dict()[name]):
            raise RuntimeError(f"parent parameter transfer failed: {name}")
    if (model.policy.optimizer.state or model.num_timesteps != 0
            or model.n_envs != WORKERS_PER_CONDITION or model.n_steps != N_STEPS_PER_WORKER
            or str(model.device) != "cpu"):
        raise RuntimeError("parallel run did not start with frozen fresh-optimizer settings")
    return model


def train(output: Path, *, condition: str, max_new_timesteps: int | None = None) -> dict:
    if condition not in CONDITIONS:
        raise ValueError("condition must be control or feedback")
    if max_new_timesteps is not None and (
            max_new_timesteps <= 0 or max_new_timesteps % CHECKPOINT_STEPS):
        raise ValueError("max-new-timesteps must be a positive multiple of 512")
    check_arm_cpu_runtime()
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("parallel run requires a CPU-only PyTorch wheel")
    verify_parent()
    manifest = training_manifest(condition)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise RuntimeError("parallel training manifest changed; refusing mixed results")
    else:
        _atomic_json(manifest_path, manifest)
    status_path = output / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    if status and status.get("complete"):
        return status
    calibration = json.loads((CALIBRATION_OUTPUT / "calibration.json").read_text(encoding="utf-8"))
    factories = [partial(
        _make_worker, condition, rank, calibration["capacity_synthetic_energy"],
        calibration["full_charge_seconds"],
    ) for rank in range(WORKERS_PER_CONDITION)]
    env = SubprocVecEnv(factories, start_method="forkserver")
    try:
        if len(env.processes) != WORKERS_PER_CONDITION or not all(p.is_alive() for p in env.processes):
            raise RuntimeError("not all eight CPU environment workers started")
        print(json.dumps({"condition": condition,
                          "environment_worker_pids": [p.pid for p in env.processes]}), flush=True)
        if status is None:
            model = _fresh_model(env)
            episodes = 0
        else:
            model = PPO.load(str(output / status["latest_model"]), env=env, device="cpu")
            episodes = int(status["episodes"])
            if (model.num_timesteps != status["timesteps"] or str(model.device) != "cpu"
                    or model.gamma != 1.0 or model.gae_lambda != 1.0
                    or model.n_envs != WORKERS_PER_CONDITION
                    or model.n_steps != N_STEPS_PER_WORKER):
                raise RuntimeError("saved parallel CPU model disagrees with frozen protocol")
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
