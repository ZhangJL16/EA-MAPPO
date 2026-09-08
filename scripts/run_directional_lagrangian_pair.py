"""Matched complete-task PPO-Lagrangian / lambda-zero control, no monitoring loop."""

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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.cohort import (
    CohortNavigation,
    projected_multiplier,
)
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import (
    TwoValuePolicy,
    collect_cohort,
    update_policy,
)
from scripts.train_directional_ppo import append_json, atomic_json

PAUSED = False


def pause(signum, frame):
    global PAUSED
    PAUSED = True


def state_hash(policy: TwoValuePolicy) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(policy.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def save(path: Path, policy: TwoValuePolicy, state: dict) -> None:
    if path.exists():
        return
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.mkdir(exist_ok=False)
    torch.save(
        {
            "model": policy.state_dict(),
            "optimizer": policy.optimizer.state_dict(),
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
        temporary / "state.pt",
    )
    atomic_json(temporary / "metadata.json", state)
    os.replace(temporary, path)
    atomic_json(path.parent / "latest.json", {"checkpoint": path.name})


def evaluation(
    policy, vec, output: Path, cohort: int, count: int, deterministic: bool
) -> dict:
    label = "deterministic" if deterministic else "stochastic"
    path = output / f"evaluation_{cohort:04d}_{label}.jsonl"
    records = (
        list(map(json.loads, path.read_text().splitlines())) if path.exists() else []
    )
    seen = {row["seed"] for row in records}
    all_seeds = list(range(593800001, 593800001 + count))
    pending = [seed for seed in all_seeds if seed not in seen]
    steps = 0
    for start in range(0, len(pending), vec.num_envs):
        if PAUSED or (output.parent / "PAUSE").exists():
            break
        # Keyed RNG per evaluation cohort; resume does not depend on prior
        # episode lengths consuming a stochastic action stream.
        seeds = pending[start : start + vec.num_envs]
        set_random_seed(693800001 + seeds[0], using_cuda=policy.device.type == "cuda")
        _, rows, audit = collect_cohort(
            policy, vec, seeds, deterministic=deterministic, training=False
        )
        steps += audit["physical_steps"]
        records.extend(rows)
        for row in rows:
            append_json(path, row)
    goals = [r["success_path_ratio"] for r in records if r["outcome"] == "safe_goal"]
    return {
        "tasks": len(records),
        "complete": len(records) == count,
        "counts": {
            k: sum(r["outcome"] == k for r in records)
            for k in ("safe_goal", "contact", "timeout")
        },
        "success_path_ratio": float(np.mean(goals)) if goals else None,
        "success_path_ratio_n": len(goals),
        "physical_steps_this_session": steps,
        "formal_500": False,
    }


def run_arm(args, arm: str, contract: dict, root: Path) -> bool:
    output = root / arm
    output.mkdir(exist_ok=True)
    env_fns: list[Callable[[], gym.Env]] = [
        partial(CohortNavigation, horizon=args.horizon, obstacles=args.obstacles)
        for _ in range(args.num_envs)
    ]
    vec = SubprocVecEnv(env_fns, start_method="forkserver")
    started = time.perf_counter()
    try:
        set_random_seed(193500001, using_cuda=args.device.startswith("cuda"))
        policy = TwoValuePolicy(
            vec.observation_space,
            vec.action_space,
            lambda _: 3e-4,
            features_extractor_class=DirectionalLidarExtractor,
            features_extractor_kwargs={"remaining_time": True},
            share_features_extractor=False,
            net_arch={"pi": [256, 256], "vf": [256, 256]},
            log_std_init=float(np.log(0.5)),
        ).to(args.device)
        state = {
            "cohort": 0,
            "transitions": 0,
            "multiplier": 0.0,
            "arm": arm,
            "contract": contract,
            "initial_policy_hash": state_hash(policy),
            "parameters": sum(p.numel() for p in policy.parameters()),
        }
        latest = output / "latest.json"
        if latest.exists():
            checkpoint = output / json.loads(latest.read_text())["checkpoint"]
            state = json.loads((checkpoint / "metadata.json").read_text())
            if state["contract"] != contract or state["arm"] != arm:
                raise ValueError("resume source/contract mismatch")
            payload = torch.load(
                checkpoint / "state.pt", map_location=args.device, weights_only=False
            )
            policy.load_state_dict(payload["model"])
            policy.optimizer.load_state_dict(payload["optimizer"])
            random.setstate(payload["python_rng"])
            np.random.set_state(payload["numpy_rng"])
            torch.set_rng_state(payload["torch_rng"].cpu())
            if args.device.startswith("cuda") and payload["cuda_rng"] is not None:
                torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])
        else:
            save(output / "checkpoint_0000", policy, state)
        for cohort in range(state["cohort"], args.cohorts):
            if PAUSED or (root / "PAUSE").exists():
                return False
            seeds = [
                483600001 + cohort * args.num_envs + i for i in range(args.num_envs)
            ]
            data, rows, collection = collect_cohort(policy, vec, seeds)
            before = state["multiplier"]
            multiplier = (
                projected_multiplier(
                    before,
                    collection["task_contact_rate"],
                    args.cost_limit,
                    args.dual_lr,
                )
                if arm == "lagrangian"
                else 0.0
            )
            metrics = update_policy(policy, data, multiplier, epochs=args.epochs)
            state.update(
                cohort=cohort + 1,
                transitions=state["transitions"] + collection["physical_steps"],
                multiplier=multiplier,
            )
            record = dict(
                cohort=cohort + 1,
                transitions=state["transitions"],
                multiplier=multiplier,
                multiplier_before=before,
                session_seconds=time.perf_counter() - started,
                **collection,
                **metrics,
            )
            for row in rows:
                append_json(
                    output / "training_episodes.jsonl", dict(cohort=cohort + 1, **row)
                )
            append_json(output / "updates.jsonl", record)
            save(output / f"checkpoint_{cohort + 1:04d}", policy, state)
            atomic_json(
                root / "status.json",
                dict(status="TRAINING", arm=arm, pid=os.getpid(), **record),
            )
            print(json.dumps(dict(arm=arm, **record)), flush=True)
        if PAUSED or (root / "PAUSE").exists():
            return False
        atomic_json(
            root / "status.json",
            {
                "status": "EVALUATING",
                "arm": arm,
                "pid": os.getpid(),
                "cohort": state["cohort"],
            },
        )
        results = {
            label: evaluation(
                policy, vec, output, state["cohort"], args.eval_tasks, deterministic
            )
            for label, deterministic in (("deterministic", True), ("stochastic", False))
        }
        complete = all(value["complete"] for value in results.values())
        atomic_json(
            output / "RESULT.json",
            {
                "status": "COMPLETE" if complete else "EVALUATION_PAUSED",
                "cohort": state["cohort"],
                "transitions": state["transitions"],
                "multiplier": state["multiplier"],
                "initial_policy_hash": state["initial_policy_hash"],
                "evaluation": results,
                "session_seconds": time.perf_counter() - started,
            },
        )
        return complete
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cohorts", type=int, default=32)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4000)
    parser.add_argument("--obstacles", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--eval-tasks", type=int, default=50)
    parser.add_argument("--cost-limit", type=float, default=0.05)
    parser.add_argument("--dual-lr", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if (
        min(args.cohorts, args.num_envs, args.horizon, args.epochs) <= 0
        or args.eval_tasks < 0
    ):
        raise ValueError("invalid budget")
    projected_multiplier(0.0, 0.0, args.cost_limit, args.dual_lr)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, pause)
    signal.signal(signal.SIGINT, pause)
    sources = [
        Path(__file__),
        ROOT / "scripts/train_directional_ppo.py",
        *[
            ROOT / "experiments/directional_navigation" / f"{name}.py"
            for name in ("cohort", "environment", "features", "lagrangian")
        ],
        ROOT / "experiments/jacobian_energy_bridge/features.py",
        ROOT / "envs/UAVEnergyDeliverySAC.py",
    ]
    contract = {
        "protocol": "complete_task_ppo_lagrangian_pair_v1",
        "num_envs": args.num_envs,
        "horizon": args.horizon,
        "obstacles": args.obstacles,
        "epochs": args.epochs,
        "cost_limit": args.cost_limit,
        "dual_lr": args.dual_lr,
        "reward_scale": 0.01,
        "gamma_reward": 1.0,
        "gamma_cost": 1.0,
        "gae_reward": 0.95,
        "gae_cost": 1.0,
        "sb3_version": stable_baselines3.__version__,
        "torch_version": torch.__version__,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    }
    if not args.resume:
        atomic_json(
            root / "manifest.json",
            {
                "contract": contract,
                "command": sys.argv,
                "comparison": "Both arms fit reward/cost values; only multiplier feedback differs.",
                "budget": "Same registered task cohorts, not equal transitions. Count actual steps, gradients and wall time.",
                "limit_note": "0.05 is an exploratory stochastic-policy task-risk target, not deployment certification.",
            },
        )
    started = time.perf_counter()
    atomic_json(
        root / "status.json",
        {"status": "STARTING", "pid": os.getpid(), "started_unix": time.time()},
    )
    try:
        for arm in ("lagrangian", "control"):
            if not run_arm(args, arm, contract, root):
                atomic_json(
                    root / "status.json",
                    {
                        "status": "PAUSED",
                        "arm": arm,
                        "session_seconds": time.perf_counter() - started,
                    },
                )
                return
        results = {
            arm: json.loads((root / arm / "RESULT.json").read_text())
            for arm in ("lagrangian", "control")
        }
        atomic_json(
            root / "RESULT.json",
            {
                "status": "COMPLETE",
                "arms": results,
                "initial_weights_match": results["lagrangian"]["initial_policy_hash"]
                == results["control"]["initial_policy_hash"],
                "session_seconds": time.perf_counter() - started,
            },
        )
        atomic_json(root / "status.json", {"status": "COMPLETE"})
    except BaseException as exc:
        atomic_json(
            root / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
