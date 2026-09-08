"""Long matched PPO continuation: KL-coupled versus completed value budget."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
import torch
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.critic_completion import (
    update_with_critic_completion,
)
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.recovery_ppo import collect_recovery
from scripts import run_directional_lagrangian_pair as storage
from scripts.audit_directional_recovery import summarize
from scripts.run_directional_cost_trace_pair import refresh_logs, restore
from scripts.train_directional_ppo import atomic_json


def paused(root: Path) -> bool:
    return storage.PAUSED or (root / "PAUSE").exists()


def evaluate(policy, vec, root, output, label, seed_start, count, deterministic):
    folder = output / label
    folder.mkdir(exist_ok=True)
    before, rows = storage.state_hash(policy), []
    for start in range(0, count, vec.num_envs):
        seeds = list(
            range(seed_start + start, seed_start + min(start + vec.num_envs, count))
        )
        path = folder / f"cohort_{start // vec.num_envs:04d}.json"
        if path.exists():
            current = json.loads(path.read_text())
            if sorted(r["seed"] for r in current) != seeds:
                raise ValueError("evaluation seed mismatch")
        else:
            if paused(root):
                return None
            set_random_seed(103000000 + seeds[0], using_cuda=True)
            _, current, _ = collect_cohort(
                policy, vec, seeds, deterministic=deterministic, training=False
            )
            for row in current:
                row.pop("outcome")
            if storage.state_hash(policy) != before:
                raise RuntimeError("evaluation modified model")
            atomic_json(path, sorted(current, key=lambda r: r["seed"]))
        rows.extend(current)
        atomic_json(
            root / "status.json",
            {
                "status": "EVALUATING",
                "pid": os.getpid(),
                "arm": output.name,
                "label": label,
                "tasks": len(rows),
                "target_tasks": count,
            },
        )
    result = summarize(rows)
    atomic_json(folder / "summary.json", result)
    return result


def run_arm(args, root, contract, replicate, complete):
    arm = f"replicate_{replicate}_" + (
        "critic_complete" if complete else "joint_stop_control"
    )
    output = root / arm
    output.mkdir(exist_ok=True)
    if (output / "RESULT.json").exists():
        return True
    parent = json.loads((args.source / "metadata.json").read_text())
    origin = parent["cohort"]
    vec = SubprocVecEnv(
        [partial(RecoveryCohort) for _ in range(8)], start_method="forkserver"
    )
    started = time.perf_counter()
    try:
        policy = TwoValuePolicy(
            vec.observation_space,
            vec.action_space,
            lambda _: 3e-4,
            features_extractor_class=DirectionalLidarExtractor,
            features_extractor_kwargs={"remaining_time": True},
            share_features_extractor=False,
            net_arch={"pi": [256, 256], "vf": [256, 256]},
            log_std_init=float(np.log(0.5)),
        ).to("cuda")
        latest = output / "latest.json"
        if latest.exists():
            checkpoint = output / json.loads(latest.read_text())["checkpoint"]
            state = json.loads((checkpoint / "metadata.json").read_text())
            if state["contract"] != contract or state["arm"] != arm:
                raise ValueError("checkpoint contract mismatch")
            restore(policy, checkpoint)
        else:
            restore(policy, args.source)
            state = {
                "cohort": origin,
                "transitions": parent["transitions"],
                "multiplier": 0.0,
                "arm": arm,
                "contract": contract,
                "fork_policy_hash": storage.state_hash(policy),
            }
            storage.save(output / f"checkpoint_{origin:04d}", policy, state)
        # Includes the saved boundary on resume so a partial milestone eval finishes
        # before the next update. No milestone outcome changes the training schedule.
        while True:
            completed = state["cohort"] - origin
            if paused(root):
                return False
            if (
                args.dev_tasks
                and completed
                and (completed % 32 == 0 or completed == args.cohorts)
            ):
                result = evaluate(
                    policy,
                    vec,
                    root,
                    output,
                    f"dev_{state['cohort']:04d}_deterministic",
                    593800001,
                    args.dev_tasks,
                    True,
                )
                if result is None:
                    return False
            if completed == args.cohorts:
                break
            cohort = state["cohort"]
            # Two continuation random streams from the SAME historical initialization.
            # Stream and scenario seed are matched within a replicate across arms.
            set_random_seed(913600001 + replicate * 10000 + cohort, using_cuda=True)
            first_seed = 813600001 + (replicate - 1) * 100000 + completed * 8
            seeds = list(range(first_seed, first_seed + 8))
            data, rows, collection = collect_recovery(policy, vec, seeds)
            metrics = update_with_critic_completion(policy, data, complete=complete)
            state.update(
                cohort=cohort + 1,
                transitions=state["transitions"] + collection["physical_steps"],
            )
            record = {
                "cohort": cohort + 1,
                "transitions": state["transitions"],
                "session_seconds": time.perf_counter() - started,
                **collection,
                **metrics,
            }
            state["last_update"] = record
            state["last_episodes"] = [dict(cohort=cohort + 1, **r) for r in rows]
            storage.save(output / f"checkpoint_{cohort + 1:04d}", policy, state)
            refresh_logs(output, origin, state["cohort"])
            atomic_json(
                root / "status.json",
                {
                    "status": "TRAINING",
                    "pid": os.getpid(),
                    "arm": arm,
                    "completed_cohorts": completed + 1,
                    "target_cohorts": args.cohorts,
                    **record,
                },
            )
            print(json.dumps({"arm": arm, **record}), flush=True)
        holdout = evaluate(
            policy,
            vec,
            root,
            output,
            "final_holdout_deterministic",
            793800001,
            args.eval_tasks,
            True,
        )
        if holdout is None:
            return False
        stochastic = evaluate(
            policy,
            vec,
            root,
            output,
            "final_dev_stochastic",
            593800001,
            args.dev_tasks,
            False,
        )
        if stochastic is None:
            return False
        atomic_json(
            output / "RESULT.json",
            {
                "status": "COMPLETE",
                "arm": arm,
                "replicate": replicate,
                "cohort": state["cohort"],
                "new_transitions": state["transitions"] - parent["transitions"],
                "holdout_deterministic": holdout,
                "development_stochastic": stochastic,
                "fork_policy_hash": state["fork_policy_hash"],
                "last_session_seconds": time.perf_counter() - started,
            },
        )
        return True
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohorts", type=int, default=96)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--eval-tasks", type=int, default=500)
    parser.add_argument("--dev-tasks", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        min(args.cohorts, args.replicates) < 1
        or min(args.eval_tasks, args.dev_tasks) < 0
    ):
        raise ValueError("invalid experiment size")
    parent = json.loads((args.source / "metadata.json").read_text())
    if parent["arm"] != "control" or parent["cohort"] != 64:
        raise ValueError("expected historical control64 source")
    for name, digest in parent["contract"]["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical source modified: {name}")
    sources = [
        "scripts/run_recovery_critic_completion.py",
        "experiments/directional_navigation/critic_completion.py",
        "experiments/directional_navigation/recovery_ppo.py",
        "experiments/directional_navigation/recovery.py",
        "scripts/run_directional_lagrangian_pair.py",
        "scripts/run_directional_cost_trace_pair.py",
    ]
    contract = {
        "protocol": "locked_recovery_critic_budget_v1",
        "parent_contract": parent["contract"],
        "parent_checkpoint_sha256": hashlib.sha256(
            (args.source / "state.pt").read_bytes()
        ).hexdigest(),
        "cohorts": args.cohorts,
        "replicates": args.replicates,
        "eval_tasks": args.eval_tasks,
        "dev_tasks": args.dev_tasks,
        "joint_epochs": 10,
        "batch_size": 256,
        "actor_gae": 0.95,
        "critic_targets": "unchanged GAE(.95) lambda return",
        "training_seed_start": 813600001,
        "holdout_seed_start": 793800001,
        "random_seed_formula": "913600001 + replicate*10000 + cohort",
        "same_historical_initialization": True,
        "automatic_promotion": False,
        "source_hashes": {
            s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources
        },
    }
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("resume manifest mismatch")
    else:
        atomic_json(
            root / "manifest.json",
            {"contract": contract, "command": sys.argv, "created_unix": time.time()},
        )
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, storage.pause)
    signal.signal(signal.SIGINT, storage.pause)
    atomic_json(root / "status.json", {"status": "STARTING", "pid": os.getpid()})
    try:
        for replicate in range(1, args.replicates + 1):
            for complete in (True, False) if replicate % 2 else (False, True):
                if not run_arm(args, root, contract, replicate, complete):
                    atomic_json(
                        root / "status.json",
                        {"status": "PAUSED", "replicate": replicate},
                    )
                    return
        results = {
            p.parent.name: json.loads(p.read_text())
            for p in root.glob("replicate_*/RESULT.json")
        }
        atomic_json(
            root / "RESULT.json",
            {"status": "COMPLETE", "arms": results, "completed_unix": time.time()},
        )
        atomic_json(root / "status.json", {"status": "COMPLETE"})
    except BaseException as exc:
        atomic_json(
            root / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
