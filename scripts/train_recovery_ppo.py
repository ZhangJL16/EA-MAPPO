"""Resumable reward-only PPO adaptation to the user-locked recovery environment."""

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

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.recovery_ppo import (
    collect_recovery,
    update_reward_only,
)
from scripts import run_directional_lagrangian_pair as storage
from scripts.audit_directional_recovery import summarize
from scripts.run_directional_cost_trace_pair import refresh_logs, restore
from scripts.train_directional_ppo import atomic_json


def evaluate(policy, vec, root, cohort, count):
    results = {}
    before = storage.state_hash(policy)
    for mode in ("deterministic", "stochastic"):
        output = root / f"evaluation_{cohort:04d}_{mode}"
        output.mkdir(exist_ok=True)
        rows = []
        for start in range(0, count, vec.num_envs):
            seeds = list(
                range(593800001 + start, 593800001 + min(start + vec.num_envs, count))
            )
            path = output / f"cohort_{start // vec.num_envs:04d}.json"
            if path.exists():
                current = json.loads(path.read_text())
                if sorted(r["seed"] for r in current) != seeds:
                    raise ValueError("saved evaluation seeds mismatch")
            else:
                if storage.PAUSED or (root / "PAUSE").exists():
                    return None
                set_random_seed(
                    693800001 + seeds[0], using_cuda=policy.device.type == "cuda"
                )
                _, current, _ = collect_cohort(
                    policy,
                    vec,
                    seeds,
                    deterministic=mode == "deterministic",
                    training=False,
                )
                for row in current:
                    row.pop("outcome")
                if storage.state_hash(policy) != before:
                    raise RuntimeError("evaluation changed weights")
                atomic_json(path, sorted(current, key=lambda r: r["seed"]))
            rows.extend(current)
            atomic_json(
                root / "status.json",
                {
                    "status": "EVALUATING",
                    "pid": os.getpid(),
                    "cohort": cohort,
                    "mode": mode,
                    "tasks": len(rows),
                },
            )
        results[mode] = summarize(rows)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-cohorts", type=int, default=32)
    parser.add_argument("--eval-tasks", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.extra_cohorts <= 0 or args.eval_tasks < 0:
        raise ValueError("invalid experiment budget")
    parent = json.loads((args.source / "metadata.json").read_text())
    if parent["arm"] != "control" or parent["cohort"] != 64:
        raise ValueError("expected original control checkpoint64")
    for name, digest in parent["contract"]["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical source changed: {name}")
    sources = set(parent["contract"]["source_hashes"]) | {
        "scripts/train_recovery_ppo.py",
        "scripts/run_directional_cost_trace_pair.py",
        "scripts/audit_directional_recovery.py",
        "experiments/directional_navigation/recovery.py",
        "experiments/directional_navigation/recovery_ppo.py",
    }
    contract = {
        "protocol": "locked_recovery_reward_only_ppo_v1",
        "parent_checkpoint_sha256": hashlib.sha256(
            (args.source / "state.pt").read_bytes()
        ).hexdigest(),
        "parent_contract": parent["contract"],
        "source_hashes": {
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
            for p in sorted(sources)
        },
        "eval_tasks": args.eval_tasks,
        "reward_gae": 0.95,
        "gamma": 1.0,
        "clip": 0.2,
        "epochs": 10,
        "batch_size": 256,
        "cudnn_deterministic": True,
        "cost_loss": False,
        "multiplier": 0.0,
        "formal_500": False,
    }
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("resume contract mismatch")
    else:
        atomic_json(
            root / "manifest.json",
            {
                "contract": contract,
                "command": sys.argv,
                "source": str(args.source.resolve()),
            },
        )
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, storage.pause)
    signal.signal(signal.SIGINT, storage.pause)
    atomic_json(root / "status.json", {"status": "STARTING", "pid": os.getpid()})
    vec = SubprocVecEnv(
        [partial(RecoveryCohort) for _ in range(8)], start_method="forkserver"
    )
    started = time.perf_counter()
    output = root / "model"
    output.mkdir(exist_ok=True)
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
            if state["contract"] != contract:
                raise ValueError("checkpoint contract mismatch")
            restore(policy, checkpoint)
        else:
            restore(policy, args.source)
            state = {
                "cohort": parent["cohort"],
                "transitions": parent["transitions"],
                "contract": contract,
                "arm": "recovery_reward_ppo",
                "multiplier": 0.0,
                "fork_policy_hash": storage.state_hash(policy),
            }
            storage.save(output / "checkpoint_0064", policy, state)
        target = parent["cohort"] + args.extra_cohorts
        for cohort in range(state["cohort"], target):
            if storage.PAUSED or (root / "PAUSE").exists():
                atomic_json(
                    root / "status.json",
                    {"status": "PAUSED", "cohort": state["cohort"]},
                )
                return
            seeds = list(range(483600001 + cohort * 8, 483600001 + (cohort + 1) * 8))
            data, rows, collection = collect_recovery(policy, vec, seeds)
            metrics = update_reward_only(policy, data)
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
            refresh_logs(output, parent["cohort"], state["cohort"])
            atomic_json(
                root / "status.json",
                {"status": "TRAINING", "pid": os.getpid(), **record},
            )
            print(json.dumps(record), flush=True)
        refresh_logs(output, parent["cohort"], state["cohort"])
        if storage.PAUSED or (root / "PAUSE").exists():
            atomic_json(
                root / "status.json", {"status": "PAUSED", "cohort": state["cohort"]}
            )
            return
        result = evaluate(policy, vec, root, state["cohort"], args.eval_tasks)
        if result is None:
            atomic_json(
                root / "status.json", {"status": "PAUSED", "cohort": state["cohort"]}
            )
            return
        atomic_json(
            root / "RESULT.json",
            {
                "status": "COMPLETE",
                "cohort": state["cohort"],
                "transitions": state["transitions"],
                "new_transitions": state["transitions"] - parent["transitions"],
                "evaluation": result,
                "formal_500": False,
                "session_seconds": time.perf_counter() - started,
            },
        )
        atomic_json(root / "status.json", {"status": "COMPLETE"})
    except BaseException as exc:
        atomic_json(
            root / "error.json", {"type": type(exc).__name__, "message": str(exc)}
        )
        raise
    finally:
        vec.close()


if __name__ == "__main__":
    main()
