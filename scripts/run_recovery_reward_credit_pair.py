"""Matched nonterminal-recovery PPO: actor GAE(.95) versus complete-return MC."""

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
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.recovery_credit import update_actor_reward_trace
from experiments.directional_navigation.recovery_ppo import collect_recovery
from scripts import run_directional_lagrangian_pair as storage
from scripts.run_directional_cost_trace_pair import refresh_logs, restore
from scripts.train_directional_ppo import atomic_json
from scripts.train_recovery_ppo import evaluate


def run_arm(args, root, contract, arm, trace):
    output = root / arm
    output.mkdir(exist_ok=True)
    parent = json.loads((args.source / "metadata.json").read_text())
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
                raise ValueError("resume contract mismatch")
            restore(policy, checkpoint)
        else:
            restore(policy, args.source)
            state = {
                "cohort": parent["cohort"],
                "transitions": parent["transitions"],
                "multiplier": 0.0,
                "arm": arm,
                "contract": contract,
                "fork_policy_hash": storage.state_hash(policy),
            }
            storage.save(output / f"checkpoint_{state['cohort']:04d}", policy, state)
        target = parent["cohort"] + contract["extra_cohorts"]
        for cohort in range(state["cohort"], target):
            if storage.PAUSED or (root / "PAUSE").exists():
                return False
            seeds = list(range(483600001 + cohort * 8, 483600001 + (cohort + 1) * 8))
            data, rows, collection = collect_recovery(policy, vec, seeds)
            metrics = update_actor_reward_trace(policy, data, rows, seeds, trace)
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
                {"status": "TRAINING", "pid": os.getpid(), "arm": arm, **record},
            )
            print(json.dumps({"arm": arm, **record}), flush=True)
        if storage.PAUSED or (root / "PAUSE").exists():
            return False
        result = evaluate(policy, vec, output, state["cohort"], contract["eval_tasks"])
        if result is None:
            return False
        atomic_json(
            output / "RESULT.json",
            {
                "status": "COMPLETE",
                "arm": arm,
                "cohort": state["cohort"],
                "trace": trace,
                "new_transitions": state["transitions"] - parent["transitions"],
                "evaluation": result,
                "session_seconds": time.perf_counter() - started,
                "fork_policy_hash": state["fork_policy_hash"],
            },
        )
        return True
    finally:
        vec.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-cohorts", type=int, default=16)
    parser.add_argument("--eval-tasks", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.extra_cohorts <= 0 or args.eval_tasks < 0:
        raise ValueError("invalid budget")
    parent = json.loads((args.source / "metadata.json").read_text())
    if parent["arm"] != "control" or parent["cohort"] != 64:
        raise ValueError("expected original control checkpoint64")
    for name, digest in parent["contract"]["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"historical source changed:{name}")
    source_files = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/recovery_credit.py",
        ROOT / "experiments/directional_navigation/recovery.py",
        ROOT / "experiments/directional_navigation/recovery_ppo.py",
        ROOT / "scripts/train_recovery_ppo.py",
        ROOT / "scripts/run_directional_cost_trace_pair.py",
    ]
    contract = {
        "protocol": "locked_recovery_actor_reward_trace_pair_v1",
        "parent_checkpoint_sha256": hashlib.sha256(
            (args.source / "state.pt").read_bytes()
        ).hexdigest(),
        "parent_contract": parent["contract"],
        "extra_cohorts": args.extra_cohorts,
        "eval_tasks": args.eval_tasks,
        "trace_by_arm": {"mc_actor": 1.0, "gae95_control": 0.95},
        "cost_loss": False,
        "multiplier": 0.0,
        "critic_target": "unchanged GAE(.95) lambda-return",
        "formal_500": False,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
    }
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("root contract mismatch")
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
    started = time.perf_counter()
    try:
        for arm, trace in contract["trace_by_arm"].items():
            if not run_arm(args, root, contract, arm, trace):
                atomic_json(root / "status.json", {"status": "PAUSED", "arm": arm})
                return
        results = {
            a: json.loads((root / a / "RESULT.json").read_text())
            for a in contract["trace_by_arm"]
        }
        atomic_json(
            root / "RESULT.json",
            {
                "status": "COMPLETE",
                "arms": results,
                "fork_weights_match": len(
                    {r["fork_policy_hash"] for r in results.values()}
                )
                == 1,
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
