"""Resumable matched continuation: MC cost advantage vs GAE-RTG, same policy fork."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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

from experiments.directional_navigation.cohort import (
    CohortNavigation,
    projected_multiplier,
)
from experiments.directional_navigation.cost_trace import update_with_cost_trace
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy, collect_cohort
from scripts import run_directional_lagrangian_pair as storage
from scripts.train_directional_ppo import atomic_json


def restore(policy: TwoValuePolicy, checkpoint: Path) -> None:
    # RNG checkpoints do not serialize backend flags. Match the original SB3
    # set_random_seed(using_cuda=True) contract BEFORE any GPU update.
    if policy.device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    payload = torch.load(
        checkpoint / "state.pt", map_location=policy.device, weights_only=False
    )
    policy.load_state_dict(payload["model"])
    policy.optimizer.load_state_dict(payload["optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if policy.device.type == "cuda":
        torch.cuda.set_rng_state_all([s.cpu() for s in payload["cuda_rng"]])


def run_arm(args, root: Path, contract: dict, arm: str, trace: float) -> bool:
    output = root / arm
    output.mkdir(exist_ok=True)
    parent = json.loads((args.source / "metadata.json").read_text())
    if parent["arm"] != "lagrangian" or parent["contract"]["dual_lr"] != 0.01:
        raise ValueError("expected a slowdual Lagrangian source checkpoint")
    original_contract = parent["contract"]
    nenv = original_contract["num_envs"]
    vec = SubprocVecEnv(
        [
            partial(
                CohortNavigation,
                horizon=original_contract["horizon"],
                obstacles=original_contract["obstacles"],
            )
            for _ in range(nenv)
        ],
        start_method="forkserver",
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
        ).to(args.device)
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
                "multiplier": parent["multiplier"],
                "arm": arm,
                "contract": contract,
                "fork_policy_hash": storage.state_hash(policy),
            }
            storage.save(output / f"checkpoint_{state['cohort']:04d}", policy, state)
        target = parent["cohort"] + args.extra_cohorts
        for cohort in range(state["cohort"], target):
            if storage.PAUSED or (root / "PAUSE").exists():
                return False
            seeds = list(
                range(483600001 + cohort * nenv, 483600001 + (cohort + 1) * nenv)
            )
            data, episodes, collection = collect_cohort(policy, vec, seeds)
            multiplier = projected_multiplier(
                state["multiplier"],
                collection["task_contact_rate"],
                original_contract["cost_limit"],
                original_contract["dual_lr"],
            )
            metrics = update_with_cost_trace(
                policy,
                data,
                episodes,
                seeds,
                multiplier,
                trace,
                epochs=original_contract["epochs"],
            )
            before = state["multiplier"]
            state.update(
                cohort=cohort + 1,
                multiplier=multiplier,
                transitions=state["transitions"] + collection["physical_steps"],
            )
            record = {
                "cohort": cohort + 1,
                "transitions": state["transitions"],
                "multiplier": multiplier,
                "multiplier_before": before,
                "session_seconds": time.perf_counter() - started,
                **collection,
                **metrics,
            }
            # Commit model/RNG and per-cohort records together before regenerating
            # convenience cumulative logs. Interruption cannot duplicate cohorts.
            state["last_update"] = record
            state["last_episodes"] = [dict(cohort=cohort + 1, **r) for r in episodes]
            storage.save(output / f"checkpoint_{cohort + 1:04d}", policy, state)
            refresh_logs(output, parent["cohort"], state["cohort"])
            atomic_json(
                root / "status.json",
                {"status": "TRAINING", "arm": arm, "pid": os.getpid(), **record},
            )
            print(json.dumps({"arm": arm, **record}), flush=True)
        refresh_logs(output, parent["cohort"], state["cohort"])
        if storage.PAUSED or (root / "PAUSE").exists():
            return False
        atomic_json(
            root / "status.json",
            {"status": "EVALUATING", "arm": arm, "pid": os.getpid()},
        )
        evaluation = {
            mode: storage.evaluation(
                policy,
                vec,
                output,
                state["cohort"],
                args.eval_tasks,
                mode == "deterministic",
            )
            for mode in ("deterministic", "stochastic")
        }
        complete = all(r["complete"] for r in evaluation.values())
        atomic_json(
            output / "RESULT.json",
            {
                "status": "COMPLETE" if complete else "PAUSED",
                "cohort": state["cohort"],
                "extra_cohorts": args.extra_cohorts,
                "total_transitions": state["transitions"],
                "new_transitions": state["transitions"] - parent["transitions"],
                "multiplier": state["multiplier"],
                "cost_actor_trace": trace,
                "fork_policy_hash": state["fork_policy_hash"],
                "evaluation": evaluation,
                "session_seconds": time.perf_counter() - started,
            },
        )
        return complete
    finally:
        vec.close()


def refresh_logs(output: Path, origin: int, current: int) -> None:
    updates, episodes = [], []
    for cohort in range(origin + 1, current + 1):
        metadata = json.loads(
            (output / f"checkpoint_{cohort:04d}/metadata.json").read_text()
        )
        updates.append(metadata["last_update"])
        episodes.extend(metadata["last_episodes"])
    for filename, records in (
        ("updates.jsonl", updates),
        ("training_episodes.jsonl", episodes),
    ):
        temporary = output / (filename + ".tmp")
        temporary.write_text("".join(json.dumps(r) + "\n" for r in records))
        temporary.replace(output / filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Completed slowdual checkpoint directory",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--extra-cohorts", type=int, default=32)
    parser.add_argument("--eval-tasks", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.extra_cohorts <= 0 or args.eval_tasks < 0:
        raise ValueError("invalid budget")
    torch.set_num_threads(1)
    signal.signal(signal.SIGTERM, storage.pause)
    signal.signal(signal.SIGINT, storage.pause)
    parent = json.loads((args.source / "metadata.json").read_text())
    for name, digest in parent["contract"]["source_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"parent source modified: {name}")
    source_files = [
        Path(__file__),
        ROOT / "experiments/directional_navigation/cost_trace.py",
    ]
    contract = {
        "protocol": "matched_cost_actor_trace_continuation_v2",
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "parent_contract": parent["contract"],
        "parent_checkpoint_sha256": hashlib.sha256(
            (args.source / "state.pt").read_bytes()
        ).hexdigest(),
        "parent_cohort": parent["cohort"],
        "trace_by_arm": {"gae95": 0.95, "mc_control": 1.0},
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_files
        },
        "eval_tasks": args.eval_tasks,
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
                "source": str(args.source),
                "budget": "same additional task cohorts, unequal physical steps",
                "note": "Both adaptive slowdual; neither arm is lambda-zero. Only actor cost trace changes; critic targets stay MC.",
            },
        )
    started = time.perf_counter()
    atomic_json(root / "status.json", {"status": "STARTING", "pid": os.getpid()})
    try:
        for arm, trace in contract["trace_by_arm"].items():
            if not run_arm(args, root, contract, arm, trace):
                atomic_json(root / "status.json", {"status": "PAUSED", "arm": arm})
                return
        results = {
            arm: json.loads((root / arm / "RESULT.json").read_text())
            for arm in contract["trace_by_arm"]
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
