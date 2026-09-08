from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from review_bundle.safety.energy.mc_regression import (
    ModelBasedEnergyRolloutEstimator,
    ModelBasedRolloutError,
)
from scripts.diagnose_return_decision_stage_b import diagnostic_args
from scripts.run_return_decision_stage_b import (
    FrozenPolicy,
    environment_from_args,
    load_policy,
    load_prerequisite_audit,
)
from scripts.train_uav_energy_delivery_sac import HeuristicGoalPolicy


_WORKER_ARGS: argparse.Namespace | None = None
_WORKER_FROZEN_POLICY: FrozenPolicy | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def initialize_worker(args: argparse.Namespace) -> None:
    global _WORKER_ARGS, _WORKER_FROZEN_POLICY
    os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    _WORKER_ARGS = args
    probe = environment_from_args(args, reserve_fraction=0.0)
    _WORKER_FROZEN_POLICY = load_policy(args, probe)
    probe.close()


def obstacle_fingerprint(environment) -> str:
    payload = [
        [float(obstacle.pos[0]), float(obstacle.pos[1]), float(obstacle.radius)]
        for obstacle in environment.obstacles
    ]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def run_rollout_condition(
    environment,
    *,
    policy,
    cbf_enabled: bool,
    start_position: np.ndarray,
    start_velocity: np.ndarray,
    goal: np.ndarray,
    rollout_label: str,
) -> dict[str, object]:
    if _WORKER_ARGS is None:
        raise RuntimeError("stall-analysis worker is not initialized")
    environment.cbf_enabled = bool(cbf_enabled)
    estimator = ModelBasedEnergyRolloutEstimator(
        policy,
        max_policy_steps=_WORKER_ARGS.oracle_max_policy_steps,
        cache_mission_suffixes=False,
    )
    started = time.perf_counter()
    try:
        prediction = estimator.estimate_context(
            environment,
            goal,
            position=start_position,
            velocity=start_velocity,
            goal_type=rollout_label,
        )
    except ModelBasedRolloutError as error:
        return {
            "status": "FAIL_ROLLOUT",
            "success": False,
            "wall_clock_seconds": float(time.perf_counter() - started),
            "error_type": type(error).__name__,
            "error": str(error),
            "rollout_diagnostics": error.rollout_diagnostics,
        }
    if not prediction.deadline_feasible:
        return {
            "status": "FAIL_DEADLINE_INFEASIBLE",
            "success": False,
            "wall_clock_seconds": float(time.perf_counter() - started),
            "completion_status": prediction.completion_status,
            "truncated_energy": float(prediction.prediction),
            "rollout_steps": int(prediction.rollout_steps),
            "rollout_diagnostics": prediction.rollout_diagnostics,
        }
    return {
        "status": "SUCCESS_GOAL_REACHED",
        "success": True,
        "wall_clock_seconds": float(time.perf_counter() - started),
        "energy": float(prediction.prediction),
        "rollout_steps": int(prediction.rollout_steps),
        "simulated_seconds": float(
            prediction.rollout_steps * environment.policy_dt
        ),
    }


def analyze_failure_state(failure_path_string: str) -> str:
    if _WORKER_ARGS is None or _WORKER_FROZEN_POLICY is None:
        raise RuntimeError("stall-analysis worker is not initialized")
    failure_path = Path(failure_path_string)
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    seed = int(failure["evaluation_seed"])
    observed = failure["rollout_diagnostics"]
    start_position = np.asarray(observed["start_position"], dtype=np.float32)
    start_velocity = np.asarray(observed["start_velocity"], dtype=np.float32)
    goal = np.asarray(observed["goal"], dtype=np.float32)
    rollout_label = str(observed["rollout_label"])

    environment = environment_from_args(_WORKER_ARGS, reserve_fraction=0.0)
    environment.bind_keyed_task_schedule(seed)
    environment.reset(seed=seed)
    conditions: dict[str, dict[str, object]] = {
        "frozen_sac_hocbf_observed": {
            "status": "FAIL_ROLLOUT",
            "success": False,
            "source": str(failure_path.resolve()),
            "rollout_diagnostics": observed,
        }
    }
    try:
        conditions["frozen_sac_no_hocbf"] = run_rollout_condition(
            environment,
            policy=_WORKER_FROZEN_POLICY,
            cbf_enabled=False,
            start_position=start_position,
            start_velocity=start_velocity,
            goal=goal,
            rollout_label=rollout_label,
        )
        conditions["heuristic_hocbf"] = run_rollout_condition(
            environment,
            policy=HeuristicGoalPolicy(),
            cbf_enabled=True,
            start_position=start_position,
            start_velocity=start_velocity,
            goal=goal,
            rollout_label=rollout_label,
        )
        conditions["heuristic_no_hocbf"] = run_rollout_condition(
            environment,
            policy=HeuristicGoalPolicy(),
            cbf_enabled=False,
            start_position=start_position,
            start_velocity=start_velocity,
            goal=goal,
            rollout_label=rollout_label,
        )
        result = {
            "protocol": "paired_return_rollout_stall_ablation_v1",
            "formal_evidence": False,
            "evaluation_seed": seed,
            "rollout_label": rollout_label,
            "start_position": start_position.astype(np.float64).tolist(),
            "start_velocity": start_velocity.astype(np.float64).tolist(),
            "goal": goal.astype(np.float64).tolist(),
            "obstacle_count": len(environment.obstacles),
            "obstacle_fingerprint": obstacle_fingerprint(environment),
            "oracle_max_policy_steps": int(_WORKER_ARGS.oracle_max_policy_steps),
            "conditions": conditions,
        }
    finally:
        environment.close()
    output = _WORKER_ARGS.output_dir / "seed_results" / f"seed_{seed}.json"
    atomic_write_json(output, result)
    return str(output)


def summarize_results(results: list[dict[str, object]]) -> dict[str, object]:
    condition_names = list(results[0]["conditions"])
    summaries: dict[str, dict[str, object]] = {}
    for condition in condition_names:
        rows = [result["conditions"][condition] for result in results]
        successes = [row for row in rows if row["success"] is True]
        steps = [int(row["rollout_steps"]) for row in successes]
        summaries[condition] = {
            "num_states": len(rows),
            "success_count": len(successes),
            "failure_count": len(rows) - len(successes),
            "success_fraction": float(len(successes) / len(rows)),
            "successful_rollout_steps_mean": (
                None if not steps else float(statistics.mean(steps))
            ),
            "successful_rollout_steps_median": (
                None if not steps else float(statistics.median(steps))
            ),
            "successful_rollout_steps_max": None if not steps else max(steps),
        }
    paired_patterns: dict[str, int] = {}
    for result in results:
        pattern = "|".join(
            f"{name}={int(bool(result['conditions'][name]['success']))}"
            for name in condition_names
        )
        paired_patterns[pattern] = paired_patterns.get(pattern, 0) + 1
    return {
        "condition_summaries": summaries,
        "paired_success_patterns": paired_patterns,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired policy-by-HOCBF causal diagnostic for Oracle stalls"
    )
    parser.add_argument("--source-launch", type=Path, required=True)
    parser.add_argument("--source-diagnostic-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args(argv)
    if args.workers <= 0:
        parser.error("--workers must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    if cli.output_dir.exists():
        raise FileExistsError(f"output already exists: {cli.output_dir}")
    failure_paths = sorted(
        (cli.source_diagnostic_dir / "seed_failures").glob("seed_*.json")
    )
    if not failure_paths:
        raise FileNotFoundError("source diagnostic contains no seed failures")
    cli.output_dir.mkdir(parents=True)
    seeds = [int(path.stem.removeprefix("seed_")) for path in failure_paths]
    args = diagnostic_args(cli.source_launch, cli.output_dir, seeds)
    args.evaluation_num_envs = min(cli.workers, len(seeds))
    prerequisite = load_prerequisite_audit(args)
    if not prerequisite.passed:
        raise RuntimeError("source configuration no longer passes prerequisites")
    manifest = {
        "protocol": "paired_return_rollout_stall_ablation_v1",
        "formal_evidence": False,
        "source_launch": str(cli.source_launch.resolve()),
        "source_launch_sha256": file_sha256(cli.source_launch),
        "source_diagnostic_dir": str(cli.source_diagnostic_dir.resolve()),
        "source_failure_sha256": {
            path.name: file_sha256(path) for path in failure_paths
        },
        "evaluation_seeds": seeds,
        "worker_count": int(args.evaluation_num_envs),
        "torch_threads": int(args.torch_threads),
        "oracle_max_policy_steps": int(args.oracle_max_policy_steps),
        "paired_conditions": [
            "frozen_sac_hocbf_observed",
            "frozen_sac_no_hocbf",
            "heuristic_hocbf",
            "heuristic_no_hocbf",
        ],
    }
    atomic_write_json(cli.output_dir / "manifest.json", manifest)
    atomic_write_json(
        cli.output_dir / "prerequisite_audit.json",
        prerequisite.as_dict(),
    )
    context = mp.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=args.evaluation_num_envs,
            mp_context=context,
            initializer=initialize_worker,
            initargs=(args,),
        ) as executor:
            futures = {
                executor.submit(analyze_failure_state, str(path)): path
                for path in failure_paths
            }
            for future in concurrent.futures.as_completed(futures):
                print(future.result(), flush=True)
        results = [
            json.loads(
                (cli.output_dir / "seed_results" / f"seed_{seed}.json").read_text(
                    encoding="utf-8"
                )
            )
            for seed in seeds
        ]
        summary = {
            **manifest,
            "status": "COMPLETED",
            **summarize_results(results),
        }
        atomic_write_json(cli.output_dir / "COMPLETED.json", summary)
    except Exception as error:
        atomic_write_json(
            cli.output_dir / "FAILED.json",
            {
                **manifest,
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


if __name__ == "__main__":
    main()
