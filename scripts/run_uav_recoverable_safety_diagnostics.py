from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.uav_recoverable_safety.diagnostics import (
    PerceptionMode,
    RecoverabilityDiagnosticConfig,
    make_hand_scenarios,
    make_random_hard_scenarios,
    record_to_dict,
    run_scenario,
    search_counterexamples,
)
from scripts.train_uav_energy_mc import load_frozen_sac


DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_formal_20260816_004619"
    / "phase1_navigation/checkpoint_transition_500000.zip"
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_shard(
    scenarios: list,
    mode_values: tuple[str, ...],
    checkpoint: str,
    device: str,
    config_payload: dict[str, object],
    save_trajectories: bool,
) -> dict[str, object]:
    import torch

    torch.set_num_threads(1)
    policy = load_frozen_sac(Path(checkpoint), device)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    config = RecoverabilityDiagnosticConfig(**config_payload)
    modes = tuple(PerceptionMode(value) for value in mode_values)
    counterexamples = []
    scenario_summaries = []
    trajectories = []
    total_steps = 0
    total_filter_seconds = 0.0
    total_rho_seconds = 0.0
    filter_times = []
    rho_times = []
    for scenario in scenarios:
        for mode in modes:
            records = run_scenario(
                scenario,
                policy_action,
                perception_mode=mode,
                config=config,
            )
            found = search_counterexamples(
                scenario,
                records,
                perception_mode=mode,
                config=config,
            )
            counterexamples.extend(found)
            total_steps += len(records)
            total_filter_seconds += sum(record.filter_seconds for record in records)
            total_rho_seconds += sum(record.rho_seconds for record in records)
            filter_times.extend(record.filter_seconds for record in records)
            rho_times.extend(record.rho_seconds for record in records)
            scenario_summaries.append(
                {
                    "scenario_id": scenario.identifier,
                    "family": scenario.family,
                    "perception_mode": mode.value,
                    "steps": len(records),
                    "counterexamples": len(found),
                    "minimum_true_rho": min(
                        record.rho_true
                        for record in records
                        if record.rho_true is not None
                    ),
                    "feasible_to_infeasible": bool(found),
                    "current_infeasible_steps": sum(
                        not record.true_feasible for record in records
                    ),
                    "constraint_disappearance_steps": sum(
                        record.rho_perceived is None and record.rho_true is not None
                        for record in records
                    ),
                    "perceived_feasible_true_infeasible_steps": sum(
                        record.perceived_feasible and not record.true_feasible
                        for record in records
                    ),
                    "braking_authority_loss_steps": sum(
                        record.required_braking > record.available_braking
                        for record in records
                    ),
                    "collision_steps": sum(
                        record.minimum_clearance <= 0.0 for record in records
                    ),
                    "near_collision_steps": sum(
                        record.minimum_clearance <= 2.0 for record in records
                    ),
                    "minimum_clearance": min(
                        record.minimum_clearance for record in records
                    ),
                    "fallback_steps": sum(record.fallback_used for record in records),
                    "fallback_unsafe_steps": sum(
                        record.fallback_used
                        and not record.fallback_satisfies_constraints
                        for record in records
                    ),
                    "goal_progress": float(
                        np.linalg.norm(scenario.goal - scenario.start)
                        - np.linalg.norm(
                            scenario.goal - np.asarray(records[-1].position)
                        )
                    ),
                }
            )
            if save_trajectories:
                trajectories.append(
                    {
                        "scenario_id": scenario.identifier,
                        "family": scenario.family,
                        "perception_mode": mode.value,
                        "records": [record_to_dict(record) for record in records],
                    }
                )
    return {
        "counterexamples": counterexamples,
        "scenario_summaries": scenario_summaries,
        "trajectories": trajectories,
        "total_steps": total_steps,
        "total_filter_seconds": total_filter_seconds,
        "total_rho_seconds": total_rho_seconds,
        "filter_times": filter_times,
        "rho_times": rho_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search feasible-to-infeasible UAV safety-filter transitions"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--random-scenarios", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lookahead-steps", type=int, default=12)
    parser.add_argument(
        "--perception-modes",
        nargs="+",
        choices=[mode.value for mode in PerceptionMode],
        default=[mode.value for mode in PerceptionMode],
    )
    parser.add_argument("--hand-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--save-trajectories", action="store_true")
    args = parser.parse_args()
    if args.random_scenarios <= 0 or args.lookahead_steps <= 0 or args.workers <= 0:
        raise ValueError("scenario count, lookahead, and worker count must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("parallel diagnostics require CPU policy inference")
    if args.workers > 1 and args.save_trajectories:
        raise ValueError("full trajectory capture is restricted to one worker")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = Path(args.checkpoint).resolve()

    scenarios = make_hand_scenarios()
    if not args.hand_only:
        scenarios.extend(make_random_hard_scenarios(args.random_scenarios, args.seed))
    config = RecoverabilityDiagnosticConfig(lookahead_steps=args.lookahead_steps)
    modes = tuple(PerceptionMode(value) for value in args.perception_modes)
    shards = [scenarios[index :: args.workers] for index in range(args.workers)]
    common_arguments = (
        tuple(mode.value for mode in modes),
        str(checkpoint),
        args.device,
        asdict(config),
        args.save_trajectories,
    )
    if args.workers == 1:
        shard_results = [run_shard(shards[0], *common_arguments)]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(run_shard, shard, *common_arguments)
                for shard in shards
            ]
            shard_results = [future.result() for future in futures]

    counterexamples = [
        record
        for result in shard_results
        for record in result["counterexamples"]
    ]
    scenario_summaries = [
        row for result in shard_results for row in result["scenario_summaries"]
    ]
    trajectories = [
        row for result in shard_results for row in result["trajectories"]
    ]
    total_steps = sum(int(result["total_steps"]) for result in shard_results)
    total_filter_seconds = sum(
        float(result["total_filter_seconds"]) for result in shard_results
    )
    total_rho_seconds = sum(
        float(result["total_rho_seconds"]) for result in shard_results
    )
    filter_times = np.asarray(
        [value for result in shard_results for value in result["filter_times"]],
        dtype=np.float64,
    )
    rho_times = np.asarray(
        [value for result in shard_results for value in result["rho_times"]],
        dtype=np.float64,
    )
    combined_times = filter_times + rho_times
    recovery_times = np.asarray(
        [record.sampled_recovery_search_seconds for record in counterexamples],
        dtype=np.float64,
    )
    with (output / "hard_states.jsonl").open("w") as handle:
        for record in counterexamples:
            handle.write(json.dumps(record_to_dict(record), sort_keys=True) + "\n")
    with (output / "trajectories.jsonl").open("w") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(trajectory, sort_keys=True) + "\n")

    causes = Counter(record.cause for record in counterexamples)
    families = Counter(record.family for record in counterexamples)
    mode_counts = Counter(record.perception_mode for record in counterexamples)
    summary = {
        "scenario_count": len(scenarios),
        "perception_modes": [mode.value for mode in modes],
        "scenario_mode_rollouts": len(scenario_summaries),
        "total_steps": total_steps,
        "counterexample_count": len(counterexamples),
        "counterexamples_with_sampled_recovery_action": sum(
            record.sampled_recovery_action_exists for record in counterexamples
        ),
        "rollouts_with_counterexample": sum(
            row["feasible_to_infeasible"] for row in scenario_summaries
        ),
        "counterexamples_by_cause": dict(causes),
        "counterexamples_by_family": dict(families),
        "counterexamples_by_perception_mode": dict(mode_counts),
        "current_infeasible_steps": sum(
            row["current_infeasible_steps"] for row in scenario_summaries
        ),
        "constraint_disappearance_steps": sum(
            row["constraint_disappearance_steps"] for row in scenario_summaries
        ),
        "perceived_feasible_true_infeasible_steps": sum(
            row["perceived_feasible_true_infeasible_steps"]
            for row in scenario_summaries
        ),
        "braking_authority_loss_steps": sum(
            row["braking_authority_loss_steps"] for row in scenario_summaries
        ),
        "collision_rollouts": sum(
            row["collision_steps"] > 0 for row in scenario_summaries
        ),
        "collision_steps": sum(row["collision_steps"] for row in scenario_summaries),
        "near_collision_steps": sum(
            row["near_collision_steps"] for row in scenario_summaries
        ),
        "minimum_clearance": min(
            row["minimum_clearance"] for row in scenario_summaries
        ),
        "fallback_steps": sum(row["fallback_steps"] for row in scenario_summaries),
        "fallback_unsafe_steps": sum(
            row["fallback_unsafe_steps"] for row in scenario_summaries
        ),
        "mean_goal_progress": float(
            np.mean([row["goal_progress"] for row in scenario_summaries])
        ),
        "mean_filter_seconds": total_filter_seconds / max(total_steps, 1),
        "mean_rho_seconds": total_rho_seconds / max(total_steps, 1),
        "combined_mean_diagnostic_seconds": (
            total_filter_seconds + total_rho_seconds
        )
        / max(total_steps, 1),
        "twenty_hz_deadline_seconds": 0.05,
        "diagnostic_within_twenty_hz_mean": (
            total_filter_seconds + total_rho_seconds
        )
        / max(total_steps, 1)
        <= 0.05,
        "filter_latency_seconds": {
            "median": float(np.median(filter_times)),
            "p90": float(np.quantile(filter_times, 0.90)),
            "p95": float(np.quantile(filter_times, 0.95)),
            "p99": float(np.quantile(filter_times, 0.99)),
            "max": float(np.max(filter_times)),
        },
        "rho_diagnostic_latency_seconds": {
            "median": float(np.median(rho_times)),
            "p90": float(np.quantile(rho_times, 0.90)),
            "p95": float(np.quantile(rho_times, 0.95)),
            "p99": float(np.quantile(rho_times, 0.99)),
            "max": float(np.max(rho_times)),
        },
        "combined_diagnostic_latency_seconds": {
            "median": float(np.median(combined_times)),
            "p90": float(np.quantile(combined_times, 0.90)),
            "p95": float(np.quantile(combined_times, 0.95)),
            "p99": float(np.quantile(combined_times, 0.99)),
            "max": float(np.max(combined_times)),
        },
        "sampled_recovery_search_latency_seconds": (
            None
            if recovery_times.size == 0
            else {
                "mean": float(np.mean(recovery_times)),
                "median": float(np.median(recovery_times)),
                "p95": float(np.quantile(recovery_times, 0.95)),
                "max": float(np.max(recovery_times)),
            }
        ),
        "go_condition_failure_is_stable": len(counterexamples) > 0,
    }
    write_json(output / "summary.json", summary)
    write_json(output / "scenario_summary.json", scenario_summaries)
    git_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    write_json(
        output / "run_manifest.json",
        {
            "exact_command": " ".join(sys.argv),
            "git_sha": git_sha,
            "source_sha256": sha256(Path(__file__)),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "seed": args.seed,
            "config": record_to_dict(config),
            "workers": args.workers,
            "full_trajectories_saved": args.save_trajectories,
            "policy_frozen": True,
            "neural_training": False,
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
