#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import numpy as np
import torch

from experiments.uav_safety_filter.benchmark import (
    RolloutResult,
    make_scenarios,
    result_dict,
    run_rollout,
    summarize,
)
from scripts.train_uav_energy_mc import load_frozen_sac


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAC = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_formal_20260816_004619"
    / "phase1_navigation/checkpoint_transition_500000.zip"
)
METHODS = (
    "sampled_data_hocbf",
    "sampled_data_energy_aware_hocbf",
    "sampled_data_lexicographic_energy_hocbf",
)

INTEGER_FIELDS = {
    "collision_steps",
    "near_collision_steps",
    "hocbf_violation_steps",
    "infeasible_steps",
    "fallback_unsafe_steps",
    "deadline_misses",
    "feasibility_margin_negative_steps",
    "steps",
    "intervention_steps",
    "max_consecutive_intervention",
    "max_candidate_obstacles",
}


def load_rollouts(path: Path) -> list[RolloutResult]:
    results = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, object] = {}
            for name, value in row.items():
                if name in {"method", "scenario_id", "family"}:
                    converted[name] = value
                elif name == "success":
                    converted[name] = value == "True"
                elif name in INTEGER_FIELDS:
                    converted[name] = int(value)
                elif name == "minimum_feasibility_margin" and value == "":
                    converted[name] = None
                else:
                    converted[name] = float(value)
            results.append(RolloutResult(**converted))
    return results


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_shard(
    scenarios: list,
    checkpoint: str,
    device: str,
    energy_weight: float,
    torch_threads: int,
) -> list:
    torch.set_num_threads(torch_threads)
    policy = load_frozen_sac(Path(checkpoint), device)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return [
        run_rollout(
            policy_action,
            scenario,
            method,
            energy_weight=energy_weight,
        )
        for scenario in scenarios
        for method in METHODS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sac-checkpoint", type=Path, default=DEFAULT_SAC)
    parser.add_argument("--scenario-count", type=int, default=334)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-a-energy-weight", type=float, default=0.1)
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()
    if args.scenario_count <= 0 or args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("scenario, worker, and thread counts must be positive")
    method_trajectories = args.scenario_count * len(METHODS)
    if method_trajectories < 1_000:
        raise ValueError("the adversarial protocol requires at least 1000 method trajectories")
    if args.device != "cpu" and args.workers > 1:
        raise ValueError("multi-worker validation requires CPU inference")
    scenarios = make_scenarios(args.scenario_count, args.seed)
    if args.summarize_existing:
        results = load_rollouts(args.output_dir / "rollouts.csv")
        if len(results) != method_trajectories:
            raise ValueError(
                f"expected {method_trajectories} existing rollouts, found {len(results)}"
            )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        shards = [scenarios[index :: args.workers] for index in range(args.workers)]
        if args.workers == 1:
            results = run_shard(
                scenarios,
                str(args.sac_checkpoint),
                args.device,
                args.candidate_a_energy_weight,
                args.torch_threads,
            )
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                parts = executor.map(
                    run_shard,
                    shards,
                    [str(args.sac_checkpoint)] * args.workers,
                    [args.device] * args.workers,
                    [args.candidate_a_energy_weight] * args.workers,
                    [args.torch_threads] * args.workers,
                )
                results = [row for part in parts for row in part]
        rows = [result_dict(row) for row in results]
        with (args.output_dir / "rollouts.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = summarize(results)
    summary["protocol"] = {
        "scenario_count": args.scenario_count,
        "methods": list(METHODS),
        "method_trajectories": len(results),
        "families": sorted({scenario.family for scenario in scenarios}),
        "candidate_a_energy_weight": args.candidate_a_energy_weight,
        "formal_500k": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    write_json(
        args.output_dir / "config.json",
        {
            "scenario_count": args.scenario_count,
            "method_trajectories": method_trajectories,
            "seed": args.seed,
            "workers": args.workers,
            "torch_threads": args.torch_threads,
            "device": args.device,
            "sac_checkpoint": str(args.sac_checkpoint.resolve()),
            "methods": list(METHODS),
            "candidate_a_energy_weight": args.candidate_a_energy_weight,
            "purpose": "theory falsification; not a formal training result",
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
