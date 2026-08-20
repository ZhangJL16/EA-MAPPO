from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.uav_safety_filter.benchmark import (
    make_scenarios,
    result_dict,
    run_rollout,
    summarize,
)
from review_bundle.safety.collision.filter import SafetyFilterMethod
from scripts.train_uav_energy_mc import load_frozen_sac


DEFAULT_SAC = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_formal_20260816_004619"
    / "phase1_navigation/checkpoint_transition_500000.zip"
)
CONDITIONS = (
    {
        "name": "exact_10hz",
        "range_noise_std": 0.0,
        "obstacle_dropout_probability": 0.0,
        "lidar_period_steps": 2,
        "perception_uncertainty_margin": 0.0,
    },
    {
        "name": "bounded_noise_unprotected",
        "range_noise_std": 0.5,
        "obstacle_dropout_probability": 0.0,
        "lidar_period_steps": 2,
        "perception_uncertainty_margin": 0.0,
    },
    {
        "name": "bounded_noise_margin_1p5m",
        "range_noise_std": 0.5,
        "obstacle_dropout_probability": 0.0,
        "lidar_period_steps": 2,
        "perception_uncertainty_margin": 1.5,
    },
    {
        "name": "obstacle_dropout_10pct",
        "range_noise_std": 0.0,
        "obstacle_dropout_probability": 0.1,
        "lidar_period_steps": 2,
        "perception_uncertainty_margin": 0.0,
    },
    {
        "name": "slow_lidar_5hz",
        "range_noise_std": 0.0,
        "obstacle_dropout_probability": 0.0,
        "lidar_period_steps": 4,
        "perception_uncertainty_margin": 0.0,
    },
    {
        "name": "combined_stress",
        "range_noise_std": 0.5,
        "obstacle_dropout_probability": 0.1,
        "lidar_period_steps": 4,
        "perception_uncertainty_margin": 1.5,
    },
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_shard(
    scenarios: list,
    checkpoint: str,
    device: str,
    energy_weight: float,
    seed: int,
    torch_threads: int,
) -> list:
    torch.set_num_threads(torch_threads)
    policy = load_frozen_sac(Path(checkpoint), device)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    results = []
    for scenario_index, scenario in enumerate(scenarios):
        for baseline in ("frozen_sac_no_filter", SafetyFilterMethod.HOCBF.value):
            results.append(run_rollout(policy_action, scenario, baseline))
        for condition_index, condition in enumerate(CONDITIONS):
            perception_seed = seed + 10_000 * condition_index + scenario_index
            for method in (
                SafetyFilterMethod.SAMPLED_DATA_HOCBF.value,
                SafetyFilterMethod.SAMPLED_DATA_ENERGY_AWARE_HOCBF.value,
            ):
                result = run_rollout(
                    policy_action,
                    scenario,
                    method,
                    energy_weight=energy_weight,
                    range_noise_std=condition["range_noise_std"],
                    obstacle_dropout_probability=condition[
                        "obstacle_dropout_probability"
                    ],
                    lidar_period_steps=condition["lidar_period_steps"],
                    perception_uncertainty_margin=condition[
                        "perception_uncertainty_margin"
                    ],
                    perception_seed=perception_seed,
                )
                results.append(
                    replace(result, method=f"{method}@{condition['name']}")
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Short LiDAR robustness benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sac-checkpoint", default=str(DEFAULT_SAC))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollouts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    if args.rollouts <= 0 or args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("rollout, worker, and thread counts must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("multi-worker robustness benchmark requires CPU")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = make_scenarios(args.rollouts, args.seed)
    if args.workers == 1:
        results = run_shard(
            scenarios,
            args.sac_checkpoint,
            args.device,
            args.energy_weight,
            args.seed + 300_000,
            args.torch_threads,
        )
    else:
        shards = [scenarios[index :: args.workers] for index in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = executor.map(
                run_shard,
                shards,
                [args.sac_checkpoint] * args.workers,
                [args.device] * args.workers,
                [args.energy_weight] * args.workers,
                [args.seed + 300_000 + index * 100_000 for index in range(args.workers)],
                [args.torch_threads] * args.workers,
            )
            results = [row for part in parts for row in part]

    rows = [result_dict(row) for row in results]
    with (output / "rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "summary.json", summarize(results))
    write_json(
        output / "config.json",
        {
            "purpose": "short sensor robustness diagnosis; not a formal result",
            "rollouts": args.rollouts,
            "scenario_method_rollouts": len(results),
            "seed": args.seed,
            "energy_weight": args.energy_weight,
            "conditions": CONDITIONS,
            "range_noise_is_clipped_at_sigma": 3.0,
            "device": args.device,
            "workers": args.workers,
        },
    )


if __name__ == "__main__":
    main()
