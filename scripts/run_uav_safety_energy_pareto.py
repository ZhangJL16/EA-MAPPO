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
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor
from scripts.train_uav_energy_mc import load_frozen_sac


DEFAULT_SAC = (
    ROOT
    / "artifacts/uav_energy_delivery_v3_formal_20260816_004619"
    / "phase1_navigation/checkpoint_transition_500000.zip"
)
DEFAULT_ENERGY_MODEL = (
    ROOT
    / "artifacts/uav_energy_delivery_mc_formal_20260817_154321"
    / "energy_model/best_validation.pt"
)
DEFAULT_GRADIENT_AUDIT = (
    ROOT / "artifacts/uav_safety_energy_action_gradient_audit_20260819_020407.json"
)


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_shard(
    scenarios: list,
    checkpoint: str,
    energy_model: str,
    device: str,
    weights: tuple[float, ...],
    energy_gradient_characteristic: float,
    energy_gradient_normalization: str,
    torch_threads: int,
) -> list:
    torch.set_num_threads(torch_threads)
    policy = load_frozen_sac(Path(checkpoint), device)
    estimator = EnergyToGoRegressor.load(Path(energy_model), device=device)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    results = []
    for scenario in scenarios:
        for method in (
            "frozen_sac_no_filter",
            SafetyFilterMethod.HOCBF.value,
            SafetyFilterMethod.SAMPLED_DATA_HOCBF.value,
        ):
            results.append(run_rollout(policy_action, scenario, method))
        for weight in weights:
            for method in (
                SafetyFilterMethod.ENERGY_AWARE_HOCBF.value,
                SafetyFilterMethod.ENERGY_GRADIENT_HOCBF.value,
                SafetyFilterMethod.SAMPLED_DATA_ENERGY_AWARE_HOCBF.value,
                SafetyFilterMethod.SAMPLED_DATA_ENERGY_GRADIENT_HOCBF.value,
            ):
                result = run_rollout(
                    policy_action,
                    scenario,
                    method,
                    energy_weight=weight,
                    energy_gradient_weight=weight,
                    energy_gradient_characteristic=energy_gradient_characteristic,
                    energy_gradient_normalization=energy_gradient_normalization,
                    energy_estimator=estimator,
                )
                results.append(
                    replace(result, method=f"{method}@weight={weight:g}")
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safety-energy-performance Pareto benchmark"
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sac-checkpoint", default=str(DEFAULT_SAC))
    parser.add_argument("--energy-model", default=str(DEFAULT_ENERGY_MODEL))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollouts", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=(0.01, 0.03, 0.1, 0.3, 1.0),
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--energy-gradient-normalization",
        choices=("fixed", "unit_direction"),
        default="fixed",
    )
    parser.add_argument("--energy-gradient-characteristic", type=float)
    parser.add_argument("--gradient-audit", default=str(DEFAULT_GRADIENT_AUDIT))
    args = parser.parse_args()
    weights = tuple(float(value) for value in args.weights)
    if (
        args.rollouts <= 0
        or args.workers <= 0
        or args.torch_threads <= 0
        or not weights
        or any(value <= 0.0 for value in weights)
    ):
        raise ValueError("rollout, worker, thread, and weight values must be positive")
    if args.workers > 1 and args.device != "cpu":
        raise ValueError("multi-worker Pareto benchmark currently requires CPU")
    gradient_characteristic = args.energy_gradient_characteristic
    gradient_characteristic_source = "cli"
    if gradient_characteristic is None:
        audit = json.loads(Path(args.gradient_audit).read_text())
        gradient_characteristic = float(
            audit["action_gradient_norm"]["p95"] * np.hypot(5.0, 3.0)
        )
        gradient_characteristic_source = str(Path(args.gradient_audit).resolve())
    if not np.isfinite(gradient_characteristic) or gradient_characteristic <= 0.0:
        raise ValueError("energy gradient characteristic must be finite and positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = make_scenarios(args.rollouts, args.seed)
    if args.workers == 1:
        results = run_shard(
            scenarios,
            args.sac_checkpoint,
            args.energy_model,
            args.device,
            weights,
            gradient_characteristic,
            args.energy_gradient_normalization,
            args.torch_threads,
        )
    else:
        shards = [scenarios[index :: args.workers] for index in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = executor.map(
                run_shard,
                shards,
                [args.sac_checkpoint] * args.workers,
                [args.energy_model] * args.workers,
                [args.device] * args.workers,
                [weights] * args.workers,
                [gradient_characteristic] * args.workers,
                [args.energy_gradient_normalization] * args.workers,
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
            "purpose": "controlled Pareto ablation; not a formal result",
            "rollouts": args.rollouts,
            "scenario_method_rollouts": len(results),
            "seed": args.seed,
            "weights": weights,
            "energy_gradient_characteristic": gradient_characteristic,
            "energy_gradient_characteristic_source": gradient_characteristic_source,
            "energy_gradient_normalization": args.energy_gradient_normalization,
            "sac_checkpoint": str(Path(args.sac_checkpoint).resolve()),
            "energy_model": str(Path(args.energy_model).resolve()),
            "device": args.device,
            "workers": args.workers,
            "torch_threads": args.torch_threads,
        },
    )


if __name__ == "__main__":
    main()
