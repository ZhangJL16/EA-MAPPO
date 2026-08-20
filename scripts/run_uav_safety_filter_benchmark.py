from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter, process_time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.uav_safety_filter.benchmark import (
    compare_top_k_priorities,
    compute_scaling,
    make_scenarios,
    result_dict,
    run_rollout,
    summarize,
)
from scripts.train_uav_energy_mc import load_frozen_sac
from review_bundle.safety.energy.mc_regression import EnergyToGoRegressor

DEFAULT_SAC = ROOT / "artifacts/uav_energy_delivery_v3_formal_20260816_004619/phase1_navigation/checkpoint_transition_500000.zip"
DEFAULT_ENERGY_MODEL = ROOT / "artifacts/uav_energy_delivery_mc_formal_20260817_154321/energy_model/best_validation.pt"
DEFAULT_GRADIENT_AUDIT = (
    ROOT / "artifacts/uav_safety_energy_action_gradient_audit_20260819_020407.json"
)
METHODS = [
    "frozen_sac_no_filter",
    "one_step_projection",
    "second_order_hocbf",
    "aggregate_hocbf",
    "energy_aware_hocbf",
    "energy_gradient_hocbf",
    "sampled_data_hocbf",
    "sampled_data_energy_aware_hocbf",
    "sampled_data_energy_gradient_hocbf",
    "sampled_data_lexicographic_energy_hocbf",
]


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def source_hash(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def run_scenario_shard(
    scenarios: list,
    checkpoint: str,
    device: str,
    energy_weight: float,
    energy_gradient_weight: float,
    energy_gradient_characteristic: float,
    energy_gradient_normalization: str,
    energy_model: str,
    torch_threads: int,
) -> list:
    torch.set_num_threads(torch_threads)
    policy = load_frozen_sac(Path(checkpoint), device)
    energy_estimator = EnergyToGoRegressor.load(Path(energy_model), device=device)

    def policy_action(observation: np.ndarray) -> np.ndarray:
        action, _ = policy.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.float64)

    return [
        run_rollout(
            policy_action,
            scenario,
            method,
            energy_weight=energy_weight,
            energy_gradient_weight=energy_gradient_weight,
            energy_gradient_characteristic=energy_gradient_characteristic,
            energy_gradient_normalization=energy_gradient_normalization,
            energy_estimator=energy_estimator,
        )
        for scenario in scenarios
        for method in METHODS
    ]


def main() -> None:
    wall_started = perf_counter()
    cpu_started = process_time()
    parser = argparse.ArgumentParser(description="Controlled UAV safety-filter benchmark")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sac-checkpoint", default=str(DEFAULT_SAC))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rollouts", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--energy-gradient-weight", type=float, default=1.0)
    parser.add_argument(
        "--energy-gradient-normalization",
        choices=("fixed", "unit_direction"),
        default="fixed",
    )
    parser.add_argument("--energy-gradient-characteristic", type=float)
    parser.add_argument("--gradient-audit", default=str(DEFAULT_GRADIENT_AUDIT))
    parser.add_argument("--energy-model", default=str(DEFAULT_ENERGY_MODEL))
    parser.add_argument("--scaling-repeats", type=int, default=30)
    parser.add_argument("--top-k-trials", type=int, default=100)
    parser.add_argument("--top-k-ranking-k", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    if (
        args.torch_threads <= 0
        or args.workers <= 0
        or args.top_k_trials <= 0
        or args.top_k_ranking_k <= 0
    ):
        raise ValueError("thread, worker, and top-K trial counts must be positive")
    torch.set_num_threads(args.torch_threads)
    gradient_characteristic = args.energy_gradient_characteristic
    gradient_characteristic_source = "cli"
    if gradient_characteristic is None:
        audit = json.loads(Path(args.gradient_audit).read_text())
        gradient_characteristic = float(
            audit["action_gradient_norm"]["p95"]
            * np.hypot(5.0, 3.0)
        )
        gradient_characteristic_source = str(Path(args.gradient_audit).resolve())
    if not np.isfinite(gradient_characteristic) or gradient_characteristic <= 0.0:
        raise ValueError("energy gradient characteristic must be finite and positive")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    scenarios = make_scenarios(args.rollouts, args.seed)
    if args.workers == 1:
        results = run_scenario_shard(
            scenarios,
            args.sac_checkpoint,
            args.device,
            args.energy_weight,
            args.energy_gradient_weight,
            gradient_characteristic,
            args.energy_gradient_normalization,
            args.energy_model,
            args.torch_threads,
        )
    else:
        if args.device != "cpu":
            raise ValueError("multi-worker controlled benchmark currently requires --device cpu")
        shards = [scenarios[index :: args.workers] for index in range(args.workers)]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            parts = executor.map(
                run_scenario_shard,
                shards,
                [args.sac_checkpoint] * args.workers,
                [args.device] * args.workers,
                [args.energy_weight] * args.workers,
                [args.energy_gradient_weight] * args.workers,
                [gradient_characteristic] * args.workers,
                [args.energy_gradient_normalization] * args.workers,
                [args.energy_model] * args.workers,
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
        output / "compute_scaling.json",
        compute_scaling(
            [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            repeats=args.scaling_repeats,
            seed=args.seed + 100_000,
        ),
    )
    write_json(
        output / "top_k_priority.json",
        compare_top_k_priorities(
            [32, 64, 128, 256],
            top_k=args.top_k_ranking_k,
            trials=args.top_k_trials,
            seed=args.seed + 200_000,
        ),
    )
    write_json(
        output / "config.json",
        {
            "sac_checkpoint": str(Path(args.sac_checkpoint).resolve()),
            "device": args.device,
            "rollouts": args.rollouts,
            "seed": args.seed,
            "energy_weight": args.energy_weight,
            "energy_gradient_weight": args.energy_gradient_weight,
            "energy_gradient_characteristic": gradient_characteristic,
            "energy_gradient_characteristic_source": gradient_characteristic_source,
            "energy_gradient_normalization": args.energy_gradient_normalization,
            "energy_model": str(Path(args.energy_model).resolve()),
            "methods": METHODS,
            "torch_threads": args.torch_threads,
            "workers": args.workers,
            "top_k_trials": args.top_k_trials,
            "top_k_ranking_k": args.top_k_ranking_k,
            "formal_result": False,
            "purpose": "controlled prototype benchmark",
        },
    )
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall_seconds = perf_counter() - wall_started
    process_cpu_seconds = (
        process_time()
        - cpu_started
        + child_usage.ru_utime
        + child_usage.ru_stime
    )
    source_paths = [
        ROOT / "review_bundle/safety/collision/hocbf.py",
        ROOT / "review_bundle/safety/collision/filter.py",
        ROOT / "review_bundle/safety/collision/geometry3d.py",
        ROOT / "review_bundle/safety/energy/gradients.py",
        ROOT / "experiments/uav_safety_filter/benchmark.py",
        Path(__file__).resolve(),
    ]
    write_json(
        output / "run_manifest.json",
        {
            "command": [sys.executable, *sys.argv],
            "git_sha": git_output("rev-parse", "HEAD"),
            "git_dirty": bool(git_output("status", "--porcelain")),
            "source_sha256": source_hash(source_paths),
            "python": sys.version,
            "platform": platform.platform(),
            "device": args.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "scenario_count": len(scenarios),
            "method_count": len(METHODS),
            "scenario_method_rollouts": len(results),
            "total_method_transitions": int(sum(row.steps for row in results)),
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "cpu_seconds_per_wall_second": process_cpu_seconds
            / max(wall_seconds, 1e-12),
            "peak_rss_mb": max(self_usage.ru_maxrss, child_usage.ru_maxrss) / 1024.0,
        },
    )


if __name__ == "__main__":
    main()
