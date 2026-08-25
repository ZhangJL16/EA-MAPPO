from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.uav_energy_parallel import ParallelUAVEnvPool, WorkerReset
from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from scripts.train_uav_energy_delivery_sac import (
    _batched_policy_actions,
    environment_kwargs_from_args,
    freeze_navigation_policy,
    parse_args as parse_training_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parallel UAV evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-type", choices=("sac", "jseb"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 6])
    parser.add_argument("--transitions", type=int, default=4800)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def benchmark_worker_count(model, training_args, workers: int, transitions: int) -> dict[str, object]:
    if transitions % workers != 0:
        raise ValueError("transitions must be divisible by every worker count")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    seeds = [190_001 + worker_id for worker_id in range(workers)]
    with ParallelUAVEnvPool(
        environment_kwargs_from_args(
            training_args,
            phase=SACTrainingPhase.NAVIGATION,
        ),
        num_workers=workers,
    ) as pool:
        states = pool.reset_many(
            WorkerReset(worker_id=worker_id, seed=seed)
            for worker_id, seed in enumerate(seeds)
        )
        started = time.perf_counter()
        completed_episodes = 0
        for vector_step in range(transitions // workers):
            worker_ids = list(range(workers))
            observations = np.stack(
                [states[worker_id].observation for worker_id in worker_ids]
            )
            actions = _batched_policy_actions(model, observations)
            states = pool.step_many(worker_ids, actions)
            reset_requests = []
            for worker_id, state in states.items():
                if state.terminated or state.truncated:
                    completed_episodes += 1
                    seeds[worker_id] += workers
                    reset_requests.append(
                        WorkerReset(worker_id=worker_id, seed=seeds[worker_id])
                    )
            if reset_requests:
                states.update(pool.reset_many(reset_requests))
        elapsed = time.perf_counter() - started
    peak_gpu_memory = (
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    )
    return {
        "workers": workers,
        "environment_transitions": transitions,
        "vector_steps": transitions // workers,
        "completed_episodes": completed_episodes,
        "wall_clock_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        "peak_torch_gpu_memory_bytes": peak_gpu_memory,
    }


def main() -> None:
    arguments = parse_args()
    output = Path(arguments.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    training_args = parse_training_args(
        [
            "--output-dir",
            str(output / "unused"),
            "--device",
            arguments.device,
            "--lidar-enabled",
            "--lidar-range",
            "100",
            "--lidar-horizontal-sectors",
            "128",
            "--lidar-vertical-sectors",
            "8",
            "--num-obstacles",
            "24",
            "--obstacle-radius-min",
            "50",
            "--obstacle-radius-max",
            "120",
            "--hocbf-enabled",
            "--hocbf-top-k",
            "16",
        ]
    )
    training_args.battery_calibration_seed = 180_001
    training_args.evaluation_progress_interval_tasks = 1
    model_class = SAC if arguments.model_type == "sac" else JacobianBridgeSAC
    model = model_class.load(arguments.checkpoint, device=arguments.device)
    freeze_navigation_policy(model)
    rows = []
    for workers in arguments.workers:
        rows.append(
            benchmark_worker_count(
                model,
                training_args,
                workers,
                arguments.transitions,
            )
        )
    payload = {
        "checkpoint": str(Path(arguments.checkpoint).resolve()),
        "model_type": arguments.model_type,
        "device": arguments.device,
        "rows": rows,
    }
    (output / "benchmark.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
