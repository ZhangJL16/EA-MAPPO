from __future__ import annotations

import json
import time

import numpy as np

from scripts.run_jacobian_safety_energy_1m import parse_args
from scripts.train_uav_energy_delivery_sac import make_navigation_vec_env


def benchmark(implementation: str, vector_steps: int = 100) -> dict[str, float | int | str]:
    args = parse_args(
        [
            "--output-dir",
            "/tmp/jseb-parallel-benchmark",
            "--navigation-repair-variant",
            "R1",
            "--repair-source-navigation-evaluation",
            "/home/zjl/mappo/artifacts/jseb500k_navigation_gate_parallel_20260826_211448/eval_checkpoint_500000.json",
            "--repair-evaluation-tasks",
            "/home/zjl/mappo/artifacts/jseb500k_navigation_gate_parallel_20260826_211448/checkpoint_selection_tasks.json",
        ]
    )
    args.training_vec_env = implementation
    environment = make_navigation_vec_env(args)
    actions = np.zeros((args.num_envs, 3), dtype=np.float32)
    environment.reset()
    started = time.perf_counter()
    for _ in range(vector_steps):
        environment.step(actions)
    wall_seconds = time.perf_counter() - started
    worker_pids = [process.pid for process in getattr(environment, "processes", [])]
    environment.close()
    transitions = vector_steps * args.num_envs
    return {
        "implementation": implementation,
        "num_envs": args.num_envs,
        "vector_steps": vector_steps,
        "transitions": transitions,
        "wall_seconds": wall_seconds,
        "transitions_per_second": transitions / wall_seconds,
        "worker_pids": worker_pids,
    }


if __name__ == "__main__":
    rows = [benchmark("dummy"), benchmark("subproc")]
    rows.append(
        {
            "speedup": rows[1]["transitions_per_second"]
            / rows[0]["transitions_per_second"]
        }
    )
    print(json.dumps(rows, indent=2))
