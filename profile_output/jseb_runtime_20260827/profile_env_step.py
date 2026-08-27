from __future__ import annotations

import argparse
import json
import time
import types
from pathlib import Path

import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from scripts.run_jacobian_safety_energy_1m import parse_args
from scripts.train_uav_energy_delivery_sac import environment_from_args


SOURCE_EVAL = "/home/zjl/mappo/artifacts/jseb500k_navigation_gate_parallel_20260826_211448/eval_checkpoint_500000.json"
SOURCE_TASKS = "/home/zjl/mappo/artifacts/jseb500k_navigation_gate_parallel_20260826_211448/checkpoint_selection_tasks.json"


def arguments() -> argparse.Namespace:
    return parse_args(
        [
            "--output-dir",
            "/tmp/jseb_profile_unused",
            "--navigation-repair-variant",
            "R1",
            "--repair-source-navigation-evaluation",
            SOURCE_EVAL,
            "--repair-evaluation-tasks",
            SOURCE_TASKS,
        ]
    )


def run_case(name: str, *, lidar: bool, hocbf: bool, steps: int) -> dict[str, float | int | str]:
    args = arguments()
    args.lidar_enabled = lidar
    args.hocbf_enabled = hocbf
    environment = environment_from_args(args, phase=SACTrainingPhase.NAVIGATION)
    observation, _ = environment.reset(seed=123)
    timings = {"lidar_seconds": 0.0, "safety_seconds": 0.0}
    lidar_calls = 0
    safety_calls = 0

    original_lidar = environment._update_lidar
    original_safety = environment._safety_filtered_action

    def timed_lidar(self):
        nonlocal lidar_calls
        started = time.perf_counter()
        try:
            return original_lidar()
        finally:
            timings["lidar_seconds"] += time.perf_counter() - started
            lidar_calls += 1

    def timed_safety(self, action):
        nonlocal safety_calls
        started = time.perf_counter()
        try:
            return original_safety(action)
        finally:
            timings["safety_seconds"] += time.perf_counter() - started
            safety_calls += 1

    environment._update_lidar = types.MethodType(timed_lidar, environment)
    environment._safety_filtered_action = types.MethodType(timed_safety, environment)
    rng = np.random.default_rng(2026)
    started = time.perf_counter()
    transitions = 0
    physics_substeps = 0
    hocbf_constraint_build_seconds = 0.0
    hocbf_solver_seconds = 0.0
    hocbf_filter_total_seconds = 0.0
    hocbf_candidate_constraints = 0
    hocbf_selected_constraints = 0
    while transitions < steps:
        action = rng.uniform(-1.0, 1.0, size=3).astype(np.float32)
        observation, _, terminated, truncated, info = environment.step(action)
        transitions += 1
        physics_substeps += int(info["physics_substeps"])
        hocbf_constraint_build_seconds += float(info["hocbf_constraint_build_seconds"])
        hocbf_solver_seconds += float(info["hocbf_solver_seconds"])
        hocbf_filter_total_seconds += float(info["hocbf_filter_total_seconds"])
        for diagnostics in info["hocbf_substep_diagnostics"]:
            hocbf_candidate_constraints += int(diagnostics.get("candidate_constraints", 0))
            hocbf_selected_constraints += int(diagnostics.get("active_constraints", 0))
        if terminated or truncated:
            observation, _ = environment.reset()
    wall = time.perf_counter() - started
    environment.close()
    return {
        "case": name,
        "transitions": transitions,
        "physics_substeps": physics_substeps,
        "wall_seconds": wall,
        "transitions_per_second": transitions / wall,
        "milliseconds_per_transition": 1000.0 * wall / transitions,
        "lidar_seconds": timings["lidar_seconds"],
        "lidar_fraction": timings["lidar_seconds"] / wall,
        "lidar_calls": lidar_calls,
        "safety_seconds": timings["safety_seconds"],
        "safety_fraction": timings["safety_seconds"] / wall,
        "safety_calls": safety_calls,
        "hocbf_constraint_build_seconds": hocbf_constraint_build_seconds,
        "hocbf_solver_seconds": hocbf_solver_seconds,
        "hocbf_filter_total_seconds": hocbf_filter_total_seconds,
        "mean_candidate_constraints_per_safety_call": hocbf_candidate_constraints / max(safety_calls, 1),
        "mean_active_constraints_per_safety_call": hocbf_selected_constraints / max(safety_calls, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    cli = parser.parse_args()
    rows = [
        run_case("full_1024ray_hocbf", lidar=True, hocbf=True, steps=cli.steps),
        run_case("lidar_only_1024ray", lidar=True, hocbf=False, steps=cli.steps),
        run_case("no_lidar_no_hocbf", lidar=False, hocbf=False, steps=cli.steps),
    ]
    cli.output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
