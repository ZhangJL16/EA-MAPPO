from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
import platform
import shlex
import sys
import traceback

import numpy as np

from agents.goal_conditioned_sac import FrozenGoalConditionedSAC
from envs.navigation import NavigationEnv, OperationalEnergyConfig, OperationalEnergyWrapper, load_scenario
from experiments.new_route.provenance import append_jsonl, code_hash, git_sha, sha256_file, utc_now, write_json


ROOT = Path(__file__).resolve().parents[2]
DISTANCE_BINS = (
    ("0-2", 0.5, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-6", 4.0, 6.0),
    ("6-8", 6.0, 8.0),
    ("8-10", 8.0, 10.0),
    ("10-12", 10.0, 12.0),
    (">12", 12.0, np.inf),
)


@dataclass(frozen=True)
class StageBNavigationConfig:
    seed: int
    checkpoint: str
    output_dir: str
    sorties: int = 150
    max_steps: int = 800
    scenario: str = "target_open_16x16.json"
    operational_energy_capacity: float = 10.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not 100 <= self.sorties <= 300:
            raise ValueError("bounded navigation validation requires 100-300 sorties")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.operational_energy_capacity <= 0.0:
            raise ValueError("operational energy capacity must be positive")


def distance_bin_label(distance: float) -> str:
    for label, lower, upper in DISTANCE_BINS:
        if lower <= distance < upper or label == ">12" and distance >= lower:
            return label
    raise ValueError(f"distance {distance} is outside the validation bins")


def _sample_pair(rng: np.random.Generator, world_size: np.ndarray, bin_index: int) -> tuple[np.ndarray, np.ndarray]:
    _, lower, upper = DISTANCE_BINS[bin_index]
    margin = 0.35
    low_xy = np.full(2, margin)
    high_xy = world_size[:2] - margin
    for _ in range(100_000):
        start_xy = rng.uniform(low_xy, high_xy)
        goal_xy = rng.uniform(low_xy, high_xy)
        distance = float(np.linalg.norm(goal_xy - start_xy))
        if lower <= distance < upper:
            return np.array([*start_xy, 1.0]), np.array([*goal_xy, 1.0])
    raise RuntimeError(f"could not sample a pair in distance bin {DISTANCE_BINS[bin_index][0]}")


def navigation_schedule(config: StageBNavigationConfig, world_size: np.ndarray) -> list[dict]:
    rng = np.random.default_rng(config.seed + 16_001)
    bin_order = np.arange(config.sorties, dtype=np.int64) % len(DISTANCE_BINS)
    rng.shuffle(bin_order)
    schedule = []
    for index, bin_index in enumerate(bin_order):
        start, goal = _sample_pair(rng, world_size, int(bin_index))
        route_type = "TASK" if index % 2 == 0 else "CHARGER"
        schedule.append(
            {
                "sortie_id": index,
                "sortie_seed": int(config.seed * 1_000_000 + 40_000 + index),
                "route_type": route_type,
                "distance_bin": DISTANCE_BINS[int(bin_index)][0],
                "start": start,
                "goal": goal,
            }
        )
    return schedule


def _run_sortie(
    policy: FrozenGoalConditionedSAC,
    config: StageBNavigationConfig,
    item: dict,
) -> dict:
    environment = NavigationEnv(config.scenario, max_episode_steps=config.max_steps)
    wrapped = OperationalEnergyWrapper(
        environment,
        OperationalEnergyConfig(capacity=config.operational_energy_capacity, enforce_exhaustion=False),
    )
    station = item["goal"] if item["route_type"] == "CHARGER" else load_scenario(config.scenario).station_position
    observation, _ = wrapped.reset(
        seed=item["sortie_seed"],
        options={
            "start_position": item["start"],
            "goal_position": item["goal"],
            "station_position": station,
            "operational_initial_energy": config.operational_energy_capacity,
        },
    )
    initial_distance = float(np.linalg.norm(item["goal"] - item["start"]))
    path_length = 0.0
    boundary_contacts = 0
    velocity_saturations = 0
    completed = False
    final_distance = initial_distance
    last_info: dict = {}
    for _ in range(config.max_steps):
        position_before = environment.state.position.copy()
        action = policy.action(observation, deterministic=True)
        observation, _, terminated, truncated, last_info = wrapped.step(action)
        path_length += float(np.linalg.norm(environment.state.position - position_before))
        boundary_contacts += int(last_info["boundary_collision"])
        velocity_saturations += int(last_info["velocity_saturated"])
        final_distance = float(last_info["distance_to_goal_after"])
        completed = bool(last_info["task_completed_now"])
        if completed or terminated or truncated:
            break
    steps = int(environment.episode_step)
    final_position = environment.state.position.copy()
    energy_used = float(wrapped.operational_energy_used)
    wrapped.close()
    return {
        "sortie_id": item["sortie_id"],
        "sortie_seed": item["sortie_seed"],
        "route_type": item["route_type"],
        "distance_bin": item["distance_bin"],
        "start_position": item["start"].tolist(),
        "goal_position": item["goal"].tolist(),
        "station_position": np.asarray(station).tolist(),
        "initial_distance": initial_distance,
        "completed": completed,
        "timeout": not completed,
        "steps": steps,
        "path_length": path_length,
        "path_efficiency": initial_distance / max(path_length, 1e-12),
        "final_position": final_position.tolist(),
        "final_goal_distance": final_distance,
        "progress": initial_distance - final_distance,
        "progress_fraction": (initial_distance - final_distance) / max(initial_distance, 1e-12),
        "boundary_contact_count": boundary_contacts,
        "boundary_contact_rate": boundary_contacts / max(steps, 1),
        "velocity_saturation_count": velocity_saturations,
        "velocity_saturation_rate": velocity_saturations / max(steps, 1),
        "operational_energy_used": energy_used,
        "operational_energy_enforced": False,
    }


def _aggregate(records: list[dict]) -> dict:
    def summarize(selected: list[dict]) -> dict:
        completed = [record for record in selected if record["completed"]]
        return {
            "sorties": len(selected),
            "completion_rate": float(np.mean([record["completed"] for record in selected])),
            "timeout_rate": float(np.mean([record["timeout"] for record in selected])),
            "mean_steps": float(np.mean([record["steps"] for record in selected])),
            "mean_path_length": float(np.mean([record["path_length"] for record in selected])),
            "mean_path_efficiency_completed": (
                float(np.mean([record["path_efficiency"] for record in completed])) if completed else 0.0
            ),
            "median_path_efficiency_completed": (
                float(np.median([record["path_efficiency"] for record in completed])) if completed else 0.0
            ),
            "mean_final_goal_distance": float(np.mean([record["final_goal_distance"] for record in selected])),
            "mean_progress_fraction": float(np.mean([record["progress_fraction"] for record in selected])),
            "boundary_contact_step_rate": float(
                sum(record["boundary_contact_count"] for record in selected)
                / max(sum(record["steps"] for record in selected), 1)
            ),
            "sorties_with_boundary_contact_rate": float(
                np.mean([record["boundary_contact_count"] > 0 for record in selected])
            ),
            "velocity_saturation_step_rate": float(
                sum(record["velocity_saturation_count"] for record in selected)
                / max(sum(record["steps"] for record in selected), 1)
            ),
        }

    by_distance = {
        label: summarize([record for record in records if record["distance_bin"] == label])
        for label, _, _ in DISTANCE_BINS
    }
    by_route = {
        route: summarize([record for record in records if record["route_type"] == route])
        for route in ("TASK", "CHARGER")
    }
    overall = summarize(records)
    long_records = [record for record in records if record["initial_distance"] >= 8.0]
    gate_checks = {
        "overall_completion_at_least_0.90": overall["completion_rate"] >= 0.90,
        "task_completion_at_least_0.85": by_route["TASK"]["completion_rate"] >= 0.85,
        "charger_completion_at_least_0.85": by_route["CHARGER"]["completion_rate"] >= 0.85,
        "long_distance_completion_at_least_0.80": summarize(long_records)["completion_rate"] >= 0.80,
        "completed_median_path_efficiency_at_least_0.50": overall[
            "median_path_efficiency_completed"
        ] >= 0.50,
        "boundary_contact_step_rate_at_most_0.10": overall["boundary_contact_step_rate"] <= 0.10,
    }
    return {
        "overall": overall,
        "by_route_type": by_route,
        "by_distance_bin": by_distance,
        "long_distance_ge_8m": summarize(long_records),
        "gate_definition_preregistered_before_execution": gate_checks,
        "NAVIGATION_TRANSFER_VALID": bool(all(gate_checks.values())),
    }


def run_navigation_transfer(config: StageBNavigationConfig) -> dict:
    output = Path(config.output_dir)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite navigation validation: {output}")
    output.mkdir(parents=True)
    checkpoint = Path(config.checkpoint).resolve()
    checkpoint_hash = sha256_file(checkpoint)
    scenario = load_scenario(config.scenario)
    if not np.allclose(scenario.world_size, [16.0, 16.0, 2.0]) or scenario.world.aabbs or scenario.world.cylinders:
        raise ValueError("Stage B navigation gate requires the obstacle-free 16x16x2 target scenario")
    schedule = navigation_schedule(config, scenario.world_size)
    config_payload = asdict(config) | {
        "experiment": "STAGE_B1_16X16_FROZEN_NAVIGATION_TRANSFER",
        "status": "RUNNING",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_modified": False,
        "observation_dimension": 77,
        "obstacles_enabled": False,
        "operational_energy_enforced": False,
        "distance_bins": [label for label, _, _ in DISTANCE_BINS],
        "git_commit_sha": git_sha(ROOT),
        "code_hash": code_hash(ROOT),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "started_at": utc_now(),
        "pid": os.getpid(),
    }
    write_json(output / "config.json", config_payload)
    write_json(output / "RUNNING.json", {"status": "RUNNING", "started_at": config_payload["started_at"]})
    try:
        policy = FrozenGoalConditionedSAC(checkpoint, device=config.device)
        records = []
        raw_path = output / "raw_navigation.jsonl"
        for item in schedule:
            record = _run_sortie(policy, config, item)
            records.append(record)
            append_jsonl(raw_path, record)
        summary = _aggregate(records) | {
            "status": "COMPLETED",
            "completed_at": utc_now(),
            "sorties": len(records),
            "checkpoint_sha256": checkpoint_hash,
            "raw_navigation_sha256": sha256_file(raw_path),
        }
        write_json(output / "results.json", summary)
        write_json(output / "COMPLETED.json", {"status": "COMPLETED", "completed_at": summary["completed_at"]})
        return summary
    except Exception as error:
        write_json(
            output / "FAILED.json",
            {"status": "FAILED", "failed_at": utc_now(), "error": repr(error), "traceback": traceback.format_exc()},
        )
        raise
