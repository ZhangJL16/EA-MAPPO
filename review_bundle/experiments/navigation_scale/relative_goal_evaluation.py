from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from envs.navigation import D_NEAR, RelativeGoalNavigationEnv


EVALUATION_DISTANCE_BINS: tuple[tuple[str, float, float], ...] = (
    ("0.3-2", 0.30, 2.0),
    ("2-4", 2.0, 4.0),
    ("4-8", 4.0, 8.0),
    ("8-12", 8.0, 12.0),
    (">12", 12.0, np.inf),
)


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, object]: ...


def _summary(records: list[dict]) -> dict:
    if not records:
        return {"sorties": 0}
    completed = [record for record in records if record["completed"]]
    total_steps = sum(record["steps"] for record in records)
    far_steps = sum(record["far_state_steps"] for record in records)
    return {
        "sorties": len(records),
        "completion_rate": float(np.mean([record["completed"] for record in records])),
        "timeout_rate": float(np.mean([record["timeout"] for record in records])),
        "mean_final_goal_distance": float(np.mean([record["final_goal_distance"] for record in records])),
        "mean_steps_to_goal_completed": (
            float(np.mean([record["steps"] for record in completed])) if completed else None
        ),
        "mean_path_length": float(np.mean([record["path_length"] for record in records])),
        "mean_path_efficiency_completed": (
            float(np.mean([record["path_efficiency"] for record in completed])) if completed else None
        ),
        "mean_progress_fraction": float(np.mean([record["progress_fraction"] for record in records])),
        "boundary_contact_step_rate": float(
            sum(record["boundary_contacts"] for record in records) / max(total_steps, 1)
        ),
        "velocity_saturation_step_rate": float(
            sum(record["velocity_saturations"] for record in records) / max(total_steps, 1)
        ),
        "mean_action_goal_alignment": float(
            sum(record["action_goal_alignment_sum"] for record in records) / max(total_steps, 1)
        ),
        "far_positive_progress_step_rate": float(
            sum(record["far_positive_progress_steps"] for record in records) / max(far_steps, 1)
        ),
        "far_mean_progress_per_step": float(
            sum(record["far_progress_sum"] for record in records) / max(far_steps, 1)
        ),
        "mean_max_consecutive_negative_progress_steps": float(
            np.mean([record["max_consecutive_negative_progress_steps"] for record in records])
        ),
        "overshoot_event_rate_per_step": float(
            sum(record["overshoot_events"] for record in records) / max(total_steps, 1)
        ),
    }


def evaluate_relative_goal_policy(
    policy: PredictPolicy,
    *,
    world_sizes: Iterable[float],
    sorties_per_bin: int,
    max_steps: int,
    seed: int,
) -> dict:
    if sorties_per_bin <= 0 or max_steps <= 0:
        raise ValueError("evaluation budgets must be positive")
    records: list[dict] = []
    sortie_id = 0
    for width in tuple(float(value) for value in world_sizes):
        environment = RelativeGoalNavigationEnv(world_size_xy=width, max_episode_steps=max_steps)
        maximum = max(interval[1] for interval in environment.effective_distance_intervals().values())
        for label, lower, upper in EVALUATION_DISTANCE_BINS:
            interval = (lower, min(upper, maximum))
            if interval[1] <= interval[0]:
                continue
            for _ in range(sorties_per_bin):
                sortie_seed = seed + sortie_id
                observation, reset_info = environment.reset(
                    seed=sortie_seed,
                    options={"distance_interval": interval},
                )
                initial_distance = float(reset_info["initial_goal_distance"])
                path_length = 0.0
                boundary_contacts = 0
                velocity_saturations = 0
                alignment_sum = 0.0
                far_steps = 0
                far_positive_steps = 0
                far_progress_sum = 0.0
                overshoot_events = 0
                max_negative_run = 0
                completed = False
                final_distance = initial_distance
                for _ in range(max_steps):
                    position_before = environment.state.position.copy()
                    action, _ = policy.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = environment.step(action)
                    path_length += float(np.linalg.norm(environment.state.position - position_before))
                    boundary_contacts += int(info["boundary_collision"])
                    velocity_saturations += int(info["velocity_saturated"])
                    alignment_sum += float(info["action_goal_alignment"])
                    max_negative_run = max(max_negative_run, int(info["consecutive_negative_progress_steps"]))
                    overshoot_events += int(info["overshoot_event"])
                    if info["far_state"]:
                        far_steps += 1
                        far_positive_steps += int(float(info["goal_progress"]) > 0.0)
                        far_progress_sum += float(info["goal_progress"])
                    final_distance = float(info["distance_to_goal_after"])
                    completed = bool(terminated)
                    if terminated or truncated:
                        break
                records.append(
                    {
                        "sortie_id": sortie_id,
                        "sortie_seed": sortie_seed,
                        "world_size_xy": width,
                        "distance_bin": label,
                        "distance_group": "NEAR" if initial_distance < D_NEAR else "FAR",
                        "initial_goal_distance": initial_distance,
                        "completed": completed,
                        "timeout": not completed,
                        "steps": int(environment.episode_step),
                        "final_goal_distance": final_distance,
                        "path_length": path_length,
                        "path_efficiency": initial_distance / max(path_length, 1e-12),
                        "progress_fraction": (initial_distance - final_distance) / max(initial_distance, 1e-12),
                        "boundary_contacts": boundary_contacts,
                        "velocity_saturations": velocity_saturations,
                        "action_goal_alignment_sum": alignment_sum,
                        "far_state_steps": far_steps,
                        "far_positive_progress_steps": far_positive_steps,
                        "far_progress_sum": far_progress_sum,
                        "max_consecutive_negative_progress_steps": max_negative_run,
                        "overshoot_events": overshoot_events,
                    }
                )
                sortie_id += 1
        environment.close()
    by_world = {
        str(width): _summary([record for record in records if record["world_size_xy"] == width])
        for width in sorted({record["world_size_xy"] for record in records})
    }
    by_bin = {
        label: _summary([record for record in records if record["distance_bin"] == label])
        for label, _, _ in EVALUATION_DISTANCE_BINS
        if any(record["distance_bin"] == label for record in records)
    }
    near = [record for record in records if record["distance_group"] == "NEAR"]
    far = [record for record in records if record["distance_group"] == "FAR"]
    long_16 = [
        record for record in records if record["world_size_xy"] == 16.0 and record["initial_goal_distance"] >= 8.0
    ]
    return {
        "status": "COMPLETED",
        "policy_update_during_evaluation": False,
        "world_sizes": sorted({record["world_size_xy"] for record in records}),
        "sorties_per_feasible_bin": sorties_per_bin,
        "max_steps": max_steps,
        "overall": _summary(records),
        "by_world_size": by_world,
        "by_distance_bin": by_bin,
        "NEAR_COMPLETION_RATE": _summary(near).get("completion_rate"),
        "FAR_COMPLETION_RATE": _summary(far).get("completion_rate"),
        "LONG_DISTANCE_COMPLETION_RATE_16X16_GE_8M": _summary(long_16).get("completion_rate"),
        "records": records,
    }
