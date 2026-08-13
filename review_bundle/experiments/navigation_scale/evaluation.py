from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import numpy as np

from envs.navigation import GOAL_DISTANCE_BINS, ScaleInvariantNavigationEnv


DEFAULT_EVALUATION_SCALES = (4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0)
TRAINING_SCALES = {4.0, 8.0, 12.0, 16.0}


class PredictPolicy(Protocol):
    def predict(self, observation: np.ndarray, *, deterministic: bool) -> tuple[np.ndarray, object]: ...


def _summarize(records: list[dict]) -> dict:
    completed = [record for record in records if record["completed"]]
    total_steps = sum(record["steps"] for record in records)
    return {
        "sorties": len(records),
        "completion_rate": float(np.mean([record["completed"] for record in records])),
        "timeout_rate": float(np.mean([record["timeout"] for record in records])),
        "mean_steps": float(np.mean([record["steps"] for record in records])),
        "mean_steps_completed": float(np.mean([record["steps"] for record in completed])) if completed else None,
        "mean_path_efficiency_completed": (
            float(np.mean([record["path_efficiency"] for record in completed])) if completed else None
        ),
        "mean_final_distance": float(np.mean([record["final_distance"] for record in records])),
        "boundary_contact_step_rate": float(
            sum(record["boundary_contacts"] for record in records) / max(total_steps, 1)
        ),
    }


def evaluate_policy(
    policy: PredictPolicy,
    *,
    scales: Iterable[float] = DEFAULT_EVALUATION_SCALES,
    sorties_per_distance_bin: int = 20,
    max_steps: int = 800,
    seed: int = 100_000,
) -> dict:
    if sorties_per_distance_bin <= 0 or max_steps <= 0:
        raise ValueError("evaluation budgets must be positive")
    records: list[dict] = []
    sortie_index = 0
    for width in tuple(float(value) for value in scales):
        environment = ScaleInvariantNavigationEnv(
            world_size_min=width,
            world_size_max=width,
            max_episode_steps=max_steps,
        )
        for label, _, _ in GOAL_DISTANCE_BINS:
            try:
                environment.reset(seed=seed + sortie_index, options={"world_size_xy": width, "distance_bin": label})
            except ValueError:
                continue
            for _ in range(sorties_per_distance_bin):
                sortie_seed = seed + sortie_index
                observation, info = environment.reset(
                    seed=sortie_seed,
                    options={"world_size_xy": width, "distance_bin": label},
                )
                initial_distance = float(info["initial_goal_distance"])
                path_length = 0.0
                boundary_contacts = 0
                completed = False
                final_distance = initial_distance
                for _ in range(max_steps):
                    position_before = environment.state.position.copy()
                    action, _ = policy.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, step_info = environment.step(action)
                    path_length += float(np.linalg.norm(environment.state.position - position_before))
                    boundary_contacts += int(step_info["boundary_collision"])
                    final_distance = float(step_info["distance_to_goal_after"])
                    completed = bool(terminated)
                    if terminated or truncated:
                        break
                records.append(
                    {
                        "sortie_id": sortie_index,
                        "sortie_seed": sortie_seed,
                        "world_size_xy": width,
                        "scale_split": "TRAIN_SCALE" if width in TRAINING_SCALES else "HELD_OUT_SCALE",
                        "distance_bin": label,
                        "initial_distance": initial_distance,
                        "completed": completed,
                        "timeout": not completed,
                        "steps": int(environment.episode_step),
                        "path_length": path_length,
                        "path_efficiency": initial_distance / max(path_length, 1e-12),
                        "final_distance": final_distance,
                        "boundary_contacts": boundary_contacts,
                    }
                )
                sortie_index += 1
        environment.close()
    by_scale = {
        str(width): _summarize([record for record in records if record["world_size_xy"] == width])
        for width in sorted({record["world_size_xy"] for record in records})
    }
    by_distance = {
        label: _summarize([record for record in records if record["distance_bin"] == label])
        for label, _, _ in GOAL_DISTANCE_BINS
        if any(record["distance_bin"] == label for record in records)
    }
    by_split = {
        split: _summarize([record for record in records if record["scale_split"] == split])
        for split in ("TRAIN_SCALE", "HELD_OUT_SCALE")
        if any(record["scale_split"] == split for record in records)
    }
    return {
        "evaluation_status": "COMPLETED",
        "scales": sorted({record["world_size_xy"] for record in records}),
        "sorties_per_feasible_distance_bin": sorties_per_distance_bin,
        "max_steps": max_steps,
        "overall": _summarize(records),
        "by_scale": by_scale,
        "by_scale_split": by_split,
        "by_distance_bin": by_distance,
        "records": records,
    }
