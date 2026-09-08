"""Deployment-time features for the frozen rechargeability critics."""

from __future__ import annotations

import numpy as np

from experiments.forked_action_safety.core import nearest_obstacle_direction


def _unit(delta: np.ndarray) -> tuple[np.ndarray, float]:
    value = np.asarray(delta, dtype=np.float32)
    distance = float(np.linalg.norm(value))
    if distance <= 1e-8:
        return np.zeros(3, dtype=np.float32), 0.0
    return (value / distance).astype(np.float32), distance


def geometry_action_features(
    *,
    position: np.ndarray,
    velocity: np.ndarray,
    task_goal: np.ndarray,
    charger_goal: np.ndarray,
    active_goal: np.ndarray,
    obstacle_layout: list[dict[str, object]],
    nearest_clearance: float,
    leg: int,
    elapsed_policy_steps: int,
    maximum_steps: int,
    nominal_action: np.ndarray,
    proposed_action: np.ndarray,
    world_extent: np.ndarray,
    d_max: float,
    horizontal_v_max: float,
    vertical_v_max: float,
    lidar_max_range: float,
) -> np.ndarray:
    """Reproduce the 26-D forked-action geometry context online."""

    position = np.asarray(position, dtype=np.float32)
    velocity = np.asarray(velocity, dtype=np.float32)
    task_goal = np.asarray(task_goal, dtype=np.float32)
    charger_goal = np.asarray(charger_goal, dtype=np.float32)
    active_goal = np.asarray(active_goal, dtype=np.float32)
    nominal_action = np.asarray(nominal_action, dtype=np.float32)
    proposed_action = np.asarray(proposed_action, dtype=np.float32)
    extent = np.asarray(world_extent, dtype=np.float32)
    if any(value.shape != (3,) for value in (
        position, velocity, task_goal, charger_goal, active_goal,
        nominal_action, proposed_action, extent,
    )):
        raise ValueError("closed-loop feature vectors must have shape (3,)")
    if not all(np.all(np.isfinite(value)) for value in (
        position, velocity, task_goal, charger_goal, active_goal,
        nominal_action, proposed_action, extent,
    )):
        raise ValueError("closed-loop feature vectors must be finite")
    if leg not in (0, 1) or not 0 <= elapsed_policy_steps < maximum_steps:
        raise ValueError("invalid leg or elapsed step")
    if min(d_max, horizontal_v_max, vertical_v_max, lidar_max_range) <= 0.0:
        raise ValueError("physical normalizers must be positive")
    task_direction, task_distance = _unit(task_goal - position)
    charger_direction, charger_distance = _unit(charger_goal - position)
    active_direction, _ = _unit(active_goal - position)
    obstacle_direction = nearest_obstacle_direction(position, obstacle_layout)
    remaining_horizon = (
        (2 * maximum_steps - elapsed_policy_steps) / (2 * maximum_steps)
        if leg == 0
        else (maximum_steps - elapsed_policy_steps) / (2 * maximum_steps)
    )
    context = np.concatenate(
        [
            position / extent,
            velocity / np.asarray(
                [horizontal_v_max, horizontal_v_max, vertical_v_max],
                dtype=np.float32,
            ),
            task_direction,
            [task_distance / d_max],
            charger_direction,
            [charger_distance / d_max],
            [float(leg), remaining_horizon, nearest_clearance / lidar_max_range],
            proposed_action,
            proposed_action - nominal_action,
            [
                float(np.dot(proposed_action, active_direction)),
                float(np.dot(proposed_action, obstacle_direction)),
                float(np.linalg.norm(proposed_action)),
            ],
        ]
    ).astype(np.float32)
    if context.shape != (26,) or not np.all(np.isfinite(context)):
        raise RuntimeError("closed-loop geometry context is invalid")
    return context


def temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    value = np.asarray(probability, dtype=np.float64)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if temperature == 1.0:
        return value.astype(np.float32)
    clipped = np.clip(value, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped) - np.log1p(-clipped)
    return (1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40.0, 40.0)))).astype(np.float32)
