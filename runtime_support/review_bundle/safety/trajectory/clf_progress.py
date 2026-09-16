from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CLFProgressResult:
    initial_value: np.ndarray
    terminal_value: np.ndarray
    decrease: np.ndarray
    distance_reduction: np.ndarray
    admissible: np.ndarray


def goal_lyapunov(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    *,
    beta: float = 0.1,
    goal_velocity: np.ndarray | None = None,
) -> np.ndarray:
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    position = np.asarray(position, dtype=np.float64)
    velocity = np.asarray(velocity, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    target_velocity = np.zeros(3) if goal_velocity is None else np.asarray(goal_velocity, dtype=np.float64)
    return 0.5 * np.sum((position - goal) ** 2, axis=-1) + 0.5 * beta * np.sum(
        (velocity - target_velocity) ** 2, axis=-1
    )


def trajectory_progress(
    positions: np.ndarray,
    velocities: np.ndarray,
    goal: np.ndarray,
    *,
    beta: float = 0.1,
    minimum_decrease: float = 0.0,
) -> CLFProgressResult:
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.asarray(velocities, dtype=np.float64)
    if positions.ndim != 3 or positions.shape != velocities.shape or positions.shape[2] != 3:
        raise ValueError("positions and velocities must have shape (batch, horizon+1, 3)")
    initial = goal_lyapunov(positions[:, 0], velocities[:, 0], goal, beta=beta)
    terminal = goal_lyapunov(positions[:, -1], velocities[:, -1], goal, beta=beta)
    initial_distance = np.linalg.norm(positions[:, 0] - np.asarray(goal), axis=1)
    terminal_distance = np.linalg.norm(positions[:, -1] - np.asarray(goal), axis=1)
    decrease = initial - terminal
    distance_reduction = initial_distance - terminal_distance
    return CLFProgressResult(
        initial_value=initial,
        terminal_value=terminal,
        decrease=decrease,
        distance_reduction=distance_reduction,
        admissible=(decrease >= minimum_decrease) & (distance_reduction > 0.0),
    )
