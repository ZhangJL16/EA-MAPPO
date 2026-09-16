from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .physics_rollout import PhysicsRolloutResult


@dataclass(frozen=True)
class TerminalEnergyEstimate:
    point: np.ndarray
    upper: np.ndarray


@dataclass(frozen=True)
class TrajectoryEnergyResult:
    rollout_energy: np.ndarray
    terminal_point: np.ndarray
    terminal_upper: np.ndarray
    total_point: np.ndarray
    total_upper: np.ndarray


TerminalEnergyEstimator = Callable[[np.ndarray, np.ndarray, np.ndarray], TerminalEnergyEstimate]


def trajectory_energy(
    rollout: PhysicsRolloutResult,
    goal: np.ndarray,
    terminal_estimator: TerminalEnergyEstimator | None = None,
) -> TrajectoryEnergyResult:
    rollout_energy = np.sum(rollout.step_energy, axis=1)
    if terminal_estimator is None:
        terminal = TerminalEnergyEstimate(
            point=np.zeros(rollout.batch_size, dtype=np.float64),
            upper=np.zeros(rollout.batch_size, dtype=np.float64),
        )
    else:
        terminal = terminal_estimator(
            rollout.positions[:, -1], rollout.velocities[:, -1], np.asarray(goal, dtype=np.float64)
        )
    point = np.asarray(terminal.point, dtype=np.float64)
    upper = np.asarray(terminal.upper, dtype=np.float64)
    if point.shape != (rollout.batch_size,) or upper.shape != point.shape:
        raise ValueError("terminal energy estimator must return one point and upper value per candidate")
    if np.any(upper < point):
        raise ValueError("terminal upper energy must dominate terminal point estimate")
    return TrajectoryEnergyResult(
        rollout_energy=rollout_energy,
        terminal_point=point,
        terminal_upper=upper,
        total_point=rollout_energy + point,
        total_upper=rollout_energy + upper,
    )
