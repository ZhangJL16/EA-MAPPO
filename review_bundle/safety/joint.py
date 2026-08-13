from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JointSafetyDecision:
    admissible: np.ndarray
    collision_admissible: np.ndarray
    energy_admissible: np.ndarray
    selected_index: int | None
    reason: str


class JointSafetyFilter:
    def __init__(self, reserve: float) -> None:
        if not np.isfinite(reserve) or reserve < 0.0:
            raise ValueError("reserve must be finite and nonnegative")
        self.reserve = float(reserve)

    def evaluate(
        self,
        collision_upper: np.ndarray,
        energy_upper: np.ndarray,
        task_scores: np.ndarray,
        *,
        collision_budget: float,
        conservative_energy: float,
    ) -> JointSafetyDecision:
        collision = np.asarray(collision_upper, dtype=np.float64)
        energy = np.asarray(energy_upper, dtype=np.float64)
        scores = np.asarray(task_scores, dtype=np.float64)
        if collision.shape != energy.shape or collision.shape != scores.shape or collision.ndim != 1:
            raise ValueError("candidate arrays must be aligned vectors")
        if not np.all(np.isfinite(collision)) or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(scores)):
            raise ValueError("candidate values must be finite")
        if np.any(collision < 0.0) or np.any(collision > 1.0) or np.any(energy < 0.0):
            raise ValueError("risk and energy bounds are outside their domains")
        collision_ok = collision <= collision_budget
        energy_ok = energy + self.reserve <= conservative_energy
        admissible = collision_ok & energy_ok
        selected = int(np.argmax(np.where(admissible, scores, -np.inf))) if np.any(admissible) else None
        return JointSafetyDecision(
            admissible=admissible,
            collision_admissible=collision_ok,
            energy_admissible=energy_ok,
            selected_index=selected,
            reason="selected_jointly_admissible" if selected is not None else "no_admissible_action",
        )
