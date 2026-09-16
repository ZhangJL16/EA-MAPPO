from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .commitment import SortieMode


@dataclass(frozen=True)
class EnergySwitchDecision:
    mode: SortieMode
    switched_now: bool
    active_goal: np.ndarray
    remaining_energy: float
    predicted_return_energy: float
    reserve: float
    continuation_margin: float
    reason: str


class EnergySwitchController:
    """Irreversible per-sortie energy stopping controller."""

    def __init__(self, reserve: float) -> None:
        if not np.isfinite(reserve) or reserve < 0.0:
            raise ValueError("reserve must be finite and nonnegative")
        self.reserve = float(reserve)
        self.mode = SortieMode.TASK
        self.task_to_charger_count = 0

    def decide(
        self,
        *,
        remaining_energy: float,
        predicted_return_energy: float,
        task_goal: np.ndarray,
        charger_goal: np.ndarray,
    ) -> EnergySwitchDecision:
        remaining = self._finite_nonnegative(remaining_energy, "remaining_energy")
        predicted = self._finite_nonnegative(predicted_return_energy, "predicted_return_energy")
        task = self._goal(task_goal, "task_goal")
        charger = self._goal(charger_goal, "charger_goal")
        margin = remaining - predicted - self.reserve
        if self.mode is SortieMode.CHARGER_COMMITTED:
            return EnergySwitchDecision(
                self.mode,
                False,
                charger,
                remaining,
                predicted,
                self.reserve,
                margin,
                "commitment_is_absorbing",
            )
        if margin > 0.0:
            return EnergySwitchDecision(
                self.mode,
                False,
                task,
                remaining,
                predicted,
                self.reserve,
                margin,
                "task_continuation_has_energy_margin",
            )
        self.mode = SortieMode.CHARGER_COMMITTED
        self.task_to_charger_count += 1
        return EnergySwitchDecision(
            self.mode,
            True,
            charger,
            remaining,
            predicted,
            self.reserve,
            margin,
            "return_energy_boundary_reached",
        )

    def complete_charge_and_start_sortie(self) -> None:
        self.mode = SortieMode.TASK
        self.task_to_charger_count = 0

    @staticmethod
    def _finite_nonnegative(value: float, name: str) -> float:
        scalar = float(value)
        if not np.isfinite(scalar) or scalar < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return scalar

    @staticmethod
    def _goal(value: np.ndarray, name: str) -> np.ndarray:
        goal = np.asarray(value, dtype=np.float64)
        if goal.shape != (3,) or not np.all(np.isfinite(goal)):
            raise ValueError(f"{name} must be a finite (3,) vector")
        return goal.copy()
