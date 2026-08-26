from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .commitment import SortieMode


@dataclass(frozen=True)
class ReturnDecisionContext:
    mode: SortieMode
    remaining_energy: float
    battery_capacity: float
    reserve: float
    distance_to_charger: float
    distance_to_task: float
    task_to_charger_distance: float
    return_now_requirement: float | None = None
    task_then_return_requirement: float | None = None


@dataclass(frozen=True)
class ReturnManagerDecision:
    commit: bool
    reason: str
    immediate_margin: float
    mission_margin: float
    decision_statistic: float


class ReturnManager(Protocol):
    manager_type: str

    def decide(self, context: ReturnDecisionContext) -> ReturnManagerDecision: ...


class QuantileEnergyReturnManager:
    """One-way return manager using externally supplied resource upper bounds."""

    manager_type = "quantile_energy_boundary"
    requires_energy_estimate = True

    def decide(self, context: ReturnDecisionContext) -> ReturnManagerDecision:
        if context.mode is SortieMode.CHARGER_COMMITTED:
            return ReturnManagerDecision(
                False,
                "commitment_is_absorbing",
                float("-inf"),
                float("-inf"),
                float("inf"),
            )
        remaining, reserve = _energy_inputs(context)
        return_now = _finite_nonnegative(
            context.return_now_requirement,
            "return_now_requirement",
        )
        task_then_return = _finite_nonnegative(
            context.task_then_return_requirement,
            "task_then_return_requirement",
        )
        immediate_margin = remaining - return_now - reserve
        mission_margin = remaining - task_then_return - reserve
        if immediate_margin <= 0.0:
            return ReturnManagerDecision(
                True,
                "immediate_return_energy_boundary",
                immediate_margin,
                mission_margin,
                return_now,
            )
        if mission_margin <= 0.0:
            return ReturnManagerDecision(
                True,
                "task_then_return_energy_boundary",
                immediate_margin,
                mission_margin,
                task_then_return,
            )
        return ReturnManagerDecision(
            False,
            "task_continuation_has_energy_margin",
            immediate_margin,
            mission_margin,
            task_then_return,
        )


class FixedSOCThresholdReturnManager:
    manager_type = "fixed_soc_threshold"
    requires_energy_estimate = False

    def __init__(self, threshold: float) -> None:
        self.threshold = float(threshold)
        if not np.isfinite(self.threshold) or not 0.0 <= self.threshold <= 1.0:
            raise ValueError("SOC threshold must lie in [0, 1]")

    def decide(self, context: ReturnDecisionContext) -> ReturnManagerDecision:
        if context.mode is SortieMode.CHARGER_COMMITTED:
            return ReturnManagerDecision(
                False,
                "commitment_is_absorbing",
                float("-inf"),
                float("-inf"),
                self.threshold,
            )
        remaining, _ = _energy_inputs(context)
        capacity = _finite_positive(context.battery_capacity, "battery_capacity")
        margin = remaining / capacity - self.threshold
        return ReturnManagerDecision(
            margin <= 0.0,
            "soc_threshold_reached" if margin <= 0.0 else "soc_above_threshold",
            margin,
            margin,
            self.threshold,
        )


class DistanceEnergyReturnManager:
    manager_type = "distance_energy_boundary"
    requires_energy_estimate = False

    def __init__(self, energy_per_meter: float) -> None:
        self.energy_per_meter = _finite_positive(energy_per_meter, "energy_per_meter")

    def decide(self, context: ReturnDecisionContext) -> ReturnManagerDecision:
        if context.mode is SortieMode.CHARGER_COMMITTED:
            return ReturnManagerDecision(
                False,
                "commitment_is_absorbing",
                float("-inf"),
                float("-inf"),
                float("inf"),
            )
        remaining, reserve = _energy_inputs(context)
        distance_to_charger = _finite_nonnegative(
            context.distance_to_charger,
            "distance_to_charger",
        )
        task_route_distance = _finite_nonnegative(
            context.distance_to_task,
            "distance_to_task",
        ) + _finite_nonnegative(
            context.task_to_charger_distance,
            "task_to_charger_distance",
        )
        return_now = distance_to_charger * self.energy_per_meter
        task_then_return = task_route_distance * self.energy_per_meter
        immediate_margin = remaining - return_now - reserve
        mission_margin = remaining - task_then_return - reserve
        if immediate_margin <= 0.0:
            reason = "distance_return_boundary"
            statistic = return_now
        elif mission_margin <= 0.0:
            reason = "distance_task_then_return_boundary"
            statistic = task_then_return
        else:
            reason = "distance_task_continuation_has_margin"
            statistic = task_then_return
        return ReturnManagerDecision(
            min(immediate_margin, mission_margin) <= 0.0,
            reason,
            immediate_margin,
            mission_margin,
            statistic,
        )


def _energy_inputs(context: ReturnDecisionContext) -> tuple[float, float]:
    remaining = _finite_nonnegative(context.remaining_energy, "remaining_energy")
    reserve = _finite_nonnegative(context.reserve, "reserve")
    return remaining, reserve


def _finite_nonnegative(value: float | None, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return scalar


def _finite_positive(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return scalar


__all__ = [
    "DistanceEnergyReturnManager",
    "FixedSOCThresholdReturnManager",
    "QuantileEnergyReturnManager",
    "ReturnDecisionContext",
    "ReturnManager",
    "ReturnManagerDecision",
]
