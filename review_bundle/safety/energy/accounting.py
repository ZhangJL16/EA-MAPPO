from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnergyTransition:
    state: np.ndarray
    action: np.ndarray
    cost: float
    next_state: np.ndarray
    charger_hit: bool

    def __post_init__(self) -> None:
        if not np.isfinite(self.cost) or self.cost < 0.0:
            raise ValueError("energy cost must be finite and nonnegative")
        if not np.all(np.isfinite(self.state)) or not np.all(np.isfinite(self.next_state)):
            raise ValueError("states must be finite")
        if not np.all(np.isfinite(self.action)):
            raise ValueError("action must be finite")


@dataclass(frozen=True)
class EnergyReturnEpisode:
    transitions: tuple[EnergyTransition, ...]
    completed: bool
    policy_hash: str
    context: dict[str, float | str]

    def __post_init__(self) -> None:
        if not self.transitions:
            raise ValueError("an energy episode needs at least one transition")
        terminal_flags = [transition.charger_hit for transition in self.transitions]
        if any(terminal_flags[:-1]):
            raise ValueError("charger hit may occur only on the last transition")
        if self.completed != terminal_flags[-1]:
            raise ValueError("completed must match the final charger-hit flag")


def returns_to_go(costs: np.ndarray) -> np.ndarray:
    values = np.asarray(costs, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("costs must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("costs must be finite and nonnegative")
    return np.cumsum(values[::-1], dtype=np.float64)[::-1].copy()
