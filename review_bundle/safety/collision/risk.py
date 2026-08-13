from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from safety.calibration import SplitConformalUpperBound


def short_horizon_labels(
    collision: np.ndarray,
    near_collision: np.ndarray,
    horizon: int,
) -> np.ndarray:
    collision_events = np.asarray(collision, dtype=bool)
    near_events = np.asarray(near_collision, dtype=bool)
    if collision_events.shape != near_events.shape or collision_events.ndim != 1:
        raise ValueError("collision streams must be aligned vectors")
    if horizon <= 0:
        raise ValueError("prediction horizon must be positive")
    events = collision_events | near_events
    labels = np.zeros(events.size, dtype=np.float32)
    for index in range(events.size):
        labels[index] = float(np.any(events[index : min(events.size, index + horizon)]))
    return labels


class OneSidedRiskCalibrator:
    """One-sided residual calibration for predicted short-horizon event risk.

    The resulting bound is an empirical/conformal marginal object for the
    matched selected-action distribution. It is not a pointwise guarantee.
    """

    def __init__(self, alpha: float) -> None:
        self.calibrator = SplitConformalUpperBound(alpha)

    def fit(self, predicted_risk: np.ndarray, labels: np.ndarray) -> "OneSidedRiskCalibrator":
        scores = np.asarray(predicted_risk, dtype=np.float64)
        outcomes = np.asarray(labels, dtype=np.float64)
        if np.any(scores < 0.0) or np.any(scores > 1.0):
            raise ValueError("predicted risks must lie in [0, 1]")
        if np.any((outcomes != 0.0) & (outcomes != 1.0)):
            raise ValueError("collision labels must be binary")
        self.calibrator.fit(scores, outcomes)
        return self

    def upper(self, predicted_risk: np.ndarray) -> np.ndarray:
        return np.clip(self.calibrator.predict(predicted_risk), 0.0, 1.0)


@dataclass
class CollisionRiskBudget:
    total_budget: float
    remaining_budget: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.total_budget <= 1.0:
            raise ValueError("total collision-risk budget must lie in [0, 1]")
        if self.remaining_budget is None:
            self.remaining_budget = float(self.total_budget)
        if not 0.0 <= self.remaining_budget <= self.total_budget:
            raise ValueError("remaining budget must lie in [0, total_budget]")

    def geometric_allocation(self, step: int, rho: float) -> float:
        if step < 0 or not 0.0 < rho < 1.0:
            raise ValueError("step must be nonnegative and rho must lie in (0, 1)")
        nominal = self.total_budget * (1.0 - rho) * rho**step
        return min(float(self.remaining_budget), nominal)

    def spend(self, allocation: float) -> None:
        if not np.isfinite(allocation) or allocation < 0.0:
            raise ValueError("allocation must be finite and nonnegative")
        if allocation > float(self.remaining_budget) + 1e-15:
            raise ValueError("collision-risk allocation exceeds remaining budget")
        self.remaining_budget = max(0.0, float(self.remaining_budget) - allocation)
