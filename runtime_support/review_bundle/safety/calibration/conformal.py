from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass(frozen=True)
class CalibrationReport:
    sample_count: int
    coverage: float
    violation_rate: float
    mean_tightness_on_covered: float
    mean_underestimate_on_violations: float


class SplitConformalUpperBound:
    def __init__(self, alpha: float) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        self.alpha = float(alpha)
        self.correction: float | None = None

    def fit(self, predictions: np.ndarray, outcomes: np.ndarray) -> "SplitConformalUpperBound":
        predicted = np.asarray(predictions, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        if predicted.shape != observed.shape or predicted.ndim != 1 or predicted.size == 0:
            raise ValueError("predictions and outcomes must be aligned nonempty vectors")
        if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(observed)):
            raise ValueError("calibration values must be finite")
        residuals = np.sort(observed - predicted)
        rank = ceil((residuals.size + 1) * (1.0 - self.alpha))
        self.correction = float("inf") if rank > residuals.size else float(residuals[rank - 1])
        return self

    def predict(self, predictions: np.ndarray) -> np.ndarray:
        if self.correction is None:
            raise RuntimeError("calibrator must be fit before prediction")
        values = np.asarray(predictions, dtype=np.float64)
        return values + self.correction

    @staticmethod
    def report(bounds: np.ndarray, outcomes: np.ndarray) -> CalibrationReport:
        upper = np.asarray(bounds, dtype=np.float64)
        observed = np.asarray(outcomes, dtype=np.float64)
        if upper.shape != observed.shape or upper.ndim != 1 or upper.size == 0:
            raise ValueError("bounds and outcomes must be aligned nonempty vectors")
        covered = observed <= upper
        tightness = upper[covered] - observed[covered]
        underestimate = observed[~covered] - upper[~covered]
        return CalibrationReport(
            sample_count=int(upper.size),
            coverage=float(np.mean(covered)),
            violation_rate=float(np.mean(~covered)),
            mean_tightness_on_covered=float(np.mean(tightness)) if tightness.size else 0.0,
            mean_underestimate_on_violations=float(np.mean(underestimate)) if underestimate.size else 0.0,
        )
