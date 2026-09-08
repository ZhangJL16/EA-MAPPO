"""Finite-sample upper risk bounds for scene-grouped safety losses.

The HBB construction follows the public RCPS reference implementation while
adding endpoint-safe arithmetic.  Each input loss must be one independent
scene-level observation in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import xlog1py, xlogy
from scipy.stats import binom


def _binary_kl(observed: float, candidate: float) -> float:
    q = float(np.clip(observed, 0.0, 1.0))
    p = float(np.clip(candidate, 1e-15, 1.0 - 1e-15))
    return float(xlogy(q, q / p) + xlog1py(1.0 - q, -q) - xlog1py(1.0 - q, -p))


def _hoeffding_plus(candidate: float, observed: float, count: int) -> float:
    return -count * _binary_kl(max(candidate, observed), candidate)


def _bentkus_plus(candidate: float, observed: float, count: int) -> float:
    tail = float(binom.cdf(np.floor(count * observed), count, candidate))
    return float(np.log(max(tail, 1e-300)) + 1.0)


def _variance_upper(std: float, count: int, delta: float) -> float:
    observed = float(std**2)
    pairs = int(np.floor(count / 2))

    def tail(candidate: float) -> float:
        hoeffding = -pairs * _binary_kl(min(candidate, observed), candidate)
        bentkus_cdf = float(
            binom.cdf(np.ceil(pairs * observed), pairs, candidate)
        )
        bentkus = float(np.log(max(bentkus_cdf, 1e-300)) + 1.0)
        maurer_pontil = (
            -(count - 1) / (2.0 * candidate) * max(candidate - observed, 0.0) ** 2
        )
        return min(hoeffding, bentkus, maurer_pontil) - np.log(delta)

    lower = max(observed, 1e-15)
    return 0.5 if tail(0.25) > 0.0 else float(np.sqrt(brentq(tail, lower, 0.25)))


def hbb_upper_mean(
    losses: np.ndarray,
    *,
    delta: float = 0.05,
    num_grid: int = 200,
) -> float:
    """Hoeffding--Bentkus--empirical-Bennett upper bound for E[loss]."""

    values = np.asarray(losses, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("HBB requires at least two independent scene losses")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("scene losses must be finite values in [0, 1]")
    if not 0.0 < delta < 1.0 or num_grid < 2:
        raise ValueError("delta and num_grid are invalid")
    count = int(values.size)
    observed = float(np.mean(values))
    if observed >= 1.0:
        return 1.0
    std_upper = _variance_upper(float(np.std(values)), count, delta / 2.0)

    def empirical_bennett(candidate: float) -> float:
        grid = np.linspace(0.0, candidate * (1.0 - 1.0 / num_grid), num_grid)
        variance = std_upper**2 + grid**2
        width = candidate - grid
        deviation = max(candidate - observed, 0.0)
        ratio = width * deviation / variance
        h2 = (1.0 + ratio) * np.log1p(ratio) - ratio
        return float(-count * np.max(variance / width**2 * h2))

    def tail(candidate: float) -> float:
        return min(
            _hoeffding_plus(candidate, observed, count),
            _bentkus_plus(candidate, observed, count),
            empirical_bennett(candidate),
        ) - np.log(delta / 2.0)

    upper_endpoint = 1.0 - 1e-10
    if tail(upper_endpoint) > 0.0:
        return 1.0
    return float(brentq(tail, max(observed, 1e-12), upper_endpoint, maxiter=10_000))


@dataclass(frozen=True)
class RCPSThreshold:
    threshold: float
    empirical_risk: float
    risk_upper_bound: float
    coverage: float
    grid_index: int
    nonvacuous: bool


def select_rcps_threshold(
    losses_by_threshold: np.ndarray,
    coverages: np.ndarray,
    thresholds: np.ndarray,
    *,
    alpha: float = 0.05,
    delta: float = 0.05,
) -> tuple[RCPSThreshold, np.ndarray]:
    """Select the least conservative threshold on a fixed nested grid.

    The selected index is the first member of the conservative suffix for
    which every more conservative HBB upper bound is at most ``alpha``.
    """

    table = np.asarray(losses_by_threshold, dtype=np.float64)
    grid = np.asarray(thresholds, dtype=np.float64)
    coverage = np.asarray(coverages, dtype=np.float64)
    if table.ndim != 2 or table.shape[1] != grid.size or coverage.shape != grid.shape:
        raise ValueError("risk table, threshold grid, and coverage do not align")
    if np.any(np.diff(grid) < 0.0):
        raise ValueError("thresholds must be ordered from permissive to conservative")
    upper = np.asarray(
        [hbb_upper_mean(table[:, index], delta=delta) for index in range(grid.size)],
        dtype=np.float64,
    )
    suffix_valid = np.logical_and.accumulate((upper <= alpha)[::-1])[::-1]
    valid = np.flatnonzero(suffix_valid)
    if valid.size == 0:
        index = grid.size - 1
        nonvacuous = False
    else:
        index = int(valid[0])
        nonvacuous = bool(np.isfinite(grid[index]) and coverage[index] > 0.0)
    return (
        RCPSThreshold(
            threshold=float(grid[index]),
            empirical_risk=float(np.mean(table[:, index])),
            risk_upper_bound=float(upper[index]),
            coverage=float(coverage[index]),
            grid_index=index,
            nonvacuous=nonvacuous,
        ),
        upper,
    )
