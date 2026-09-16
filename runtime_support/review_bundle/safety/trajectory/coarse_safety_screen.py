from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class CoarseScreenMode(str, Enum):
    HEURISTIC_RANK = "heuristic_rank"
    CERTIFIED_REJECT = "certified_reject"


@dataclass(frozen=True)
class CoarseScreenResult:
    survivor_indices: np.ndarray
    rejected_indices: np.ndarray
    scores: np.ndarray
    minimum_approximate_clearance: np.ndarray
    certified_rejection: np.ndarray
    mode: CoarseScreenMode


def coarse_safety_screen(
    approximate_positions: np.ndarray,
    obstacle_centers: np.ndarray,
    safe_distances: np.ndarray,
    *,
    keep_count: int,
    mode: CoarseScreenMode = CoarseScreenMode.HEURISTIC_RANK,
    position_error_bound: np.ndarray | float | None = None,
    progress_scores: np.ndarray | None = None,
    energy_proxy: np.ndarray | None = None,
) -> CoarseScreenResult:
    positions = np.asarray(approximate_positions, dtype=np.float64)
    centers = np.asarray(obstacle_centers, dtype=np.float64)
    distances = np.asarray(safe_distances, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError("approximate_positions must have shape (batch, horizon, 3)")
    if centers.ndim != 2 or centers.shape[1] != 3 or distances.shape != (centers.shape[0],):
        raise ValueError("obstacle centers and safe distances must align")
    if keep_count <= 0:
        raise ValueError("keep_count must be positive")
    delta = positions[:, :, None, :] - centers[None, None, :, :]
    clearance = np.linalg.norm(delta, axis=-1) - distances[None, None, :]
    minimum = np.min(clearance, axis=(1, 2))
    progress = np.zeros(positions.shape[0]) if progress_scores is None else np.asarray(
        progress_scores, dtype=np.float64
    )
    energy = np.zeros(positions.shape[0]) if energy_proxy is None else np.asarray(
        energy_proxy, dtype=np.float64
    )
    if progress.shape != minimum.shape or energy.shape != minimum.shape:
        raise ValueError("progress and energy proxies must align with candidates")
    scale = max(float(np.std(energy)), 1e-12)
    scores = minimum + 0.1 * progress - 0.01 * energy / scale
    certified_rejection = np.zeros(positions.shape[0], dtype=bool)
    eligible = np.ones(positions.shape[0], dtype=bool)
    if mode is CoarseScreenMode.CERTIFIED_REJECT:
        if position_error_bound is None:
            raise ValueError("CERTIFIED_REJECT requires a position_error_bound")
        error = np.asarray(position_error_bound, dtype=np.float64)
        if error.ndim == 0:
            error = np.full(positions.shape[:2], float(error))
        elif error.shape == (positions.shape[1],):
            error = np.broadcast_to(error[None, :], positions.shape[:2])
        if error.shape != positions.shape[:2] or np.any(error < 0.0):
            raise ValueError("position_error_bound must be nonnegative and align with candidates")
        upper_clearance = clearance + error[:, :, None]
        certified_rejection = np.any(upper_clearance < 0.0, axis=(1, 2))
        eligible &= ~certified_rejection
    eligible_indices = np.flatnonzero(eligible)
    order = eligible_indices[np.argsort(scores[eligible_indices])[::-1]]
    survivors = order[: min(keep_count, order.size)]
    rejected = np.setdiff1d(np.arange(positions.shape[0]), survivors, assume_unique=True)
    return CoarseScreenResult(
        survivor_indices=survivors,
        rejected_indices=rejected,
        scores=scores.copy(),
        minimum_approximate_clearance=minimum.copy(),
        certified_rejection=certified_rejection,
        mode=mode,
    )
