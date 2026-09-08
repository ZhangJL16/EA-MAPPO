from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    action_gain: float
    residual_sigma: float
    residual_rho: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("intervention name must be nonempty")
        if not 0.0 < self.action_gain <= 1.5:
            raise ValueError("action gain must lie in (0, 1.5]")
        if not 0.0 <= self.residual_sigma <= 1.0:
            raise ValueError("residual sigma must lie in [0, 1]")
        if not 0.0 <= self.residual_rho < 1.0:
            raise ValueError("residual rho must lie in [0, 1)")

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def descriptor(self) -> np.ndarray:
        return np.asarray(
            [self.action_gain, self.residual_sigma, self.residual_rho],
            dtype=np.float32,
        )


DEFAULT_INTERVENTIONS = (
    InterventionSpec("nominal", 1.00, 0.00, 0.00),
    InterventionSpec("conservative", 0.75, 0.03, 0.90),
    InterventionSpec("residual_low", 1.00, 0.10, 0.85),
    InterventionSpec("residual_medium", 1.00, 0.20, 0.85),
    InterventionSpec("residual_high", 1.00, 0.35, 0.85),
)


class CorrelatedActionIntervention:
    """A reproducible local behavior policy around a frozen actor."""

    def __init__(self, spec: InterventionSpec, *, seed: int) -> None:
        self.spec = spec
        self.rng = np.random.default_rng(int(seed))
        self.residual = np.zeros(3, dtype=np.float32)

    def reset_leg(self) -> None:
        self.residual.fill(0.0)

    def __call__(self, nominal_action: np.ndarray) -> np.ndarray:
        nominal = np.asarray(nominal_action, dtype=np.float32)
        if nominal.shape != (3,) or not np.all(np.isfinite(nominal)):
            raise ValueError("nominal action must be a finite 3-vector")
        if self.spec.residual_sigma > 0.0:
            innovation = self.rng.standard_normal(3).astype(np.float32)
            rho = self.spec.residual_rho
            self.residual = (
                rho * self.residual + np.sqrt(1.0 - rho * rho) * innovation
            ).astype(np.float32)
        action = (
            self.spec.action_gain * nominal
            + self.spec.residual_sigma * self.residual
        )
        return np.clip(action, -1.0, 1.0).astype(np.float32)


def intervention_seed(base_seed: int, scene_index: int, intervention_index: int) -> int:
    if min(base_seed, scene_index, intervention_index) < 0:
        raise ValueError("seed components must be nonnegative")
    sequence = np.random.SeedSequence([base_seed, scene_index, intervention_index])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def grouped_outer_fold(scene_indices: np.ndarray, folds: int = 3) -> np.ndarray:
    values = np.asarray(scene_indices, dtype=np.int64)
    if values.ndim != 1 or folds < 2 or np.any(values < 0):
        raise ValueError("invalid grouped-fold request")
    return values % int(folds)


def relative_range(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return float("nan")
    return float((np.max(finite) - np.min(finite)) / max(np.median(finite), 1e-12))
