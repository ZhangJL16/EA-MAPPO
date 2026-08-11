from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


ScoreFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True, slots=True)
class ProposalCandidateBatch:
    latent: np.ndarray
    physical_action: np.ndarray
    nominal_successor_observation: np.ndarray

    def __post_init__(self) -> None:
        latent = np.asarray(self.latent, dtype=np.float64)
        action = np.asarray(self.physical_action, dtype=np.float64)
        successor = np.asarray(self.nominal_successor_observation, dtype=np.float64)
        if latent.ndim != 2 or latent.shape[1] != 3:
            raise ValueError("latent candidates must have shape (K,3)")
        if action.shape != latent.shape:
            raise ValueError("physical actions must have shape (K,3)")
        if successor.ndim != 2 or successor.shape[0] != latent.shape[0]:
            raise ValueError("successor observations must have leading dimension K")
        if latent.shape[0] < 1 or not all(np.all(np.isfinite(value)) for value in (latent, action, successor)):
            raise ValueError("proposal candidates must be nonempty and finite")
        object.__setattr__(self, "latent", latent.copy())
        object.__setattr__(self, "physical_action", action.copy())
        object.__setattr__(self, "nominal_successor_observation", successor.copy())


@dataclass(frozen=True, slots=True)
class DualFieldProposalAudit:
    selected_index: int
    collision_lower: tuple[float, ...]
    energy_upper: tuple[float, ...]
    shortlist: tuple[int, ...]
    fallback_to_candidate_zero: bool
    reason: str


class TrainingBehaviorProposalRanker:
    """Ranks already-supported training proposals without granting authority."""

    def __init__(self, shortlist_fraction: float = 0.5) -> None:
        if not np.isfinite(shortlist_fraction) or not 0.0 < shortlist_fraction <= 1.0:
            raise ValueError("shortlist_fraction must lie in (0,1]")
        self.shortlist_fraction = float(shortlist_fraction)

    @staticmethod
    def map_latent(center: np.ndarray, generator: np.ndarray, latent: np.ndarray) -> np.ndarray:
        center = np.asarray(center, dtype=np.float64)
        generator = np.asarray(generator, dtype=np.float64)
        latent = np.asarray(latent, dtype=np.float64)
        if center.shape != (3,) or generator.shape != (3, 3) or latent.ndim != 2 or latent.shape[1] != 3:
            raise ValueError("center, generator, and latent shapes must be (3,), (3,3), and (K,3)")
        return center[None, :] + np.tanh(latent) @ generator.T

    def select(
        self,
        candidates: ProposalCandidateBatch,
        observation: np.ndarray,
        score: ScoreFunction | None,
        *,
        generator_authority: bool,
        model_compatible: bool = True,
    ) -> tuple[int, DualFieldProposalAudit]:
        count = candidates.latent.shape[0]
        fallback_reason = None
        if not generator_authority:
            fallback_reason = "GENERATOR_AUTHORITY_UNAVAILABLE"
        elif score is None:
            fallback_reason = "FIELD_CHECKPOINT_UNAVAILABLE"
        elif not model_compatible:
            fallback_reason = "FIELD_CHECKPOINT_INCOMPATIBLE"
        if fallback_reason is not None:
            audit = DualFieldProposalAudit(0, (), (), (0,), True, fallback_reason)
            return 0, audit
        try:
            collision, energy = score(
                np.asarray(observation, dtype=np.float64),
                candidates.physical_action.copy(),
                candidates.nominal_successor_observation.copy(),
            )
            collision = np.asarray(collision, dtype=np.float64)
            energy = np.asarray(energy, dtype=np.float64)
        except Exception:
            collision = energy = np.empty(0, dtype=np.float64)
        if collision.shape != (count,) or energy.shape != (count,) or not (
            np.all(np.isfinite(collision)) and np.all(np.isfinite(energy))
        ):
            audit = DualFieldProposalAudit(0, (), (), (0,), True, "FIELD_OUTPUT_INVALID")
            return 0, audit
        shortlist_size = max(1, int(np.ceil(self.shortlist_fraction * count)))
        collision_order = np.lexsort((np.arange(count), -collision))
        shortlist = collision_order[:shortlist_size]
        selected = int(shortlist[np.lexsort((shortlist, energy[shortlist]))[0]])
        audit = DualFieldProposalAudit(
            selected,
            tuple(float(value) for value in collision),
            tuple(float(value) for value in energy),
            tuple(int(value) for value in shortlist),
            False,
            "DUAL_FIELD_RANKED_TRAINING_BEHAVIOR",
        )
        return selected, audit
