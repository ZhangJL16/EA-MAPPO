from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CandidateEvaluation:
    action_sequences: np.ndarray
    certified: np.ndarray
    progress_admissible: np.ndarray
    progress_decrease: np.ndarray
    energy_upper: np.ndarray
    minimum_clearance: np.ndarray

    def __post_init__(self) -> None:
        actions = np.asarray(self.action_sequences, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[2] != 3:
            raise ValueError("action_sequences must have shape (batch, horizon, 3)")
        batch = actions.shape[0]
        for name in (
            "certified",
            "progress_admissible",
            "progress_decrease",
            "energy_upper",
            "minimum_clearance",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != (batch,):
                raise ValueError(f"{name} must align with candidates")
        object.__setattr__(self, "action_sequences", actions.copy())


@dataclass(frozen=True)
class SelectionResult:
    selected_index: int | None
    selected_sequence: np.ndarray
    used_fallback: bool
    reason: str
    certified_candidate_count: int
    progress_candidate_count: int


def select_safe_trajectory(
    evaluation: CandidateEvaluation,
    fallback_sequence: np.ndarray,
) -> SelectionResult:
    fallback = np.asarray(fallback_sequence, dtype=np.float64)
    if fallback.shape != evaluation.action_sequences.shape[1:]:
        raise ValueError("fallback_sequence must match candidate horizon and action dimensions")
    certified = np.flatnonzero(np.asarray(evaluation.certified, dtype=bool))
    progress = certified[np.asarray(evaluation.progress_admissible, dtype=bool)[certified]]
    if progress.size == 0:
        return SelectionResult(
            selected_index=None,
            selected_sequence=fallback.copy(),
            used_fallback=True,
            reason="no_certified_progress_candidate",
            certified_candidate_count=int(certified.size),
            progress_candidate_count=0,
        )
    energy = np.asarray(evaluation.energy_upper, dtype=np.float64)
    decrease = np.asarray(evaluation.progress_decrease, dtype=np.float64)
    clearance = np.asarray(evaluation.minimum_clearance, dtype=np.float64)
    order = np.lexsort((-clearance[progress], -decrease[progress], energy[progress]))
    selected = int(progress[order[0]])
    return SelectionResult(
        selected_index=selected,
        selected_sequence=evaluation.action_sequences[selected].copy(),
        used_fallback=False,
        reason="minimum_energy_among_certified_progress_candidates",
        certified_candidate_count=int(certified.size),
        progress_candidate_count=int(progress.size),
    )
