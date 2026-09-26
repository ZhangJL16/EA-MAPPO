"""Policy-independent goal stream conditioned on executable full-battery trips.

Candidate identifiers, not policy steps or RNG draws, define the stream.  The
same nth accepted target is therefore available to every compared policy, but
the policy only receives the current target after its predecessor completes.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .calibration import run_job
from .routes import PlanarRouter
from .synthetic import SyntheticMap


@dataclass(frozen=True)
class SelectedTarget:
    position_xy: tuple[float, float]
    accepted_candidate_id: int
    next_candidate_id: int
    rejected_reference_count: int
    rejected_energy_count: int
    free_space_proposals: int
    reference_round_trip_energy: float
    reference_round_trip_seconds: float


def select_next_target(
    case: SyntheticMap,
    router: PlanarRouter,
    full_battery_energy: float,
    *,
    next_candidate_id: int = 0,
    max_candidates: int = 1000,
) -> SelectedTarget:
    if not isfinite(full_battery_energy) or full_battery_energy <= 0:
        raise ValueError("full-battery energy must be finite and positive")
    if not isinstance(next_candidate_id, int) or next_candidate_id < 0:
        raise ValueError("next candidate ID must be nonnegative")
    if not isinstance(max_candidates, int) or max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    rejected_reference = 0
    rejected_energy = 0
    free_space_proposals = 0
    for candidate_id in range(next_candidate_id, next_candidate_id + max_candidates):
        row = run_job(case, router, candidate_id)
        free_space_proposals += row["proposal_attempts"]
        if row["outcome"] != "reference_round_trip_success":
            rejected_reference += 1
            continue
        if row["round_trip_energy"] > full_battery_energy:
            rejected_energy += 1
            continue
        return SelectedTarget(
            tuple(row["target_xy"]), candidate_id, candidate_id + 1,
            rejected_reference, rejected_energy, free_space_proposals,
            row["round_trip_energy"], row["round_trip_seconds"],
        )
    raise RuntimeError(
        f"no full-battery feasible target among {max_candidates} candidates "
        f"(reference failures={rejected_reference}, energy failures={rejected_energy})"
    )
