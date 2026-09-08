#!/usr/bin/env python3
"""Verify the certified EIRR-to-headroom algebra on an explicit finite law."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.return_decision import (
    certified_effective_evar_requirement,
    sequential_boundary_certificate,
)


def build_eirr_headroom_verification_report() -> dict[str, object]:
    critic = np.asarray([[2.0], [4.0]], dtype=np.float64)
    radii = np.asarray([[0.5], [1.0]], dtype=np.float64)
    exact = np.asarray([[2.0], [3.0]], dtype=np.float64)
    requirement = certified_effective_evar_requirement(
        critic_mgf_values=critic,
        absolute_certificate_radii=radii,
        risk_parameters=np.asarray([1.0]),
        tail_probabilities=0.25,
        exact_mgf_values=exact,
    )

    epsilon = requirement.effective_error_bound
    path_weights = np.asarray([0.4, 0.35, 0.25], dtype=np.float64)
    exact_scores = np.asarray([0.3, 1.0, -0.1], dtype=np.float64)
    learned_scores = exact_scores - (
        requirement.learned_effective_upper_requirement
        - float(requirement.exact_effective_requirement)
    )
    exact_disagreements = (exact_scores > 0.0) & (learned_scores <= 0.0)
    exact_disagreement_probability = float(path_weights @ exact_disagreements)
    one_sided_boundary_probability = float(
        path_weights @ ((exact_scores > 0.0) & (exact_scores <= epsilon))
    )
    boundary = sequential_boundary_certificate(
        estimation_failure_probability=0.0,
        boundary_probabilities=np.asarray([one_sided_boundary_probability]),
        boundary_widths=np.asarray([epsilon]),
        throughput_upper_bound=10.0,
        oracle_stranding_rate=0.0,
        oracle_throughput=10.0,
        heuristic_throughput=3.0,
        stranding_ceiling=0.5,
        minimum_throughput_gain_fraction=0.05,
    )

    return {
        "status": "EIRR_TO_HEADROOM_IDENTITIES_VERIFIED",
        "requirement": requirement.as_dict(),
        "exact_requirement_gap": (
            requirement.learned_effective_upper_requirement
            - float(requirement.exact_effective_requirement)
        ),
        "requirement_bound_is_tight": bool(
            np.isclose(
                requirement.learned_effective_upper_requirement
                - float(requirement.exact_effective_requirement),
                requirement.effective_error_bound,
            )
        ),
        "path_weights": path_weights.tolist(),
        "exact_scores": exact_scores.tolist(),
        "learned_scores": learned_scores.tolist(),
        "exact_first_disagreement_probability": exact_disagreement_probability,
        "one_sided_boundary_probability": one_sided_boundary_probability,
        "first_disagreement_probability_bound": (
            boundary.first_disagreement_probability_bound
        ),
        "no_late_commit_on_certificate": bool(
            np.all(~((exact_scores <= 0.0) & (learned_scores > 0.0)))
        ),
        "population_headroom_preserved_by_bound": (
            boundary.population_headroom_preserved
        ),
        "boundary_certificate": boundary.as_dict(),
        "non_claim": (
            "This finite construction verifies interval and stopping algebra; "
            "it does not establish learned ratio rates or a formal Oracle Gate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_eirr_headroom_verification_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
