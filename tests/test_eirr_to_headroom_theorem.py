from __future__ import annotations

import numpy as np
import pytest

from experiments.energy_mc.return_decision import (
    certified_effective_evar_requirement,
)
from scripts.verify_eirr_to_headroom_theorem import (
    build_eirr_headroom_verification_report,
)


def test_effective_requirement_interval_and_boundary_bound_are_tight() -> None:
    report = build_eirr_headroom_verification_report()
    assert report["status"] == "EIRR_TO_HEADROOM_IDENTITIES_VERIFIED"
    assert report["requirement_bound_is_tight"] is True
    assert report["exact_first_disagreement_probability"] == pytest.approx(0.4)
    assert report["first_disagreement_probability_bound"] == pytest.approx(0.4)
    assert report["no_late_commit_on_certificate"] is True
    assert report["population_headroom_preserved_by_bound"] is True


def test_effective_requirement_contains_exact_value_for_multiple_risks() -> None:
    result = certified_effective_evar_requirement(
        critic_mgf_values=np.asarray([[2.0, 4.0], [3.0, 6.0]]),
        absolute_certificate_radii=np.asarray([[0.2, 0.4], [0.3, 0.6]]),
        risk_parameters=np.asarray([0.5, 1.0]),
        tail_probabilities=np.asarray([0.1, 0.05]),
        exact_mgf_values=np.asarray([[2.1, 3.8], [2.8, 6.2]]),
    )
    assert result.exact_mgf_inside_certificate is True
    assert result.learned_requirement_is_conservative is True
    assert result.commitment_branch_index == 1
    assert result.effective_requirement_semantics == (
        "task_then_return_stopping_boundary"
    )
    assert result.effective_lower_requirement <= result.exact_effective_requirement
    assert (
        result.exact_effective_requirement
        <= result.learned_effective_upper_requirement
    )
    assert (
        result.learned_effective_upper_requirement
        - result.exact_effective_requirement
        <= result.effective_error_bound + 1e-12
    )


def test_effective_requirement_rejects_nonpositive_lower_mgf() -> None:
    with pytest.raises(ValueError, match="lower certified MGF endpoint"):
        certified_effective_evar_requirement(
            critic_mgf_values=np.asarray([[0.1]]),
            absolute_certificate_radii=np.asarray([[0.1]]),
            risk_parameters=np.asarray([1.0]),
            tail_probabilities=0.05,
        )


def test_direct_return_branch_does_not_override_mission_stopping_boundary() -> None:
    result = certified_effective_evar_requirement(
        critic_mgf_values=np.asarray([[100.0], [2.0]]),
        absolute_certificate_radii=np.asarray([[1.0], [0.1]]),
        risk_parameters=np.asarray([1.0]),
        tail_probabilities=np.asarray([0.05, 0.05]),
        exact_mgf_values=np.asarray([[100.0], [2.0]]),
    )
    expected_mission = np.log(2.0) + np.log(20.0)
    assert result.exact_effective_requirement == pytest.approx(expected_mission)
    assert result.exact_effective_requirement < result.exact_branch_requirements[0]
