from __future__ import annotations

import numpy as np
import pytest

from experiments.energy_mc.return_decision import sequential_boundary_certificate
from scripts.verify_irreversible_boundary_theorem import (
    build_boundary_verification_report,
)


def test_explicit_coupling_saturates_boundary_occupation_bound() -> None:
    report = build_boundary_verification_report()
    assert report["status"] == "IRREVERSIBLE_BOUNDARY_IDENTITIES_VERIFIED"
    assert report["exact_first_disagreement_probability"] == pytest.approx(0.6)
    assert report["first_disagreement_probability_bound"] == pytest.approx(0.6)
    assert report["exact_stranding_deviation"] <= report["stranding_deviation_bound"]
    assert report["exact_throughput_deviation"] <= report["throughput_deviation_bound"]
    assert report["population_headroom_preserved_by_bound"] is True


def test_population_headroom_fails_when_boundary_mass_consumes_margin() -> None:
    certificate = sequential_boundary_certificate(
        estimation_failure_probability=0.01,
        boundary_probabilities=np.asarray([0.02, 0.03]),
        boundary_widths=np.asarray([0.1, 0.1]),
        throughput_upper_bound=100.0,
        oracle_stranding_rate=0.01,
        oracle_throughput=110.0,
        heuristic_throughput=100.0,
        stranding_ceiling=0.05,
        minimum_throughput_gain_fraction=0.05,
    )
    assert certificate.first_disagreement_probability_bound == pytest.approx(0.06)
    assert certificate.learned_stranding_upper == pytest.approx(0.07)
    assert certificate.population_safety_ceiling_preserved is False
    assert certificate.population_headroom_preserved is False


def test_boundary_certificate_requires_complete_population_gate_inputs() -> None:
    with pytest.raises(ValueError, match="all population headroom inputs"):
        sequential_boundary_certificate(
            estimation_failure_probability=0.0,
            boundary_probabilities=np.asarray([0.01]),
            boundary_widths=np.asarray([0.1]),
            throughput_upper_bound=10.0,
            oracle_stranding_rate=0.0,
        )
