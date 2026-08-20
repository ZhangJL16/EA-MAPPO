from __future__ import annotations

import numpy as np

from experiments.memory_safety.grounded_contraction_diagnostic import (
    ContractionDiagnosticConfig,
    FOGMConstraintProposal,
    GenericHistoryProposal,
    _candidate_features,
    _constraints_from_row,
    _selection,
)


def _row() -> dict[str, object]:
    return {
        "trajectory_id": 1,
        "step": 20,
        "track_id": "track",
        "base_measurement": [2.0, 0.0, 0.0],
        "base_age_seconds": 0.0,
        "base_epoch": 20,
        "candidate_measurements": [[1.8, 0.0, 0.0], [1.6, 0.0, 0.0], [20.0, 20.0, 20.0]],
        "candidate_ages_seconds": [0.1, 0.2, 0.1],
        "candidate_epochs": [18, 16, 18],
        "candidate_track_ids": ["track", "track", "wrong"],
        "candidate_association_verified": [True, True, False],
        "true_state": [2.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }


def test_grounded_features_expose_verifiable_provenance_but_generic_does_not() -> None:
    generic, labels, valid = _candidate_features(_row(), 2.0, grounded=False)
    grounded, grounded_labels, grounded_valid = _candidate_features(_row(), 2.0, grounded=True)
    assert generic.shape == (3, 5)
    assert grounded.shape == (3, 10)
    assert np.array_equal(labels, grounded_labels)
    assert np.array_equal(valid, grounded_valid)
    assert valid.tolist() == [True, True, False]


def test_oracle_only_selects_valid_history_before_same_verifier() -> None:
    config = ContractionDiagnosticConfig(dataset_dir="unused", output_dir="unused", proposal_budget=2)
    base, candidates = _constraints_from_row(_row(), config.sensor_error_bound)
    selected = _selection(
        "C5_oracle_valid_history",
        _row(),
        candidates,
        config,
        GenericHistoryProposal(5, 4),
        FOGMConstraintProposal(10, 4),
    )
    assert base.track_id == "track"
    assert len(selected) == 2
    assert all(item.track_id == "track" and item.association_verified for item in selected)


def test_current_only_never_proposes_history() -> None:
    config = ContractionDiagnosticConfig(dataset_dir="unused", output_dir="unused")
    _, candidates = _constraints_from_row(_row(), config.sensor_error_bound)
    selected = _selection(
        "C0_current_only",
        _row(),
        candidates,
        config,
        GenericHistoryProposal(5, 4),
        FOGMConstraintProposal(10, 4),
    )
    assert selected == []
