from __future__ import annotations

import numpy as np

from review_bundle.safety.grounded_memory import (
    HistoricalPositionConstraint,
    SetMembershipConfig,
    VerifiedHistorySetUpdater,
    state_box_from_constraints,
)


def _measurement(
    position: np.ndarray,
    age: float,
    epoch: int,
    *,
    track: str = "track",
    association: bool = True,
) -> HistoricalPositionConstraint:
    return HistoricalPositionConstraint(
        measurement=position,
        age_seconds=age,
        track_id=track,
        epoch=epoch,
        sensor_error_bound=0.1,
        association_verified=association,
        provenance="deterministic_test",
    )


def test_multiframe_intersection_contains_constant_velocity_truth_and_contracts() -> None:
    config = SetMembershipConfig(
        sensor_error_bound=0.1,
        velocity_bound=10.0,
        acceleration_bound=3.0,
        jerk_bound=1.0,
    )
    current_position = np.array([5.0, -2.0, 1.0])
    velocity = np.array([2.0, -1.0, 0.5])
    acceleration = np.zeros(3)
    base = _measurement(current_position, 0.0, 10)
    history = [
        _measurement(current_position - age * velocity, age, 10 - index)
        for index, age in enumerate((0.2, 0.4, 0.6), start=1)
    ]
    base_box = state_box_from_constraints([base], config)
    grounded = VerifiedHistorySetUpdater(config).update(base, history).state_box
    truth = np.concatenate((current_position, velocity, acceleration))
    assert base_box is not None
    assert base_box.contains(truth)
    assert grounded.contains(truth)
    assert np.all(grounded.widths <= base_box.widths + 1e-9)
    assert grounded.log_volume < base_box.log_volume


def test_wrong_track_and_unverified_association_are_rejected() -> None:
    config = SetMembershipConfig(sensor_error_bound=0.1)
    base = _measurement(np.zeros(3), 0.0, 5)
    update = VerifiedHistorySetUpdater(config).update(
        base,
        [
            _measurement(np.zeros(3), 0.2, 4, track="other"),
            _measurement(np.zeros(3), 0.4, 3, association=False),
        ],
    )
    assert update.accepted_epochs == (5,)
    assert update.rejection_reasons == (
        "track_id_mismatch",
        "association_unverified",
    )


def test_inconsistent_history_is_rejected_without_changing_base() -> None:
    config = SetMembershipConfig(
        sensor_error_bound=0.1,
        velocity_bound=1.0,
        acceleration_bound=1.0,
        jerk_bound=0.1,
    )
    base = _measurement(np.zeros(3), 0.0, 10)
    inconsistent = _measurement(np.full(3, 100.0), 0.1, 9)
    update = VerifiedHistorySetUpdater(config).update(base, [inconsistent])
    base_box = state_box_from_constraints([base], config)
    assert base_box is not None
    assert update.rejection_reasons == ("set_membership_infeasible",)
    assert np.allclose(update.state_box.lower, base_box.lower)
    assert np.allclose(update.state_box.upper, base_box.upper)


def test_batch_update_matches_sequential_on_valid_and_invalid_proposals() -> None:
    config = SetMembershipConfig(sensor_error_bound=0.1, velocity_bound=4.0)
    base = _measurement(np.zeros(3), 0.0, 10)
    proposals = [
        _measurement(np.full(3, -0.2), 0.1, 9),
        _measurement(np.full(3, 50.0), 0.2, 8, track="wrong"),
    ]
    updater = VerifiedHistorySetUpdater(config)
    sequential = updater.update(base, proposals)
    batch = updater.update_batch(base, proposals)
    assert sequential.accepted_epochs == batch.accepted_epochs
    assert sequential.rejected_epochs == batch.rejected_epochs
    assert sequential.rejection_reasons == batch.rejection_reasons
    assert np.allclose(sequential.state_box.lower, batch.state_box.lower)
    assert np.allclose(sequential.state_box.upper, batch.state_box.upper)
