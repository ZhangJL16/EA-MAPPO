from __future__ import annotations

import numpy as np
import pytest

from experiments.directional_navigation.conditional_model_access import (
    ConditionalModelAccessBranch,
)
from experiments.directional_navigation.dvoi_h_branch import DVOIHBranch
from scripts.run_conditional_history_collection import resample_seed, validate_resample_row


def test_nominal_cmi_plant_matches_dvoi_history_plant() -> None:
    reference = DVOIHBranch(obstacles=48, guard=12_000)
    candidate = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    try:
        reference_observation = reference.reset_world(1_134_000_001)
        candidate_observation = candidate.reset_world(1_134_000_001)
        assert reference.physical_world_record() == candidate.physical_world_record()
        for name in reference_observation:
            np.testing.assert_array_equal(reference_observation[name], candidate_observation[name])
        action = np.zeros(3, dtype=np.float32)
        for _ in range(3):
            reference_observation, reference_info, reference_done = reference.step_policy(action)
            candidate_observation, candidate_info, candidate_done = candidate.step_policy(action)
            assert reference_done == candidate_done
            for name in reference_observation:
                np.testing.assert_array_equal(reference_observation[name], candidate_observation[name])
            for name in ("contact", "collision_count", "termination", "returned"):
                assert reference_info[name] == candidate_info[name]
            for name in ("nominal_action", "executed_action", "realized_acceleration"):
                np.testing.assert_array_equal(reference_info[name], candidate_info[name])
            assert np.count_nonzero(candidate.base.last_realized_execution_error) == 0
    finally:
        reference.close()
        candidate.close()


def test_branch_disturbance_is_reproducible_and_bounded() -> None:
    first = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    second = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    try:
        seed = 8_675_309
        first.reset_world(1_134_000_001)
        second.reset_world(1_134_000_001)
        first.arm_resample(seed)
        second.arm_resample(seed)
        first.begin_branch("R")
        second.begin_branch("R")
        action = np.zeros(3, dtype=np.float32)
        observed_nonzero = False
        for _ in range(5):
            first_observation, _, _ = first.step_policy(action)
            second_observation, _, _ = second.step_policy(action)
            np.testing.assert_array_equal(
                first.base.last_requested_execution_error,
                second.base.last_requested_execution_error,
            )
            np.testing.assert_array_equal(
                first.base.last_realized_execution_error,
                second.base.last_realized_execution_error,
            )
            assert np.max(np.abs(first.base.last_requested_execution_error)) <= 0.12 + 1e-7
            observed_nonzero |= bool(np.count_nonzero(first.base.last_realized_execution_error))
            for name in first_observation:
                np.testing.assert_array_equal(first_observation[name], second_observation[name])
        assert observed_nonzero
        summary = first.disturbance_summary(steps=5)
        assert summary["execution_error_rms_norm"] <= summary["execution_error_max_norm"]
    finally:
        first.close()
        second.close()


def test_disturbance_rms_uses_filter_invocation_count() -> None:
    branch = ConditionalModelAccessBranch(obstacles=48, guard=12_000)
    try:
        branch.base.execution_error_squared_norm_sum = 0.08
        branch.base.execution_error_sample_count = 2
        branch.base.execution_error_max_norm = 0.2
        summary = branch.disturbance_summary(steps=1)
        assert summary["execution_error_rms_norm"] == pytest.approx(0.2)
        assert summary["execution_error_rms_norm"] <= summary["execution_error_max_norm"]
    finally:
        branch.close()


def test_resample_seed_pairs_actions_and_separates_blocks() -> None:
    arguments = (
        "conditional-history-information-20260912-v2",
        "CMI_CONFIRM",
        "sha256:" + "1" * 64,
        256,
    )
    evaluation = resample_seed(*arguments, "evaluation", 0)
    oracle = resample_seed(*arguments, "oracle_selection", 0)
    assert evaluation != oracle
    # Action is deliberately absent from the key, so R and C use this same value.
    assert evaluation == resample_seed(*arguments, "evaluation", 0)
    assert evaluation != resample_seed(*arguments, "evaluation", 1)


def test_partial_validation_rejects_collision_or_utility_reinterpretation() -> None:
    identity = "sha256:" + "2" * 64
    row = {
        "schema_version": "conditional-history-resample-v1",
        "world_identity": identity,
        "anchor_step": 256,
        "block": "evaluation",
        "replicate": 0,
        "action": "R",
        "disturbance_seed": 7,
        "policy_steps": 10,
        "outcome": {
            "termination": "returned",
            "returned": True,
            "collision_count": 0,
            "collision_free_arrival": True,
            "task_increment": 0,
            "initial_energy": 100.0,
            "remaining_energy": 90.0,
            "operational_failure": False,
            "utility": 0.0,
        },
        "execution_error_rms_norm": 0.04,
        "execution_error_max_norm": 0.08,
    }
    validate_resample_row(
        row, identity=identity, anchor_step=256, action="R", replicate=0,
        disturbance_seed=7, guard=12_000,
    )
    row["outcome"]["collision_count"] = 1
    with pytest.raises(ValueError, match="collision-free arrival"):
        validate_resample_row(
            row, identity=identity, anchor_step=256, action="R", replicate=0,
            disturbance_seed=7, guard=12_000,
        )

