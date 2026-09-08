"""Single-factor cost trace and transparent trajectory logging contracts."""

import numpy as np
import pytest

from experiments.directional_navigation.cohort import CohortNavigation
from experiments.directional_navigation.cost_trace import actor_cost_trace
from scripts.audit_directional_timeouts import TracedCohort, trajectory_metrics


def test_trace_changes_actor_only_and_mc_control_is_bitwise_identity():
    data = {
        "ac": np.ones(4, dtype=np.float32),
        "rc": np.ones(4, dtype=np.float32),
        "ar": np.arange(4, dtype=np.float32),
        "rr": np.arange(4, dtype=np.float32),
    }
    rows = [{"seed": 10, "steps": 4, "outcome": "contact"}]
    mc, vc = actor_cost_trace(data, rows, [10], 1.0)
    gae, _ = actor_cost_trace(data, rows, [10], 0.95)
    assert mc["ac"] is data["ac"]
    np.testing.assert_allclose(vc, 0.0)
    np.testing.assert_allclose(gae["ac"], [0.95**3, 0.95**2, 0.95, 1.0])
    for key in ("rc", "ar", "rr"):
        assert gae[key] is data[key]


def test_gae_has_zero_terminal_bootstrap_without_changing_timeout_cost():
    data = {
        "ac": np.full(3, -0.8, dtype=np.float32),
        "rc": np.zeros(3, dtype=np.float32),
    }
    result, _ = actor_cost_trace(
        data, [{"seed": 2, "steps": 3, "outcome": "timeout"}], [2], 0.95
    )
    np.testing.assert_allclose(
        result["ac"], -0.8 * np.array([0.95**2, 0.95, 1.0]), atol=1e-6
    )
    np.testing.assert_array_equal(result["rc"], np.zeros(3))


def test_order_uses_registered_seeds_not_finish_order():
    data = {
        "ac": np.array([1.0, 1.0, 0.0], np.float32),
        "rc": np.array([1.0, 1.0, 0.0], np.float32),
    }
    rows = [
        {"seed": 9, "steps": 1, "outcome": "safe_goal"},
        {"seed": 20, "steps": 2, "outcome": "contact"},
    ]
    result, _ = actor_cost_trace(data, rows, [20, 9], 0.95)
    np.testing.assert_allclose(result["ac"], [0.95, 1.0, 0.0])
    with pytest.raises(ValueError):
        actor_cost_trace(data, rows, [9, 20], 0.95)


def test_trace_does_not_change_observations_rewards_or_physics():
    ordinary = CohortNavigation(horizon=4, obstacles=0)
    traced = TracedCohort(horizon=4, obstacles=0)
    try:
        a, _ = ordinary.reset(seed=456)
        b, _ = traced.reset(seed=456)
        np.testing.assert_array_equal(a, b)
        for _ in range(4):
            aa = ordinary.step(np.array([0.1, -0.2, 0.05]))
            bb = traced.step(np.array([0.1, -0.2, 0.05]))
            np.testing.assert_array_equal(aa[0], bb[0])
            assert aa[1:4] == bb[1:4]
        assert bb[4]["navigation_episode"]["trajectory_audit"]["tail_steps"] == 4
    finally:
        ordinary.close()
        traced.close()


def test_full_resolution_summary_distinguishes_motion_from_progress():
    trace = np.zeros((501, 9))
    trace[:, 0] = np.arange(501)
    trace[:, 1] = np.sin(np.arange(501) / 5) * 10
    trace[:, 4] = 50 + trace[:, 1]
    trace[:, 5] = 5
    trace[:, 7] = 20
    metrics = trajectory_metrics(trace, 5)
    assert metrics["tail_inefficient_motion_flag"]
    assert not metrics["tail_stall_flag"]
    assert not metrics["entered_goal_ball_at_policy_sample"]
