"""Single-factor actor reward-credit contract."""

import numpy as np
import pytest

from experiments.directional_navigation.recovery_credit import actor_reward_trace


def test_mc_actor_advantage_reconstructs_reward_and_leaves_critic_target():
    # V=[.5,.3,.2]; r=[1,-2,3], lambda=.95 GAE generated analytically.
    v = np.array([0.5, 0.3, 0.2], np.float32)
    r = np.array([1.0, -2.0, 3.0], np.float32)
    a = np.zeros(3, np.float32)
    carry = 0.0
    for i in range(2, -1, -1):
        nv = 0.0 if i == 2 else v[i + 1]
        carry = float(r[i] + nv - v[i] + 0.95 * carry)
        a[i] = carry
    data = {
        "ar": a,
        "rr": a + v,
        "obs": np.ones((3, 2), np.float32),
        "rc": np.zeros(3, np.float32),
    }
    # Tensor rewards use the fixed .01 training scale; raw_return is unscaled.
    rows = [{"seed": 7, "steps": 3, "raw_return": 200.0}]
    out = actor_reward_trace(data, rows, [7], 1.0)
    np.testing.assert_allclose(out["ar"], np.cumsum(r[::-1])[::-1] - v, atol=1e-6)
    assert (
        out["rr"] is data["rr"]
        and out["obs"] is data["obs"]
        and out["rc"] is data["rc"]
    )


def test_gae_control_is_object_identity_and_order_is_validated():
    data = {"ar": np.ones(2, np.float32), "rr": np.ones(2, np.float32)}
    rows = [{"seed": 1, "steps": 2, "raw_return": 0.0}]
    assert actor_reward_trace(data, rows, [1], 0.95) is data
    with pytest.raises(ValueError):
        actor_reward_trace(data, rows, [2], 1.0)
    with pytest.raises(ValueError):
        actor_reward_trace(data, rows, [1], 0.9)
