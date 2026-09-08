"""Algebra checks for the failure diagnosis; not navigation efficacy tests."""

import numpy as np
import torch

from experiments.directional_navigation.cohort import projected_multiplier, terminal_gae
from experiments.directional_navigation.lagrangian import clipped_objective
from scripts.analyze_directional_lagrangian_failure import wilson


def test_all_failure_mc_centering_is_only_baseline_variation():
    costs = np.array([0.0, 0.0, 0.0, 1.0])
    values = np.array([0.3, 0.5, 0.8, 0.9])
    adv, target = terminal_gae(costs, values, 1.0)
    np.testing.assert_allclose(target, 1.0)
    np.testing.assert_allclose(adv - adv.mean(), -(values - values.mean()))


def test_joint_normalization_removes_positive_common_denominator():
    ar = torch.tensor([0.2, -0.5, 0.4, 1.3], dtype=torch.float64)
    ac = torch.tensor([0.8, 0.1, -0.4, 0.2], dtype=torch.float64)
    logp = torch.tensor(
        [0.1, -0.2, 0.05, 0.01], dtype=torch.float64, requires_grad=True
    )
    old = torch.zeros_like(logp)
    lam = 49.3
    loss = clipped_objective(logp, old, ar, ac, lam)
    raw = ar - lam * ac
    standardized = (raw - raw.mean()) / raw.std()
    expected = -torch.minimum(
        standardized * logp.exp(), standardized * logp.exp().clamp(0.8, 1.2)
    ).mean()
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=1e-6)


def test_reward_unit_scaling_requires_dual_unit_scaling():
    rho, lam, eta = 0.01, 3.0, 1.0
    updated = projected_multiplier(lam, 0.8, 0.05, eta)
    scaled = projected_multiplier(rho * lam, 0.8, 0.05, rho * eta)
    assert abs(scaled - rho * updated) < 1e-12


def test_zero_events_wilson_does_not_certify_zero_risk():
    low, high = wilson(0, 50)
    assert abs(low) < 1e-12
    assert 0.071 < high < 0.072
