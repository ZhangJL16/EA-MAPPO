from __future__ import annotations

import numpy as np
import pytest
import torch

from safety.energy import (
    MonotoneQuantileCritic,
    quantile_huber_loss,
    quantile_ssp_target,
    returns_to_go,
    scalar_ssp_target,
)


def test_scalar_ssp_target_and_charger_terminal() -> None:
    cost = torch.tensor([2.0, 3.0])
    next_value = torch.tensor([5.0, 7.0])
    terminal = torch.tensor([False, True])
    torch.testing.assert_close(scalar_ssp_target(cost, next_value, terminal), torch.tensor([7.0, 3.0]))


def test_energy_ssp_rejects_discounting() -> None:
    with pytest.raises(ValueError, match="gamma=1"):
        scalar_ssp_target(torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([False]), gamma=0.99)


def test_quantile_target_terminal_zeroes_bootstrap() -> None:
    cost = torch.tensor([1.0, 2.0])
    next_quantiles = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
    terminal = torch.tensor([False, True])
    expected = torch.tensor([[4.0, 5.0], [2.0, 2.0]])
    torch.testing.assert_close(quantile_ssp_target(cost, next_quantiles, terminal), expected)


def test_monotone_quantile_head_never_crosses() -> None:
    torch.manual_seed(3)
    critic = MonotoneQuantileCritic(8)
    quantiles = critic(torch.randn(32, 8))
    assert torch.all(quantiles[:, 1:] >= quantiles[:, :-1])


def test_quantile_huber_has_finite_gradient() -> None:
    prediction = torch.tensor([[1.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.5, 3.0]])
    loss = quantile_huber_loss(prediction, target, torch.tensor([0.5, 0.9]))
    loss.backward()
    assert torch.isfinite(loss)
    assert prediction.grad is not None and torch.all(torch.isfinite(prediction.grad))


def test_returns_to_go_matches_actual_energy_sum() -> None:
    np.testing.assert_allclose(returns_to_go(np.array([1.0, 2.0, 4.0])), np.array([7.0, 6.0, 4.0]))
