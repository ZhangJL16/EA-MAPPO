from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.energy_td_diagnostics.core import (
    DiagnosticDataset,
    TrainingConfig,
    build_n_step_targets,
    compute_mc_returns,
    terminal_anchor_statistics,
    train_diagnostic_estimator,
    value_scale_alarm,
)
from review_bundle.safety.energy.td import quantile_ssp_target


def deterministic_dataset(costs: list[float], trajectories: int = 1) -> DiagnosticDataset:
    single_costs = np.asarray(costs, dtype=np.float32)
    transition_count = single_costs.size * trajectories
    trajectory_ids = np.repeat(np.arange(trajectories, dtype=np.int32), single_costs.size)
    repeated_costs = np.tile(single_costs, trajectories)
    terminals = np.zeros(transition_count, dtype=bool)
    terminals[single_costs.size - 1 :: single_costs.size] = True
    horizons = np.tile(np.arange(single_costs.size, 0, -1, dtype=np.int32), trajectories)
    step_indices = np.tile(np.arange(single_costs.size, dtype=np.int32), trajectories)
    states = np.zeros((transition_count, 7), dtype=np.float32)
    states[:, 6] = horizons / float(single_costs.size)
    next_states = np.zeros_like(states)
    for trajectory_index in range(trajectories):
        start = trajectory_index * single_costs.size
        stop = start + single_costs.size
        next_states[start : stop - 1] = states[start + 1 : stop]
    actions = np.zeros((transition_count, 3), dtype=np.float32)
    returns = compute_mc_returns(repeated_costs, trajectory_ids, terminals)
    zeros3 = np.zeros((transition_count, 3), dtype=np.float32)
    return DiagnosticDataset(
        states=states,
        actions=actions,
        costs=repeated_costs,
        next_states=next_states,
        next_actions=actions.copy(),
        terminals=terminals,
        returns=returns,
        horizons=horizons,
        trajectory_ids=trajectory_ids,
        step_indices=step_indices,
        positions=zeros3.copy(),
        velocities=zeros3.copy(),
        goals=zeros3.copy(),
        distances=horizons.astype(np.float32),
        initial_distances=np.full(transition_count, float(single_costs.size), dtype=np.float32),
        boundary_contacts=np.zeros(transition_count, dtype=bool),
    )


@pytest.mark.parametrize(
    ("costs", "expected"),
    [
        ([2.5], [2.5]),
        ([1.0, 2.0], [3.0, 2.0]),
        ([1.0, 1.0, 1.0, 1.0, 1.0], [5.0, 4.0, 3.0, 2.0, 1.0]),
    ],
)
def test_mc_return_reconstruction(costs: list[float], expected: list[float]) -> None:
    dataset = deterministic_dataset(costs)
    np.testing.assert_allclose(dataset.returns, expected)


def test_terminal_target_is_last_realized_cost() -> None:
    dataset = deterministic_dataset([0.5, 0.25])
    assert dataset.terminals[-1]
    assert dataset.returns[-1] == pytest.approx(dataset.costs[-1])
    target = quantile_ssp_target(
        torch.tensor([dataset.costs[-1]]),
        torch.tensor([[10.0, 20.0, 30.0, 40.0]]),
        torch.tensor([True]),
        gamma=1.0,
    )
    torch.testing.assert_close(
        target,
        torch.full((1, 4), float(dataset.costs[-1])),
    )


def test_n_step_return_stops_at_terminal_without_bootstrap() -> None:
    dataset = deterministic_dataset([1.0, 2.0, 3.0, 4.0])
    costs, bootstrap_indices, terminals = build_n_step_targets(dataset, 3)
    np.testing.assert_allclose(costs, [6.0, 9.0, 7.0, 4.0])
    np.testing.assert_array_equal(bootstrap_indices, [2, 3, 3, 3])
    np.testing.assert_array_equal(terminals, [False, True, True, True])


def test_terminal_anchor_batch_probability() -> None:
    dataset = deterministic_dataset([0.05] * 500, trajectories=2)
    statistics = terminal_anchor_statistics(
        dataset.terminals, dataset.trajectory_ids, batch_size=128
    )
    assert statistics["terminal_fraction"] == pytest.approx(1.0 / 500.0)
    assert statistics["probability_batch_has_zero_terminal"] > 0.77
    assert statistics["mean_terminal_samples_per_batch"] == pytest.approx(0.256)


def test_value_scale_alarm_uses_empirical_return_scale() -> None:
    normal = value_scale_alarm(np.array([20.0, 30.0]), np.array([10.0, 40.0]))
    exploded = value_scale_alarm(np.array([5000.0, 6000.0]), np.array([10.0, 40.0]))
    assert normal["VALUE_SCALE_DIVERGENCE"] is False
    assert exploded["VALUE_SCALE_DIVERGENCE"] is True


@pytest.mark.parametrize(
    "kind",
    [
        "scalar_td",
        "quantile_td",
        "free_four_quantile_td",
        "uniform_quantile_td",
        "unweighted_four_quantile_td",
        "uniform_four_quantile_td",
        "mc_scalar",
    ],
)
def test_known_return_ssp_estimators_remain_finite(kind: str, tmp_path) -> None:
    dataset = deterministic_dataset([1.0] * 10, trajectories=16)
    result = train_diagnostic_estimator(
        dataset,
        dataset,
        TrainingConfig(
            kind=kind,
            updates=80,
            batch_size=16,
            replay_capacity=160,
            learning_starts=16,
            learning_rate=1e-3,
            hidden_dim=16,
            log_every=40,
            device="cpu",
        ),
        output_dir=tmp_path / kind,
    )
    final = result["final"]
    assert np.isfinite(final["loss"])
    assert np.isfinite(final["prediction_primary_max"])
    assert final["value_scale_alarm"]["VALUE_SCALE_DIVERGENCE"] is False


def test_no_bootstrap_mc_is_independent_of_target_tau() -> None:
    dataset = deterministic_dataset([0.5] * 5, trajectories=20)
    common = dict(
        kind="mc_scalar",
        updates=60,
        batch_size=20,
        replay_capacity=100,
        learning_starts=20,
        learning_rate=1e-3,
        hidden_dim=16,
        log_every=60,
        seed=5,
        device="cpu",
    )
    first = train_diagnostic_estimator(dataset, dataset, TrainingConfig(**common, target_tau=0.01))
    second = train_diagnostic_estimator(dataset, dataset, TrainingConfig(**common, target_tau=0.0001))
    assert first["final"]["metrics"]["MAE"] == pytest.approx(
        second["final"]["metrics"]["MAE"], abs=1e-8
    )


def test_hard_target_update_requires_positive_interval() -> None:
    with pytest.raises(ValueError, match="positive interval"):
        TrainingConfig(kind="scalar_td", target_update="hard")
