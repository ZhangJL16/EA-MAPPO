from __future__ import annotations

import hashlib

import numpy as np
import torch

from experiments.rechargeability_safety.core import (
    ACTION_SLICE,
    ConservativeHazardResidualCritic,
    FEATURE_NAMES,
    MonotoneBudgetCritic,
    ResidualMonotoneBudgetCritic,
    bellman_target,
    build_pre_action_context,
    decode_active_goal_state,
    grouped_nested_split,
)
from scripts.run_rechargeability_critic_gate import binary_auroc
from scripts.run_forked_lidar_residual_critic_gate import canonical_anchor_lookup


def test_active_goal_prefix_round_trip() -> None:
    d_max = 6000.0
    goal = np.asarray([2200.0, 1700.0, 180.0], dtype=np.float32)
    position = np.asarray([1000.0, 900.0, 100.0], dtype=np.float32)
    velocity = np.asarray([10.0, -5.0, 2.5], dtype=np.float32)
    delta = goal - position
    distance = float(np.linalg.norm(delta))
    compact = np.concatenate(
        [
            velocity / np.asarray([20.0, 20.0, 5.0], dtype=np.float32),
            delta / distance,
            [np.log1p(distance) / np.log1p(d_max)],
        ]
    ).astype(np.float32)
    decoded_position, decoded_velocity = decode_active_goal_state(
        compact,
        goal,
        d_max=d_max,
        horizontal_v_max=20.0,
        vertical_v_max=5.0,
    )
    np.testing.assert_allclose(decoded_position, position, atol=2e-3)
    np.testing.assert_allclose(decoded_velocity, velocity, atol=1e-6)


def test_context_contains_only_declared_pre_action_fields() -> None:
    proposed = np.asarray([0.2, -0.3, 0.4], dtype=np.float32)
    context = build_pre_action_context(
        position=np.asarray([1000.0, 1200.0, 80.0]),
        velocity=np.asarray([2.0, 3.0, -1.0]),
        task_goal=np.asarray([1500.0, 1200.0, 80.0]),
        charger_goal=np.asarray([100.0, 100.0, 20.0]),
        world_extent=np.asarray([4000.0, 4000.0, 400.0]),
        d_max=5670.0,
        horizontal_v_max=20.0,
        vertical_v_max=5.0,
        leg=0,
        remaining_horizon_fraction=0.75,
        frozen_action=np.zeros(3),
        proposed_action=proposed,
        previous_action=np.zeros(3),
        previous_energy_fraction=0.001,
        previous_filter_intervention=0.02,
        intervention_descriptor=np.asarray([1.0, 0.1, 0.85]),
    )
    assert context.shape == (len(FEATURE_NAMES),)
    np.testing.assert_allclose(context[ACTION_SLICE], proposed)
    forbidden = {
        "current_energy",
        "suffix_energy",
        "collision",
        "terminal",
        "current_filter_intervention",
        "executed_action",
    }
    assert forbidden.isdisjoint(FEATURE_NAMES)


def test_budget_monotonicity_is_architectural() -> None:
    torch.manual_seed(4)
    model = MonotoneBudgetCritic(6, hidden_dim=12, budget_knots=9)
    context = torch.randn(5, 6).repeat_interleave(31, dim=0)
    budget = torch.linspace(0.0, 0.7, 31).repeat(5)
    probability = model(context, budget).reshape(5, 31)
    assert torch.all(probability[:, 1:] >= probability[:, :-1] - 1e-7)
    negative = model(torch.randn(3, 6), torch.full((3,), -0.01))
    assert torch.equal(negative, torch.zeros_like(negative))


def test_zero_residual_exactly_recovers_geometry_and_stays_monotone() -> None:
    torch.manual_seed(9)
    geometry = MonotoneBudgetCritic(5, hidden_dim=12, budget_knots=9)
    residual = ResidualMonotoneBudgetCritic(geometry, 7, hidden_dim=10)
    geometry_context = torch.randn(4, 5).repeat_interleave(21, dim=0)
    residual_context = torch.randn(4, 7).repeat_interleave(21, dim=0)
    budget = torch.linspace(0.0, 0.7, 21).repeat(4)
    expected = geometry(geometry_context, budget)
    actual = residual(geometry_context, residual_context, budget)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)
    values = actual.reshape(4, 21)
    assert torch.all(values[:, 1:] >= values[:, :-1] - 1e-7)
    assert not any(parameter.requires_grad for parameter in residual.geometry_critic.parameters())


def test_zero_residual_is_finite_for_large_geometry_increments() -> None:
    geometry = MonotoneBudgetCritic(2, hidden_dim=4, budget_knots=5)
    with torch.no_grad():
        for parameter in geometry.parameters():
            parameter.zero_()
        geometry.network[-1].bias[1:].fill_(160.0)
    residual = ResidualMonotoneBudgetCritic(geometry, 3, hidden_dim=4)
    context = torch.zeros(9, 2)
    residual_context = torch.zeros(9, 3)
    budget = torch.linspace(0.0, 0.7, 9)
    expected = geometry(context, budget)
    actual = residual(context, residual_context, budget)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-6)


def test_conservative_hazard_is_dominated_and_budget_monotone() -> None:
    torch.manual_seed(11)
    geometry = MonotoneBudgetCritic(5, hidden_dim=12, budget_knots=9)
    hazard = ConservativeHazardResidualCritic(
        geometry, 7, hidden_dim=10, geometry_temperature=1.7
    )
    geometry_context = torch.randn(4, 5).repeat_interleave(21, dim=0)
    hazard_context = torch.randn(4, 7).repeat_interleave(21, dim=0)
    budget = torch.linspace(0.0, 0.7, 21).repeat(4)
    calibrated_geometry = torch.sigmoid(
        torch.logit(geometry(geometry_context, budget).clamp(1e-6, 1 - 1e-6)) / 1.7
    )
    actual = hazard(geometry_context, hazard_context, budget)
    assert torch.isfinite(actual).all()
    assert torch.all(actual <= calibrated_geometry)
    values = actual.reshape(4, 21)
    assert torch.all(values[:, 1:] >= values[:, :-1] - 1e-7)
    assert not any(parameter.requires_grad for parameter in hazard.geometry_critic.parameters())


def test_conservative_hazard_keeps_gradient_in_near_zero_initialization() -> None:
    geometry = MonotoneBudgetCritic(3, hidden_dim=8)
    model = ConservativeHazardResidualCritic(geometry, 4, hidden_dim=8)
    probability = model(torch.randn(6, 3), torch.randn(6, 4), torch.full((6,), 0.3))
    probability.sum().backward()
    final = model.hazard_network[-1]
    assert final.weight.grad is not None
    assert float(final.weight.grad.abs().sum()) > 0.0


def test_canonical_anchor_lookup_ignores_legacy_off_by_one_index() -> None:
    observations = np.arange(12, dtype=np.float32).reshape(3, 4)
    anchors = []
    for row in range(3):
        anchors.append(
            {
                "anchor_id": f"anchor_{row}",
                "anchor_index": row - 1,
                "anchor_observation_sha256": hashlib.sha256(
                    observations[row].tobytes()
                ).hexdigest(),
            }
        )
    lookup = canonical_anchor_lookup({"anchors": anchors}, observations)
    assert [lookup[f"anchor_{row}"][0] for row in range(3)] == [0, 1, 2]


def test_action_gradient_reaches_proposed_action() -> None:
    torch.manual_seed(5)
    model = MonotoneBudgetCritic(len(FEATURE_NAMES), hidden_dim=16)
    context = torch.randn(8, len(FEATURE_NAMES), requires_grad=True)
    model(context, torch.full((8,), 0.2)).sum().backward()
    assert context.grad is not None
    assert float(context.grad[:, ACTION_SLICE].abs().sum()) > 0.0


def test_bellman_target_terminal_and_budget_cases() -> None:
    target = bellman_target(
        next_probability=torch.tensor([0.8, 0.7, 0.6, 0.5]),
        budget=torch.tensor([0.2, 0.1, 0.2, 0.1]),
        realized_energy=torch.tensor([0.1, 0.2, 0.1, 0.2]),
        terminal_success=torch.tensor([1, 1, 0, 0], dtype=torch.bool),
        terminal_failure=torch.tensor([0, 0, 1, 0], dtype=torch.bool),
    )
    torch.testing.assert_close(target, torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_grouped_split_has_no_scene_overlap() -> None:
    scenes = np.repeat(np.arange(60), 25)
    train, validation, test = grouped_nested_split(scenes, outer_fold=1)
    train_scenes = set(scenes[train])
    validation_scenes = set(scenes[validation])
    test_scenes = set(scenes[test])
    assert train_scenes
    assert validation_scenes
    assert test_scenes
    assert not train_scenes & validation_scenes
    assert not train_scenes & test_scenes
    assert not validation_scenes & test_scenes


def test_grouped_split_handles_sparse_bucket_encoded_scene_ids() -> None:
    sparse_scene_ids = np.asarray(
        [0, 1, 2, 30, 31, 32, 60, 61, 62, 90, 91, 92, 120, 121, 122]
    )
    scenes = np.repeat(sparse_scene_ids, 40)
    for fold in range(3):
        train, validation, test = grouped_nested_split(scenes, outer_fold=fold)
        partitions = [set(scenes[mask]) for mask in (train, validation, test)]
        assert all(partitions)
        assert not partitions[0] & partitions[1]
        assert not partitions[0] & partitions[2]
        assert not partitions[1] & partitions[2]
        assert set.union(*partitions) == set(sparse_scene_ids)


def test_binary_auroc_handles_ties_without_sklearn() -> None:
    assert binary_auroc(np.asarray([0, 1]), np.asarray([0.0, 1.0])) == 1.0
    assert binary_auroc(np.asarray([0, 1]), np.asarray([0.5, 0.5])) == 0.5
