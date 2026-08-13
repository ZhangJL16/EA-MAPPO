from __future__ import annotations

import numpy as np
import torch

from experiments.energy_transfer.stage_a_bootstrap import (
    build_dataset,
    collect_bootstrap_trajectory,
    split_sorties,
    trajectory_payload,
)
from experiments.energy_transfer.retraining import (
    CriticTrainingProtocol,
    deterministic_energy_dataset,
    train_quantile_td_repair,
    train_scalar_td_repair,
)
from experiments.energy_transfer import decide_managed_goal
from envs.navigation import NavigationEnv, OperationalEnergyConfig, OperationalEnergyWrapper
from safety.switching import EnergySwitchController, SortieMode


class GoalFollower:
    def action(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray:
        del deterministic
        position = observation[0:3]
        velocity = observation[3:6]
        goal = observation[6:9]
        return np.clip(8.0 * (goal - position) - 0.8 * velocity, -1.0, 1.0).astype(np.float32)


class ConstantReturnPredictor:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, observation: np.ndarray, action: np.ndarray) -> float:
        assert observation.shape == (77,)
        assert action.shape == (3,)
        return self.value


def _trajectory(seed: int, prefix_steps: int):
    return collect_bootstrap_trajectory(
        GoalFollower(),
        scenario="random_persistent_open.json",
        sortie_seed=seed,
        prefix_steps=prefix_steps,
        max_return_steps=500,
        operational_energy_capacity=3.0,
        policy_hash="test-policy",
    )


def test_stage_a_collects_random_prefix_and_complete_return_supervision() -> None:
    trajectory = _trajectory(71, 7)
    assert trajectory.prefix_steps_executed == 7
    assert trajectory.prefix_energy > 0.0
    assert trajectory.operational_energy_at_commitment < trajectory.operational_energy_at_reset
    assert trajectory.episode.completed
    assert len(trajectory.episode.transitions) > 1
    payload = trajectory_payload(0, trajectory, np.array([4.0, 4.0, 2.0]))
    transitions = payload["transitions"]
    costs = np.array([row["per_step_energy"] for row in transitions])
    expected = np.cumsum(costs[::-1])[::-1]
    np.testing.assert_allclose([row["return_energy_to_go"] for row in transitions], expected)
    assert transitions[-1]["charger_hit"]
    assert transitions[-1]["return_energy_to_go"] == transitions[-1]["per_step_energy"]
    assert all(row["distance_to_charger"] >= 0.0 for row in transitions)
    assert all(row["path_length_remaining"] >= 0.0 for row in transitions)
    assert transitions[0]["operational_soc_before"] < 1.0
    assert transitions[-1]["operational_energy_after"] < transitions[0]["operational_energy_before"]


def test_sortie_split_prevents_transition_leakage() -> None:
    trajectories = [_trajectory(100 + index, index) for index in range(10)]
    split = split_sorties(len(trajectories), seed=3)
    assert not set(split["train"]) & set(split["calibration"])
    assert not set(split["train"]) & set(split["test"])
    assert not set(split["calibration"]) & set(split["test"])
    train = build_dataset(trajectories, split["train"])
    test = build_dataset(trajectories, split["test"])
    assert not set(train.sortie_indices) & set(test.sortie_indices)
    assert train.features.shape[1] == 80
    assert train.features.shape[0] == train.returns.shape[0]
    assert train.path_lengths_to_go.shape == train.returns.shape
    assert train.velocity_magnitudes.shape == train.returns.shape
    assert train.sortie_seeds.shape == train.returns.shape


def test_censored_return_has_no_return_to_go_supervision() -> None:
    trajectory = collect_bootstrap_trajectory(
        GoalFollower(),
        scenario="random_persistent_open.json",
        sortie_seed=333,
        prefix_steps=5,
        max_return_steps=1,
        operational_energy_capacity=3.0,
        policy_hash="test-policy",
    )
    assert not trajectory.episode.completed
    payload = trajectory_payload(9, trajectory, np.array([4.0, 4.0, 2.0]))
    assert payload["censored"]
    assert payload["total_return_energy"] is None
    assert payload["total_return_path_length"] is None
    assert all(row["return_energy_to_go"] is None for row in payload["transitions"])
    assert all(row["path_length_remaining"] is None for row in payload["transitions"])


def test_repaired_td_recovers_one_two_and_multistep_deterministic_returns() -> None:
    protocol = CriticTrainingProtocol(
        total_updates=400,
        mc_pretrain_updates=300,
        normalize_returns=True,
        anchor_near_terminal=True,
        low_scale_initialization=True,
        target_update="polyak0.01",
        weight_nonuniform_quantile_atoms=True,
    )
    for length in (1, 2, 10):
        dataset = deterministic_energy_dataset([0.1] * length)
        datasets = {"train": dataset, "calibration": dataset, "test": dataset}
        scalar, scalar_scale, _ = train_scalar_td_repair(datasets, protocol, seed=length)
        quantile, quantile_scale, _ = train_quantile_td_repair(datasets, protocol, seed=length)
        with torch.no_grad():
            features = torch.from_numpy(dataset.features)
            scalar_prediction = scalar(features).numpy() * scalar_scale
            quantile_prediction = quantile(features).numpy() * quantile_scale
        np.testing.assert_allclose(scalar_prediction, dataset.returns, atol=0.03)
        np.testing.assert_allclose(quantile_prediction[:, 0], dataset.returns, atol=0.06)
        assert np.all(quantile_prediction[:, 1:] >= quantile_prediction[:, :-1])


def test_managed_goal_interface_switches_navigation_goal_once() -> None:
    environment = OperationalEnergyWrapper(
        NavigationEnv(max_episode_steps=20),
        OperationalEnergyConfig(capacity=1.0, reserve=0.1, enforce_exhaustion=True),
    )
    _, _ = environment.reset(seed=44, options={"randomize_station": True})
    task_goal = environment.navigation_env.goal.copy()
    controller = EnergySwitchController(reserve=0.1)
    decision = decide_managed_goal(
        environment,
        GoalFollower(),
        ConstantReturnPredictor(0.95),
        controller,
        task_goal=task_goal,
    )
    assert decision.switch.mode is SortieMode.CHARGER_COMMITTED
    assert decision.switch.switched_now
    np.testing.assert_array_equal(environment.navigation_env.goal, environment.navigation_env.station_position)
    environment.navigation_env.goal = task_goal.copy()
    second = decide_managed_goal(
        environment,
        GoalFollower(),
        ConstantReturnPredictor(0.01),
        controller,
        task_goal=task_goal,
    )
    assert not second.switch.switched_now
    assert controller.task_to_charger_count == 1
    np.testing.assert_array_equal(environment.navigation_env.goal, environment.navigation_env.station_position)
