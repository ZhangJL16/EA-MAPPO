"""Count semantics, reward-only gradients, and recovery checkpoint regression."""

import copy

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import TwoValuePolicy
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.recovery_ppo import (
    collect_recovery,
    update_reward_only,
    validate_count_targets,
)
from scripts import train_recovery_ppo
from scripts.run_directional_cost_trace_pair import restore
from scripts.run_directional_lagrangian_pair import save


def make_policy(env):
    torch.manual_seed(4101)
    return TwoValuePolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        features_extractor_class=DirectionalLidarExtractor,
        features_extractor_kwargs={"remaining_time": True},
        share_features_extractor=False,
        net_arch={"pi": [32], "vf": [32]},
    )


def test_count_validation_not_binary_outcome_or_completion_order():
    rows = [
        {"seed": 2, "steps": 2, "collision_count": 0},
        {"seed": 1, "steps": 3, "collision_count": 2},
    ]
    data = {"rc": np.array([2.0, 1.0, 1.0, 0.0, 0.0])}
    validate_count_targets(data, rows, [1, 2])
    with pytest.raises(ValueError):
        validate_count_targets(
            {"rc": np.array([1.0, 1.0, 1.0, 0.0, 0.0])}, rows, [1, 2]
        )
    with pytest.raises(ValueError):
        validate_count_targets(data, rows, [2, 1])


def test_cost_tensors_cannot_affect_reward_only_update():
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=3, obstacles=0)])
    try:
        first = make_policy(vec)
        data, rows, audit = collect_recovery(first, vec, [123])
        assert audit["physical_steps"] == 3 and audit["total_collision_count"] == 0
        assert "outcome" not in rows[0] and "task_contact_rate" not in audit
        second = copy.deepcopy(first)
        poisoned = dict(data, ac=np.full(3, np.nan), rc=np.full(3, 1e10))
        np.random.seed(71)
        a = update_reward_only(first, data, epochs=2, batch_size=3)
        np.random.seed(71)
        b = update_reward_only(second, poisoned, epochs=2, batch_size=3)
        assert a["gradient_steps"] > 0 and b["cost_head_unchanged"]
        assert all(
            torch.equal(v, second.state_dict()[k])
            for k, v in first.state_dict().items()
        )
        assert all(p.grad is None for p in first.cost_net.parameters())
    finally:
        vec.close()


def test_save_restore_repeats_next_update_bitwise(tmp_path):
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=3, obstacles=0)])
    try:
        policy = make_policy(vec)
        data, _, _ = collect_recovery(policy, vec, [321])
        update_reward_only(policy, data, epochs=1)
        path = tmp_path / "checkpoint_0001"
        save(path, policy, {"cohort": 1})
        update_reward_only(policy, data, epochs=1)
        second = make_policy(vec)
        restore(second, path)
        update_reward_only(second, data, epochs=1)
        assert all(
            torch.equal(v, second.state_dict()[k])
            for k, v in policy.state_dict().items()
        )
    finally:
        vec.close()


def test_evaluation_commits_complete_groups_and_resume_skips_them(
    tmp_path, monkeypatch
):
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=2, obstacles=0)])
    try:
        policy = make_policy(vec)
        first = train_recovery_ppo.evaluate(policy, vec, tmp_path, 65, 2)
        assert first["deterministic"]["tasks"] == 2
        assert first["stochastic"]["physical_steps"] == 4

        def unexpected(*args, **kwargs):
            raise AssertionError("completed evaluation group must not rerun")

        monkeypatch.setattr(train_recovery_ppo, "collect_cohort", unexpected)
        assert train_recovery_ppo.evaluate(policy, vec, tmp_path, 65, 2) == first
    finally:
        vec.close()
