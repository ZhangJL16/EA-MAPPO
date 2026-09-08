"""Focused invariants for critic-only continuation and checkpoint restoration."""

import copy

import numpy as np
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.critic_completion import (
    update_with_critic_completion,
    value_parameters,
)
from experiments.directional_navigation.recovery import RecoveryCohort
from experiments.directional_navigation.recovery_ppo import (
    collect_recovery,
    update_reward_only,
)
from scripts.run_directional_cost_trace_pair import restore
from scripts.run_directional_lagrangian_pair import save
from tests.test_recovery_ppo import make_policy


def test_final_evaluation_resumes_without_repeating_saved_tasks(tmp_path, monkeypatch):
    from scripts import run_recovery_critic_completion as runner

    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=2, obstacles=0)])
    try:
        policy = make_policy(vec)
        result = runner.evaluate(
            policy, vec, tmp_path, tmp_path, "holdout", 791, 2, True
        )
        assert result["tasks"] == 2

        def unexpected(*args, **kwargs):
            raise AssertionError("saved task evaluated twice")

        monkeypatch.setattr(runner, "collect_cohort", unexpected)
        assert (
            runner.evaluate(policy, vec, tmp_path, tmp_path, "holdout", 791, 2, True)
            == result
        )
    finally:
        vec.close()


def test_kl_stop_leaves_policy_fixed_but_value_budget_finishes():
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=4, obstacles=0)])
    try:
        policy = make_policy(vec)
        data, _, _ = collect_recovery(policy, vec, [312])
        # Force the first KL check to stop the actor before any optimizer step.
        data["old_logprob"] = data["old_logprob"] + 5
        data["rr"] = data["rr"] + 1
        before = {k: p.detach().clone() for k, p in policy.named_parameters()}
        value_ids = {id(p) for p in value_parameters(policy)}
        metrics = update_with_critic_completion(
            policy, data, complete=True, epochs=2, batch_size=4
        )
        assert metrics["policy_gradient_steps"] == 0
        assert metrics["critic_gradient_steps"] == 2
        assert metrics["critic_extra_steps"] == 2
        assert (
            metrics["value_target_mse_after_completion"]
            < metrics["value_target_mse_after_joint"]
        )
        assert any(
            not torch.equal(p, before[k])
            for k, p in policy.named_parameters()
            if id(p) in value_ids
        )
        assert all(
            torch.equal(p, before[k])
            for k, p in policy.named_parameters()
            if id(p) not in value_ids
        )
    finally:
        vec.close()


def test_control_is_exact_and_completion_preserves_sampling_rng():
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=4, obstacles=0)])
    try:
        first = make_policy(vec)
        data, _, _ = collect_recovery(first, vec, [313])
        second, third = copy.deepcopy(first), copy.deepcopy(first)
        data["old_logprob"] += 5
        for policy, kind in (
            (first, "old"),
            (second, "control"),
            (third, "completion"),
        ):
            np.random.seed(184)
            torch.manual_seed(91)
            if kind == "old":
                update_reward_only(policy, data, epochs=2, batch_size=4)
            else:
                update_with_critic_completion(
                    policy, data, complete=kind == "completion", epochs=2, batch_size=4
                )
            draws = (np.random.rand(), torch.rand(1).item())
            if kind == "old":
                expected = draws
            else:
                assert draws == expected
        assert all(
            torch.equal(p, second.state_dict()[k])
            for k, p in first.state_dict().items()
        )
    finally:
        vec.close()


def test_existing_checkpoint_restores_critic_completion_optimizer(tmp_path):
    vec = DummyVecEnv([lambda: RecoveryCohort(horizon=4, obstacles=0)])
    try:
        first = make_policy(vec)
        data, _, _ = collect_recovery(first, vec, [314])
        data["old_logprob"] += 5
        update_with_critic_completion(
            first, data, complete=True, epochs=2, batch_size=4
        )
        checkpoint = tmp_path / "checkpoint_0001"
        save(checkpoint, first, {"cohort": 1})
        update_with_critic_completion(
            first, data, complete=True, epochs=2, batch_size=4
        )
        second = make_policy(vec)
        restore(second, checkpoint)
        update_with_critic_completion(
            second, data, complete=True, epochs=2, batch_size=4
        )
        assert all(
            torch.equal(p, second.state_dict()[k])
            for k, p in first.state_dict().items()
        )
        a, b = (
            first.optimizer.state_dict()["state"],
            second.optimizer.state_dict()["state"],
        )
        assert a.keys() == b.keys()
        assert all(
            torch.equal(v, b[k][name])
            for k, entry in a.items()
            for name, v in entry.items()
        )
    finally:
        vec.close()
