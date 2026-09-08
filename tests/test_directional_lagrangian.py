from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.cohort import (
    CohortNavigation,
    complete_task_cost,
    projected_multiplier,
    terminal_gae,
)
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.lagrangian import (
    TwoValuePolicy,
    clipped_objective,
    collect_cohort,
    update_policy,
)


def make_policy(env):
    torch.manual_seed(101)
    return TwoValuePolicy(
        env.observation_space,
        env.action_space,
        lambda _: 3e-4,
        features_extractor_class=DirectionalLidarExtractor,
        features_extractor_kwargs={"remaining_time": True},
        share_features_extractor=False,
        net_arch={"pi": [32], "vf": [32]},
    )


def test_multiplier_increases_for_excess_risk_and_projects():
    assert projected_multiplier(0.0, 0.4, 0.05, 1.0) == pytest.approx(0.35)
    assert projected_multiplier(0.1, 0.0, 0.05, 1.0) == pytest.approx(0.05)
    assert projected_multiplier(0.01, 0.0, 0.05, 1.0) == 0.0


def test_incomplete_cohort_not_accepted():
    seeds = list(range(8))
    short = [{"seed": i, "outcome": "contact"} for i in range(7)]
    with pytest.raises(ValueError):
        complete_task_cost(short, seeds)
    assert (
        complete_task_cost(short + [{"seed": 7, "outcome": "timeout"}], seeds) == 0.875
    )


def test_mc_event_cost_does_not_discount_late_contact_or_bootstrap_timeout():
    _, late = terminal_gae(np.array([0.0, 0.0, 1.0]), np.array([0.2, 0.4, 0.7]), 1.0)
    _, early = terminal_gae(np.array([1.0]), np.array([0.2]), 1.0)
    _, timeout = terminal_gae(np.zeros(3), np.full(3, 0.8), 1.0)
    assert np.allclose(late, 1.0) and np.allclose(early, 1.0)
    assert np.allclose(timeout, 0.0)


def test_cost_advantage_decreases_unsafe_action_probability():
    logits = torch.zeros(2, requires_grad=True)
    old = torch.full((2,), float(np.log(0.5)))
    reward = torch.tensor([2.0, 1.0])
    cost = torch.tensor([1.0, 0.0])
    loss = clipped_objective(
        logits.log_softmax(0), old, reward, cost, 4.0, normalize=False
    )
    loss.backward()
    assert logits.grad[0] > 0.0 and logits.grad[1] < 0.0
    logits.grad.zero_()
    clipped_objective(
        logits.log_softmax(0), old, reward, cost, 0.0, normalize=False
    ).backward()
    assert logits.grad[0] < 0.0  # unconstrained preference is reversed by cost


def test_parked_slot_does_not_take_extra_physics_steps():
    env = CohortNavigation(horizon=1, obstacles=0)
    env.reset(seed=123)
    env.step(np.zeros(3))
    time = env.base.simulation_time
    env.reset()  # VecEnv's automatic reset parks this slot.
    for _ in range(3):
        _, r, done, truncated, info = env.step(np.ones(3))
        assert info["parked"] and r == 0.0 and not done and not truncated
    assert env.base.simulation_time == time
    env.reset(seed=124)
    assert not env.parked
    env.close()


def test_exact_one_state_cmdp_actor_learns_opposite_preferences():
    outcomes = []
    for multiplier in (0.0, 4.0):
        logits = torch.zeros(2, requires_grad=True)
        optimizer = torch.optim.SGD([logits], lr=0.5)
        for _ in range(80):
            old_logp = logits.detach().log_softmax(0)
            weights = old_logp.exp() * 2.0  # exact on-policy expectation, two actions
            loss = clipped_objective(
                logits.log_softmax(0),
                old_logp,
                torch.tensor([2.0, 1.0]) * weights,
                torch.tensor([1.0, 0.0]) * weights,
                multiplier,
                normalize=False,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        outcomes.append(logits.detach().softmax(0)[0].item())
    assert outcomes[0] > 0.8 and outcomes[1] < 0.2


def test_partial_cohort_does_not_include_parked_slots_as_safe_tasks():
    env = DummyVecEnv(
        [lambda: CohortNavigation(horizon=2, obstacles=0) for _ in range(2)]
    )
    policy = make_policy(env)
    data, rows, audit = collect_cohort(policy, env, [123])
    assert len(rows) == 1 and len(data["ar"]) == 2
    assert audit["physical_steps"] == 2 and audit["parked_slots"] == 2
    env.close()


def test_complete_collection_update_and_optimizer_roundtrip(tmp_path):
    env = DummyVecEnv(
        [lambda: CohortNavigation(horizon=2, obstacles=0) for _ in range(2)]
    )
    policy = make_policy(env)
    data, rows, audit = collect_cohort(policy, env, [123, 124])
    assert len(rows) == 2 and len(data["ar"]) == 4
    assert audit["physical_steps"] == 4 and audit["task_contact_rate"] == 0.0
    assert np.allclose(data["rc"], 0.0)
    actor_ids = {id(p) for p in policy.pi_features_extractor.parameters()}
    critic_ids = {id(p) for p in policy.vf_features_extractor.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    metrics = update_policy(policy, data, 0.5, epochs=2, batch_size=4)
    assert metrics["gradient_steps"] > 0
    assert policy.cost_net.weight in policy.optimizer.state
    path = tmp_path / "state.pt"
    torch.save(
        {"model": policy.state_dict(), "optimizer": policy.optimizer.state_dict()}, path
    )
    restored = make_policy(env)
    payload = torch.load(path, weights_only=True)
    restored.load_state_dict(payload["model"])
    restored.optimizer.load_state_dict(payload["optimizer"])
    obs = torch.as_tensor(data["obs"])
    with torch.no_grad():
        a = policy.evaluate_both(obs, deterministic=True)
        b = restored.evaluate_both(obs, deterministic=True)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    env.close()
