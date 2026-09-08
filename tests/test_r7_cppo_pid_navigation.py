from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.r7_cppo_pid import (
    CPPOPIDConfig,
    PIDLagrangian,
    PIDLagrangianConfig,
    R7ActorCritic,
    R7ModelConfig,
    RolloutBatch,
    compute_gae,
    cppo_pid_update,
)
from scripts.train_r7_cppo_pid_navigation import (
    _step_cost,
    environment_kwargs,
    load_evaluation_tasks,
    load_evaluation_tasks_by_source_indices,
    parse_args,
)
from scripts.run_r7_pilot_then_500k import promotion_decision


def small_model() -> R7ActorCritic:
    return R7ActorCritic(
        R7ModelConfig(
            horizontal_sectors=8,
            vertical_sectors=2,
            hidden_dim=32,
        )
    )


def test_r7_reuses_structured_lidar_and_separates_actor_critic() -> None:
    torch.manual_seed(3)
    model = small_model()
    observations = torch.rand(5, model.observation_dim)
    observations[:, :6] = 2.0 * observations[:, :6] - 1.0
    actions, latents, log_probs, entropies, values = model.act(observations)
    assert actions.shape == (5, 3)
    assert latents.shape == (5, 3)
    assert log_probs.shape == (5,)
    assert entropies.shape == (5,)
    assert values.shape == (5, 2)
    assert torch.all(actions >= -1.0) and torch.all(actions <= 1.0)
    audit = model.architecture_audit()
    assert audit["actor_critic_features_shared"] is False
    assert audit["critic_heads"] == ["reward_value", "scalar_cmdp_cost_value"]
    assert audit["actor_extractor"]["lidar_tensor_shape"] == [2, 2, 8]
    actor_ids = {id(parameter) for parameter in model.actor_parameters()}
    critic_ids = {id(parameter) for parameter in model.critic_parameters()}
    assert actor_ids.isdisjoint(critic_ids)


def test_squashed_gaussian_stored_latent_likelihood_is_exact() -> None:
    torch.manual_seed(5)
    model = small_model()
    observations = torch.rand(4, model.observation_dim)
    _actions, latents, old_log_probs, _entropy, _values = model.act(observations)
    new_log_probs, _new_entropy = model.evaluate_latents(observations, latents)
    assert torch.allclose(new_log_probs, old_log_probs, atol=1e-6)


def test_pid_lagrangian_recovers_below_positive_cost_limit() -> None:
    pid = PIDLagrangian(
        PIDLagrangianConfig(
            kp=0.1,
            ki=0.01,
            kd=0.0,
            proportional_ema_alpha=0.0,
            derivative_ema_alpha=0.0,
            cost_limit=10.0,
            initial_integral=0.0,
        )
    )
    high = pid.update(20.0)
    assert high > 0.0
    for _ in range(5):
        pid.update(0.0)
    assert pid.penalty == pytest.approx(0.0)
    assert pid.integral == pytest.approx(0.0)


def test_pid_lagrangian_state_round_trip() -> None:
    config = PIDLagrangianConfig(cost_limit=10.0)
    source = PIDLagrangian(config)
    source.update(20.0)
    restored = PIDLagrangian(config)
    restored.load_state_dict(source.state_dict())
    assert restored.state_dict() == source.state_dict()


def test_evaluation_tasks_are_stratified_across_requested_buckets(tmp_path) -> None:
    rows = []
    for bucket in ("100-500", "500-1500", "1500-2500"):
        for index in range(4):
            rows.append({"distance_bucket": bucket, "id": f"{bucket}-{index}"})
    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        __import__("json").dumps({"seed": 17, "tasks": rows}), encoding="utf-8"
    )
    tasks, seed = load_evaluation_tasks(
        task_file, 6, ("100-500", "500-1500", "1500-2500")
    )
    assert seed == 17
    assert [task["distance_bucket"] for task in tasks] == [
        "100-500",
        "500-1500",
        "1500-2500",
        "100-500",
        "500-1500",
        "1500-2500",
    ]


def test_evaluation_tasks_can_resume_by_exact_source_index(tmp_path) -> None:
    rows = [
        {"distance_bucket": "100-500", "id": index} for index in range(6)
    ]
    task_file = tmp_path / "tasks.json"
    task_file.write_text(
        __import__("json").dumps({"seed": 23, "tasks": rows}), encoding="utf-8"
    )
    tasks, seed = load_evaluation_tasks_by_source_indices(
        task_file, [5, 1, 3], ("100-500",)
    )
    assert seed == 23
    assert [task["source_task_index"] for task in tasks] == [5, 1, 3]
    assert [task["id"] for task in tasks] == [5, 1, 3]


def test_pilot_promotion_requires_strong_safe_navigation() -> None:
    completed = {
        "training_episode_summary": {"successes": 3},
        "final_navigation": {
            "overall_success_rate": 0.84,
            "distance_bucket_success": {
                "100-500": 1.0,
                "500-1500": 0.8,
                "1500-2500": 0.8,
                "2500-4000": 0.6,
                ">4000": 1.0,
            },
            "mean_path_ratio": 1.4,
            "obstacle_collision_steps": 0,
            "boundary_contact_steps": 0,
            "hocbf_intervention_step_rate": 0.1,
        },
    }
    assert promotion_decision(completed)["promote_to_500k"] is True
    completed["final_navigation"]["mean_path_ratio"] = 2.0
    assert promotion_decision(completed)["promote_to_500k"] is False


def test_r7_defaults_to_sampled_data_hocbf_and_squared_correction_cost() -> None:
    args = parse_args(["--output-dir", "unused", "--skip-final-evaluation"])
    assert environment_kwargs(args)["hocbf_sampled_data_robust"] is True
    cost, intervened, collision, correction = _step_cost(
        {
            "hocbf_intervened": True,
            "hocbf_substep_diagnostics": [
                {"intervention_norm": 0.5},
                {"intervention_norm": 1.0},
            ],
            "obstacle_collision": False,
            "boundary_contact": False,
        },
        collision_weight=25.0,
    )
    assert intervened is True
    assert collision is False
    assert correction == pytest.approx((0.5**2 + 1.0**2) / 2.0)
    assert cost == pytest.approx(correction)


def test_r7_collision_is_fail_visible_in_addition_to_correction_cost() -> None:
    cost, _intervened, collision, correction = _step_cost(
        {
            "nominal_action": np.array([1.0, 0.0, 0.0]),
            "executed_action": np.array([0.5, 0.0, 0.0]),
            "obstacle_collision": True,
        },
        collision_weight=25.0,
    )
    assert collision is True
    assert correction == pytest.approx(0.25)
    assert cost == pytest.approx(25.25)


def test_gae_resets_at_episode_boundary() -> None:
    signal = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.zeros_like(signal)
    dones = torch.tensor([[0.0], [1.0], [0.0]])
    advantages, returns = compute_gae(
        signal,
        values,
        dones,
        torch.tensor([4.0]),
        gamma=1.0,
        gae_lambda=1.0,
    )
    assert torch.allclose(advantages[:, 0], torch.tensor([3.0, 2.0, 7.0]))
    assert torch.equal(advantages, returns)


def test_cppo_pid_update_has_separate_finite_actor_and_critic_steps() -> None:
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    model = small_model()
    actor_optimizer = torch.optim.Adam(model.actor_parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(model.critic_parameters(), lr=1e-3)
    time_steps, environments = 4, 2
    observations = torch.rand(time_steps, environments, model.observation_dim)
    with torch.no_grad():
        _actions, latents, log_probs, _entropy, values = model.act(
            observations.reshape(-1, model.observation_dim)
        )
        last_values = model.values(torch.rand(environments, model.observation_dim))
    rollout = RolloutBatch(
        observations=observations,
        latents=latents.reshape(time_steps, environments, 3),
        old_log_probs=log_probs.reshape(time_steps, environments),
        values=values.reshape(time_steps, environments, 2),
        rewards=torch.randn(time_steps, environments),
        costs=torch.rand(time_steps, environments) * 0.1,
        dones=torch.zeros(time_steps, environments),
        last_values=last_values,
    )
    pid = PIDLagrangian(
        PIDLagrangianConfig(cost_limit=1.0, initial_integral=0.0)
    )
    metrics = cppo_pid_update(
        model,
        actor_optimizer,
        critic_optimizer,
        rollout,
        pid,
        CPPOPIDConfig(
            update_epochs=2,
            minibatch_size=4,
            target_kl=1.0,
        ),
        device=torch.device("cpu"),
        rng=rng,
        episode_cost_mean=2.0,
    )
    finite_metrics = {
        key: value for key, value in metrics.items() if key != "episode_cost_mean_for_pid"
    }
    assert all(np.isfinite(value) for value in finite_metrics.values())
    assert metrics["actor_minibatch_updates"] > 0
    assert metrics["critic_minibatch_updates"] > 0
    assert metrics["lagrange_multiplier"] > 0.0
    assert metrics["combined_advantage_std"] > 0.0
