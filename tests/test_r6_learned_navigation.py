from __future__ import annotations

import numpy as np
import torch

from envs.UAVEnergyDeliverySAC import SACTrainingPhase, UAVEnergyDeliverySACEnv
from experiments.r6_learned_navigation import (
    ConstraintState,
    PPOConfig,
    R6ModelConfig,
    RecurrentSetActorCritic,
    RolloutBatch,
    compute_gae,
    ppo_update,
)


def small_config() -> R6ModelConfig:
    return R6ModelConfig(
        horizontal_sectors=4,
        vertical_sectors=2,
        core_dim=16,
        ray_dim=16,
        attention_heads=4,
        recurrent_dim=24,
    )


def test_recurrent_set_actor_shapes_bounds_and_finite_gradients() -> None:
    torch.manual_seed(7)
    config = small_config()
    model = RecurrentSetActorCritic(config)
    observation = torch.rand(3, config.observation_dim)
    observation[:, :6] = 2.0 * observation[:, :6] - 1.0
    feedback = torch.zeros(3, config.feedback_dim)
    hidden = model.initial_hidden(3, device=torch.device("cpu"))
    starts = torch.tensor([1.0, 0.0, 1.0])
    action, log_prob, entropy, values, next_hidden = model.step(
        observation, feedback, hidden, starts
    )
    assert action.shape == (3, 3)
    assert log_prob.shape == (3,)
    assert entropy.shape == (3,)
    assert values.shape == (3, 3)
    assert next_hidden.shape == (3, config.recurrent_dim)
    assert torch.all(action >= -1.0) and torch.all(action <= 1.0)
    loss = action.square().mean() + values.square().mean() - 0.01 * entropy.mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_sequence_likelihood_matches_step_recurrence() -> None:
    torch.manual_seed(11)
    config = small_config()
    model = RecurrentSetActorCritic(config)
    length, batch = 4, 2
    observations = torch.rand(length, batch, config.observation_dim)
    observations[..., :6] = 2.0 * observations[..., :6] - 1.0
    feedback = torch.randn(length, batch, config.feedback_dim) * 0.1
    starts = torch.zeros(length, batch)
    starts[0] = 1.0
    starts[2, 1] = 1.0
    hidden = model.initial_hidden(batch, device=torch.device("cpu"))
    actions = []
    step_log_probs = []
    step_values = []
    for index in range(length):
        action, log_prob, _entropy, values, hidden = model.step(
            observations[index],
            feedback[index],
            hidden,
            starts[index],
            deterministic=True,
        )
        actions.append(action)
        step_log_probs.append(log_prob)
        step_values.append(values)
    sequence_log_probs, _entropy, sequence_values, sequence_means = model.evaluate_sequence(
        observations,
        feedback,
        model.initial_hidden(batch, device=torch.device("cpu")),
        starts,
        torch.stack(actions),
    )
    assert torch.allclose(sequence_log_probs, torch.stack(step_log_probs), atol=1e-6)
    assert torch.allclose(sequence_values, torch.stack(step_values), atol=1e-6)
    assert torch.allclose(sequence_means, torch.stack(actions), atol=1e-6)


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


def test_constraint_dual_ascent_is_projected_and_cost_specific() -> None:
    constraints = ConstraintState(
        collision_multiplier=0.2,
        intervention_multiplier=0.3,
        collision_budget=0.1,
        intervention_budget=0.4,
        dual_learning_rate=2.0,
        maximum_multiplier=0.5,
    )
    constraints.update(collision_mean=1.0, intervention_mean=0.0)
    assert constraints.collision_multiplier == 0.5
    assert constraints.intervention_multiplier == 0.0


def test_ppo_update_uses_nominal_likelihood_and_executed_distillation() -> None:
    torch.manual_seed(13)
    rng = np.random.default_rng(13)
    config = small_config()
    model = RecurrentSetActorCritic(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    time_steps, environments = 4, 2
    observations = torch.rand(time_steps, environments, config.observation_dim)
    observations[..., :6] = 2.0 * observations[..., :6] - 1.0
    feedback = torch.zeros(time_steps, environments, config.feedback_dim)
    episode_starts = torch.zeros(time_steps, environments)
    episode_starts[0] = 1.0
    hiddens = torch.zeros(time_steps, environments, config.recurrent_dim)
    hidden = model.initial_hidden(environments, device=torch.device("cpu"))
    actions = []
    log_probs = []
    values = []
    for index in range(time_steps):
        hiddens[index] = hidden
        action, log_prob, _entropy, step_values, hidden = model.step(
            observations[index],
            feedback[index],
            hidden,
            episode_starts[index],
        )
        actions.append(action.detach())
        log_probs.append(log_prob.detach())
        values.append(step_values.detach())
    nominal = torch.stack(actions)
    executed = (0.5 * nominal).clamp(-1.0, 1.0)
    rollout = RolloutBatch(
        observations=observations,
        feedback=feedback,
        hiddens=hiddens,
        episode_starts=episode_starts,
        actions=nominal,
        executed_actions=executed,
        old_log_probs=torch.stack(log_probs),
        values=torch.stack(values),
        rewards=torch.randn(time_steps, environments),
        collision_costs=torch.zeros(time_steps, environments),
        intervention_costs=torch.full((time_steps, environments), 0.2),
        dones=torch.zeros(time_steps, environments),
        last_values=torch.zeros(environments, 3),
    )
    constraints = ConstraintState(dual_learning_rate=0.1)
    metrics = ppo_update(
        model,
        optimizer,
        rollout,
        constraints,
        PPOConfig(
            update_epochs=1,
            sequence_length=2,
            minibatch_sequences=2,
            target_kl=1.0,
        ),
        device=torch.device("cpu"),
        rng=rng,
    )
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["distillation_loss"] > 0.0
    assert constraints.collision_multiplier == 0.0
    assert constraints.intervention_multiplier > 0.0


def test_environment_exposes_nominal_and_executed_actions_for_r6() -> None:
    environment = UAVEnergyDeliverySACEnv(
        phase=SACTrainingPhase.NAVIGATION,
        lidar_enabled=True,
        lidar_horizontal_sectors=8,
        lidar_vertical_sectors=2,
        num_obstacles=1,
        cbf_enabled=True,
        max_steps_per_task=20,
        phase1_episode_max_policy_steps=20,
    )
    environment.reset(seed=19)
    _observation, _reward, _terminated, _truncated, info = environment.step(
        np.asarray([0.2, -0.1, 0.3], dtype=np.float32)
    )
    assert np.asarray(info["nominal_action"]).shape == (3,)
    assert np.asarray(info["executed_action"]).shape == (3,)
    assert np.isfinite(float(info["hocbf_intervention_norm"]))
    assert np.isfinite(float(info["progress"]))
    environment.close()
