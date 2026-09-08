"""Regression lock for the user's collision and nonterminal recovery contract."""

import numpy as np
import pytest

from envs.UAVEnergyDeliverySAC import StaticCylinderObstacle
from experiments.directional_navigation.cohort import terminal_gae
from experiments.directional_navigation.recovery import RecoveryCohort


def test_repeated_collision_cost_is_count_to_go_not_probability():
    costs = np.array([1.0, 0.0, 1.0, 0.0], np.float32)
    _, returns = terminal_gae(costs, np.zeros(4, np.float32), 1.0)
    np.testing.assert_array_equal(returns, [2.0, 1.0, 1.0, 0.0])


def make_env(horizon=20):
    env = RecoveryCohort(horizon=horizon, obstacles=0)
    env.reset(
        seed=701,
        options={
            "start_position": np.array([1000.0, 1000.0, 100.0], np.float32),
            "start_velocity": np.zeros(3, np.float32),
            "task_point": np.array([1500.0, 1000.0, 100.0], np.float32),
        },
    )
    return env


def test_reward_is_fixed_repeat_discount_not_geometric():
    env = make_env()
    try:
        base = env.base
        penalties = []
        for contact in (True, True, True, False, True):
            base.obstacles = (
                [StaticCylinderObstacle(np.array([1001.0, 1000.0]), 1.0)]
                if contact
                else []
            )
            base.agent.pos[:] = [1000.0, 1000.0, 100.0]
            base.agent.vel[:] = 0.0
            _, reward, done, _, info = env.step(np.zeros(3))
            assert not done
            penalties.append(info["raw_reward"] + base.time_penalty)
            assert reward == pytest.approx(info["raw_reward"] * 0.01)
        # Clean step has zero velocity/progress at the injected fixed location.
        np.testing.assert_allclose(
            penalties, [-1.2, -0.42, -0.42, 0.0, -1.2], atol=1e-5
        )
        assert env.collision_count == 4
        assert "boundary_collision_count" not in info
        assert "obstacle_collision_count" not in info
    finally:
        env.close()


def test_repair_zeroes_all_velocity_and_does_not_terminate():
    env = make_env()
    try:
        base = env.base
        base.obstacles = [StaticCylinderObstacle(np.array([1001.0, 1000.0]), 1.0)]
        base.agent.vel[:] = [20.0, 2.0, 3.0]
        _, _, done, truncated, info = env.step(np.zeros(3))
        assert not done and not truncated and info["cost"] == 1.0
        np.testing.assert_array_equal(base.agent.vel, np.zeros(3))
        assert np.linalg.norm(base.agent.pos[:2] - base.obstacles[0].pos) >= 1.5 - 1e-5
        env.step(np.array([-1.0, 0.0, 0.0]))  # Same task can take the next action.
        assert env.episode_seed == 701 and env.steps == 2
        base.agent.prev_pos[:] = [-1.0, 1000.0, 100.0]
        base.agent.vel[:] = [1.0, 2.0, 3.0]
        assert base._apply_boundary_constraints(base.agent)[0]
        np.testing.assert_array_equal(base.agent.vel, np.zeros(3))
    finally:
        env.close()


def test_simultaneous_contacts_count_once_and_goal_after_contact_not_safe(monkeypatch):
    env = make_env()
    try:
        base = env.base
        observation = base._active_goal_sac_observation()

        def step(action):
            base.agent.prev_collided = bool(base.agent.collided)
            base.agent.collided = True
            return (
                observation,
                0.0,
                env.steps == 1,
                False,
                {
                    "obstacle_collision": True,
                    "boundary_contact": True,
                    "is_success": env.steps == 1,
                    "reward_components": base._reward_components(
                        progress=0.0,
                        goal=base.active_goal,
                        velocity=np.zeros(3),
                        task_completed=env.steps == 1,
                        boundary_contact=True,
                        obstacle_collision=True,
                        safety_intervention_norm=0.0,
                    ),
                    "realized_energy_cost": 1.0,
                    "physics_substeps": 4,
                },
            )

        monkeypatch.setattr(base, "step", step)
        _, _, done, _, info = env.step(np.zeros(3))
        assert not done and info["collision_count"] == 1
        assert info["raw_reward"] == pytest.approx(-2.4 - base.time_penalty)
        _, _, done, _, info = env.step(np.zeros(3))
        assert done and info["collision_count"] == 2
        assert info["raw_reward"] == pytest.approx(-0.84 - base.time_penalty)
        row = info["navigation_episode"]
        assert row["goal_reached"] and not row["safe_goal"]
        assert row["termination"] == "goal"
    finally:
        env.close()


def test_reward_settings_locked_and_deadline_parking():
    env = make_env(horizon=2)
    try:
        for key in (
            "boundary_penalty",
            "obstacle_collision_penalty",
            "repeat_collision_scale",
        ):
            with pytest.raises(ValueError, match="user-locked"):
                setattr(env.base, key, 0.0)
        env.step(np.zeros(3))
        _, _, done, truncated, info = env.step(np.zeros(3))
        assert done and not truncated
        assert info["navigation_episode"]["termination"] == "deadline"
        env.reset()  # VecEnv autoreset parks instead of adding a new task.
        assert env.step(np.zeros(3))[4] == {"parked": True}
        env.reset(seed=702)
        assert not env.parked and env.collision_count == 0
    finally:
        env.close()
