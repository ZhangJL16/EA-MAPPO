from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.UAVEnergyDeliverySAC import StaticCylinderObstacle
from experiments.directional_navigation.environment import FirstContactNavigation
from experiments.directional_navigation.features import DirectionalLidarExtractor


def space(time: bool = False) -> gym.spaces.Box:
    return gym.spaces.Box(-1.0, 1.0, (2055 + int(time),), dtype=np.float32)


def bearing_inputs() -> torch.Tensor:
    x = torch.zeros(4, 2055)
    x[:, :7] = torch.tensor([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.7])
    x[:, 7:1031] = 1.0
    for i in range(4):
        ranges = x[i, 7:1031].reshape(8, 128)
        hits = x[i, 1031:].reshape(8, 128)
        ranges[:, i * 32 : i * 32 + 8] = 0.2
        hits[:, i * 32 : i * 32 + 8] = 1.0
    return x


def test_ordered_readout_breaks_bearing_alias_without_extra_backbone_parameters():
    torch.manual_seed(41)
    ordered = DirectionalLidarExtractor(space())
    control = DirectionalLidarExtractor(space(), readout="global_tiled")
    control.load_state_dict(ordered.state_dict())
    x = bearing_inputs()
    a, b = ordered(x), control(x)
    assert a.shape == b.shape == (4, 2080)
    assert torch.allclose(b, b[:1].expand_as(b), atol=1e-6)
    for i in range(1, 4):
        assert (a[i] - a[0]).abs().max() > 0.01
    assert sum(p.numel() for p in ordered.parameters()) == 6256


def test_goal_time_sensitivity_and_gradient():
    torch.manual_seed(43)
    model = DirectionalLidarExtractor(space(True), remaining_time=True)
    x = torch.cat([bearing_inputs(), torch.ones(4, 1)], 1).requires_grad_()
    y = model(x)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, 7:1031].abs().sum() > 0
    changed = x.detach().clone()
    changed[:, 3:6] *= -1
    assert not torch.allclose(model(changed)[:, :32], y[:, :32])
    changed[:, -1] = 0.3
    assert torch.allclose(model(changed)[:, -1], torch.full((4,), 0.3))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": (0, 16)},
        {"grid": (3, 16)},
        {"grid": (2, 129)},
        {"readout": "bad"},
        {"remaining_time": True},
    ],
)
def test_invalid_contract(kwargs):
    with pytest.raises(ValueError):
        DirectionalLidarExtractor(space(), **kwargs)


class ShapeEnv(gym.Env):
    observation_space = space()
    action_space = gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return bearing_inputs()[0].numpy(), {}

    def step(self, action):
        return bearing_inputs()[0].numpy(), 0.0, True, False, {}


def test_sb3_checkpoint_roundtrip_and_separate_extractors(tmp_path):
    env = DummyVecEnv([ShapeEnv])
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=4,
        batch_size=4,
        seed=13,
        device="cpu",
        policy_kwargs={
            "features_extractor_class": DirectionalLidarExtractor,
            "share_features_extractor": False,
            "net_arch": {"pi": [32], "vf": [32]},
        },
    )
    assert model.policy.pi_features_extractor is not model.policy.vf_features_extractor
    x = env.reset()
    before = model.predict(x, deterministic=True)[0]
    model.save(tmp_path / "model")
    restored = PPO.load(tmp_path / "model", device="cpu")
    assert np.array_equal(before, restored.predict(x, deterministic=True)[0])
    env.close()


def test_finite_deadline_not_bootstrapped_as_time_limit():
    env = FirstContactNavigation(horizon=2, obstacles=0)
    obs, _ = env.reset(seed=9)
    assert obs[-1] == 1.0
    obs, _, done, truncated, _ = env.step(np.zeros(3))
    assert obs[-1] == 0.5 and not done and not truncated
    obs, _, done, truncated, info = env.step(np.zeros(3))
    assert obs[-1] == 0.0 and done and not truncated
    assert info["cost"] == 0.0 and info["navigation_episode"]["outcome"] == "timeout"
    with pytest.raises(RuntimeError):
        env.step(np.zeros(3))
    env.close()


def test_first_contact_keeps_physical_repair_and_no_progress_credit():
    env = FirstContactNavigation(obstacles=0)
    env.reset(
        seed=701,
        options={
            "start_position": np.array([1000.0, 1000.0, 100.0], np.float32),
            "start_velocity": np.array([20.0, 0.0, 0.0], np.float32),
            "task_point": np.array([1500.0, 1000.0, 100.0], np.float32),
        },
    )
    base = env.unwrapped
    base.obstacles = [StaticCylinderObstacle(np.array([1001.0, 1000.0]), 1.0)]
    base._update_lidar()
    _, reward, done, truncated, info = env.step(np.zeros(3))
    assert done and not truncated and info["cost"] == 1.0
    assert info["navigation_episode"]["outcome"] == "contact"
    assert reward < 0.0 and info["physics_substeps"] == 4
    assert np.allclose(base.agent.vel, 0.0)
    # The physical repair may land exactly on the inflated cylinder surface;
    # the placement sampler's stricter clearance predicate is a different contract.
    distance = np.linalg.norm(base.agent.pos[:2] - base.obstacles[0].pos)
    assert distance >= base.obstacles[0].radius + base.safe_radius - 1e-5
    env.close()


def test_contact_overrides_goal(monkeypatch):
    env = FirstContactNavigation(obstacles=0)
    obs, _ = env.reset(seed=8)
    components = env.unwrapped._zero_reward_components()
    components.update(
        task_completion_reward_component=100.0,
        progress_reward_component=10.0,
        time_penalty_component=-0.01,
        obstacle_penalty_component=-1.2,
    )
    monkeypatch.setattr(
        env.env,
        "step",
        lambda action: (
            obs[:-1],
            108.79,
            True,
            False,
            {
                "obstacle_collision": True,
                "boundary_contact": False,
                "is_success": True,
                "reward_components": components,
                "realized_energy_cost": 1.0,
                "physics_substeps": 4,
            },
        ),
    )
    _, reward, done, truncated, info = env.step(np.zeros(3))
    assert done and not truncated and not info["is_success"]
    assert reward == pytest.approx(-0.0121)
    env.close()


def test_vector_autoreset_keeps_terminal_observation():
    vec = DummyVecEnv([lambda: FirstContactNavigation(horizon=1, obstacles=0)])
    vec.reset()
    obs, _, dones, infos = vec.step(np.zeros((1, 3)))
    assert dones[0] and obs[0, -1] == 1.0
    assert infos[0]["terminal_observation"][-1] == 0.0
    assert not infos[0]["TimeLimit.truncated"]
    vec.close()
