"""Necessary semantics, real gradients, initialization, and resume regressions."""

from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.directional_navigation.correction_checkpoint import load_correction_checkpoint
from experiments.directional_navigation.correction_sac import CorrectionSAC
from experiments.directional_navigation.correction_supervision import (
    CorrectionRecovery, FreshHOCBFTeacher, TeacherConfig, correction_loss, physical_actions,
)
from experiments.directional_navigation.standard_baselines import save_checkpoint
from scripts.run_hocbf_correction_sac import DEFAULT_SOURCE, make_model, parser


def make_env():
    return CorrectionRecovery(seed_start=120001, seed_stride=1, horizon=7, obstacles=0)


def sensor_fixture():
    teacher = FreshHOCBFTeacher(TeacherConfig())
    obs = np.zeros(2056, np.float32)
    obs[7:1031] = 1
    obs[-1] = 1
    ray = int(np.argmax(teacher.directions[:, 0]))
    direction = teacher.directions[ray]
    obs[7 + ray] = .05
    obs[1031 + ray] = 1
    obs[:3] = 2 * direction / [20., 20., 5.]
    return teacher, obs, direction


def small_model(vec, cls=CorrectionSAC, **kwargs):
    torch.set_num_threads(1)
    return cls("MlpPolicy", vec, seed=32, device="cpu", learning_rate=3e-4,
               batch_size=8, buffer_size=100, learning_starts=8,
               policy_kwargs={"net_arch": [16, 16]}, **kwargs)


def test_physical_map_and_projection_gradient():
    env = make_env()
    actions = torch.tensor([[.8, .9, -.5], [.1, .2, .3]], requires_grad=True)
    physical = physical_actions(actions, TeacherConfig())
    expected = np.stack([env.base._normalized_action_to_acceleration(a)
                         for a in actions.detach().numpy()])
    np.testing.assert_allclose(physical.detach().numpy(), expected, atol=1e-6)
    current = torch.tensor([[1., 0., 0.], [2., 0., 0.]], requires_grad=True)
    target = torch.tensor([[.2, 0., 0.], [0., 0., 0.]], requires_grad=True)
    loss = correction_loss(current, target, torch.tensor([True, False]))
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[.4, 0., 0.], [0., 0., 0.]]))
    assert target.grad is None
    env.close()


def test_fresh_teacher_changes_with_current_action_and_rejects_fallback(monkeypatch):
    teacher, obs, direction = sensor_fixture()
    unsafe = 3. * direction
    corrected, valid, reason = teacher.project(obs, unsafe)
    assert valid and reason == "projected"
    assert np.linalg.norm(corrected - unsafe) > .1
    safe, valid, _ = teacher.project(obs, corrected)
    assert valid
    np.testing.assert_allclose(safe, corrected, atol=1e-5)
    original = teacher.filter.filter_lidar_points

    def fail(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, diagnostics=replace(result.diagnostics, fallback_used=True))

    monkeypatch.setattr(teacher.filter, "filter_lidar_points", fail)
    _, valid, reason = teacher.project(obs, unsafe)
    assert not valid and reason == "solver_invalid"
    with pytest.raises(ValueError, match="nonfinite"):
        teacher.project(np.full(2056, np.nan), unsafe)


def test_teacher_matches_anchor_filter_not_substep_average():
    env = make_env()
    try:
        obs, _ = env.reset(seed=91, options={
            "start_position": np.array([5., 1000., 100.]),
            "start_velocity": np.array([-2., 0., 0.]),
            "task_point": np.array([1500., 1000., 100.])})
        action = np.array([.3, 0., 0.], np.float32)
        physical = env.base._normalized_action_to_acceleration(action)
        target, valid, _ = FreshHOCBFTeacher(TeacherConfig()).project(obs, physical)
        actual, diagnostics = env.base._safety_filtered_action(action)
        assert valid and not diagnostics["emergency_brake"]
        np.testing.assert_allclose(target, env.base._normalized_action_to_acceleration(actual), atol=2e-4)
        # The native solver fails for this more aggressive query. Such output
        # is execution fallback, NOT an admissible teacher demonstration.
        unsafe = np.array([-.8, 0., 0.], np.float32)
        _, valid, reason = FreshHOCBFTeacher(TeacherConfig()).project(
            obs, env.base._normalized_action_to_acceleration(unsafe))
        _, diagnostics = env.base._safety_filtered_action(unsafe)
        assert not valid and reason == "solver_invalid" and diagnostics["fallback_used"]
    finally:
        env.close()


def test_locked_recovery_and_effective_reward_unchanged():
    from envs.UAVEnergyDeliverySAC import StaticCylinderObstacle
    env = make_env()
    env.reset(seed=91, options={"start_position": np.array([1000., 1000., 100.]),
                               "task_point": np.array([1500., 1000., 100.])})
    # Force unavoidable contact, independently of HOCBF, to audit repair rules.
    penalties = []
    try:
        for contact in (True, True, True, False, True):
            env.base.obstacles = [StaticCylinderObstacle(np.array([1001., 1000.]), 1.)] if contact else []
            env.base.agent.pos[:] = [1000., 1000., 100.]
            env.base.agent.vel[:] = 0
            _, reward, done, _, info = env.step(np.zeros(3))
            penalties.append(info["raw_reward"] + env.base.time_penalty)
            assert not done and info["cost"] == int(contact)
            assert reward == pytest.approx(info["raw_reward"] * .01)
            if contact:
                np.testing.assert_array_equal(env.base.agent.vel, np.zeros(3))
            assert not any("boundary_collision" in k or "obstacle_collision" in k for k in info)
        np.testing.assert_allclose(penalties, [-1.2, -.42, -.42, 0., -1.2], atol=1e-5)
        assert env.collision_count == 4
        assert env.base.safety_intervention_penalty == 0
    finally:
        env.close()


def test_zero_weight_update_is_bit_identical_to_native_sac():
    vec_a, vec_b = DummyVecEnv([make_env]), DummyVecEnv([make_env])
    try:
        reference = small_model(vec_a, SAC)
        vec_a._reset_seeds()
        reference.learn(24, reset_num_timesteps=False)
        actual = small_model(vec_b, correction_weight=0.)
        vec_b._reset_seeds()
        actual.learn(24, reset_num_timesteps=False)
        for key, value in reference.policy.state_dict().items():
            torch.testing.assert_close(value, actual.policy.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(reference.log_ent_coef, actual.log_ent_coef, rtol=0, atol=0)
        assert actual.correction_totals["queries"] == 0
    finally:
        vec_a.close()
        vec_b.close()


def test_real_correction_updates_actor_and_exact_checkpoint_resume(tmp_path):
    vec, restored_vec = DummyVecEnv([make_env]), DummyVecEnv([make_env])
    try:
        model = small_model(vec, correction_interval=1, correction_batch_size=4)
        vec._reset_seeds()
        model.learn(16, reset_num_timesteps=False)
        _, obs, _ = sensor_fixture()
        # A targeted replay fixture ensures the new loss is genuinely nonzero.
        for _ in range(20):
            model.replay_buffer.add(obs[None], obs[None], np.array([[.8, 0., 0.]]),
                                    np.zeros(1), np.zeros(1), [{}])
        with torch.no_grad():
            model.actor.mu.weight.zero_()
            model.actor.mu.bias[:] = torch.tensor([2., 0., 0.])
            model.actor.log_std.weight.zero_()
            model.actor.log_std.bias.fill_(-4.)
        before = model.actor.mu.bias.detach().clone()
        model.train(2, batch_size=8)
        assert model.correction_totals["nonzero_updates"] > 0
        assert not torch.equal(before, model.actor.mu.bias)
        save_checkpoint(model, vec, tmp_path, {"test": "correction"}, {"marker": 1})
        model.learn(8, reset_num_timesteps=False)
        expected_obs = model._last_obs.copy()
        restored, records = load_correction_checkpoint(restored_vec, tmp_path,
                                                       {"test": "correction"}, "cpu")
        restored.learn(8, reset_num_timesteps=False)
        assert isinstance(restored, CorrectionSAC) and records == {"marker": 1}
        np.testing.assert_array_equal(restored._last_obs, expected_obs)
        for key, value in model.policy.state_dict().items():
            torch.testing.assert_close(value, restored.policy.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(model.log_ent_coef, restored.log_ent_coef, rtol=0, atol=0)
        for key, value in model.correction_totals.items():
            if key != "teacher_seconds":
                assert restored.correction_totals[key] == value
        assert restored.replay_buffer.pos == model.replay_buffer.pos
        with pytest.raises(ValueError, match="contract"):
            load_correction_checkpoint(restored_vec, tmp_path, {}, "cpu")
    finally:
        vec.close()
        restored_vec.close()


def test_actual_old_checkpoint_warm_start_preserves_function(tmp_path):
    vec = DummyVecEnv([make_env])
    try:
        args = parser().parse_args(["--output-dir", str(tmp_path), "--device", "cpu",
                                   "--buffer-size", "100", "--num-envs", "1"])
        model = make_model(vec, args)
        source = SAC.load(DEFAULT_SOURCE, device="cpu")
        obs = vec.reset()
        np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0],
                                      source.predict(obs, deterministic=True)[0])
        for key, value in source.policy.state_dict().items():
            torch.testing.assert_close(value, model.policy.state_dict()[key], rtol=0, atol=0)
        assert model.num_timesteps == 0 and model.source_transitions == 524288
        assert model.replay_buffer.size() == 0
        assert not model.actor.optimizer.state and not model.critic.optimizer.state
        torch.testing.assert_close(model.log_ent_coef, source.log_ent_coef, rtol=0, atol=0)
    finally:
        vec.close()


def test_runner_preparation_cannot_train_and_is_idempotent(tmp_path):
    command = [sys.executable, "scripts/run_hocbf_correction_sac.py",
               "--output-dir", str(tmp_path / "prepared"), "--device", "cpu"]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)["training_started"] is False
    subprocess.run(command, check=True, capture_output=True, text=True)
    root = tmp_path / "prepared"
    assert json.loads((root / "status.json").read_text())["status"] == "READY_NOT_STARTED"
    assert not (root / "sac").exists()


def test_single_tiny_subprocess_runner_checkpoint(tmp_path):
    root = tmp_path / "smoke"
    command = [sys.executable, "scripts/run_hocbf_correction_sac.py",
        "--output-dir", str(root), "--device", "cpu", "--start",
        "--additional-steps", "16", "--num-envs", "1", "--rollout-steps", "8",
        "--checkpoint-steps", "8", "--batch-size", "8", "--buffer-size", "32",
        "--learning-starts", "8", "--horizon", "7", "--obstacles", "0",
        "--correction-batch-size", "2", "--correction-interval", "1"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=90)
    status = json.loads((root / "status.json").read_text())
    assert status["status"] == "TRAINING_COMPLETE_AWAITING_USER"
    assert status["adaptation_transitions"] == 16
    assert status["correction_totals"]["queries"] > 0
    assert not status["automatic_evaluation"]
    assert (root / "sac/checkpoint_000000016/state.pt").exists()
