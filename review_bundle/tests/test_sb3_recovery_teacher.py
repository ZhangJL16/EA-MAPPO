from __future__ import annotations

import gc
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from stable_baselines3 import SAC

from envs.certified_uav import EnergyNavigationConfig, PersistentEnergyNavigationEnv, make_random_persistent_uav_env
from scripts.sb3_energy_harness import evaluate_energy_model
from scripts.sb3_recovery_teacher import (
    CertifiedRecoveryOracle,
    CertifiedRecoveryRolloutTeacher,
    TeacherTransition,
    load_and_verify_certificate_artifact,
    prefill_sb3_replay_buffer,
    supervised_actor_warm_start,
    write_certificate_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "certificates/random_persistent_open_energy_x2_certified_execution.json"
NAVIGATION_TEACHER = ROOT / "artifacts/phase1_sb3_sac_1m_gpu/seed0/checkpoint_step_1000000.zip"


class FlightEnergyMultiplierTests(unittest.TestCase):
    @staticmethod
    def environment(multiplier: float) -> PersistentEnergyNavigationEnv:
        return PersistentEnergyNavigationEnv(
            max_episode_steps=5,
            energy_config=EnergyNavigationConfig(flight_energy_multiplier=multiplier),
        )

    def test_one_x_compatibility_and_two_x_realized_cost(self):
        options = {
            "start_position": [2.0, 2.0, 1.0],
            "start_velocity": [0.1, 0.0, 0.0],
            "goal_position": [3.0, 3.0, 1.0],
            "initial_energy_fraction": 1.0,
        }
        one = self.environment(1.0)
        two = self.environment(2.0)
        one.reset(seed=1, options=options)
        two.reset(seed=1, options=options)
        action = np.array([0.5, -0.25, 0.1], dtype=np.float32)
        _, _, _, _, one_info = one.step(action)
        _, _, _, _, two_info = two.step(action)
        self.assertAlmostEqual(one_info["actual_flight_energy"], one_info["base_flight_energy"])
        self.assertAlmostEqual(two_info["actual_flight_energy"], 2.0 * one_info["actual_flight_energy"])
        self.assertAlmostEqual(two_info["base_flight_energy"], one_info["base_flight_energy"])
        self.assertEqual(two_info["flight_energy_multiplier"], 2.0)
        np.testing.assert_array_equal(one.config.a_max, two.config.a_max)
        np.testing.assert_array_equal(one.config.v_max, two.config.v_max)
        self.assertEqual(one.config.dt, two.config.dt)
        self.assertEqual(one.energy_navigation_config.battery_capacity, two.energy_navigation_config.battery_capacity)
        self.assertEqual(one.energy_navigation_config.charging_rate, two.energy_navigation_config.charging_rate)

    def test_multiplier_does_not_scale_gross_charging(self):
        options = {
            "start_position": [0.4, 0.5, 1.0],
            "goal_position": [3.0, 3.0, 1.0],
            "initial_energy_fraction": 0.3,
        }
        values = []
        for multiplier in (1.0, 2.0):
            environment = self.environment(multiplier)
            environment.reset(seed=2, options=options)
            _, _, _, _, info = environment.step(np.zeros(3, dtype=np.float32))
            values.append(info["gross_charge_received"])
        self.assertAlmostEqual(values[0], 0.4)
        self.assertAlmostEqual(values[1], 0.4)

    def test_two_x_depletion_remains_nonterminal_and_nonteleporting(self):
        environment = self.environment(2.0)
        start = np.array([2.0, 2.0, 1.0])
        environment.reset(seed=3, options={
            "start_position": start,
            "goal_position": [3.0, 3.0, 1.0],
            "initial_energy_fraction": 1e-8,
        })
        _, _, terminated, _, info = environment.step(np.zeros(3, dtype=np.float32))
        self.assertFalse(terminated)
        self.assertTrue(info["energy_stranded"])
        np.testing.assert_allclose(environment.state.velocity, np.zeros(3))
        self.assertGreater(np.linalg.norm(environment.state.position - environment.scenario.station_position), 1.0)

    def test_clean_actor_observation_is_unchanged(self):
        one = self.environment(1.0)
        two = self.environment(2.0)
        self.assertEqual(one.observation_fields, two.observation_fields)
        self.assertEqual(one.observation_space.shape, two.observation_space.shape)
        forbidden = {"energy_margin", "required_return_energy", "goal_delta", "station_delta", "recovery_mode"}
        self.assertTrue(forbidden.isdisjoint(two.observation_fields))


class CertificateMultiplierBindingTests(unittest.TestCase):
    def test_one_x_and_two_x_runtime_compatibility_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            one_path = Path(directory) / "one.json"
            one = make_random_persistent_uav_env(flight_energy_multiplier=1.0)
            write_certificate_artifact(one_path, one)
            load_and_verify_certificate_artifact(one_path, one)
            one.close()
            del one
            gc.collect()

            two = make_random_persistent_uav_env(flight_energy_multiplier=2.0)
            with self.assertRaisesRegex(RuntimeError, "CERTIFICATE_RUNTIME_DEPENDENCY_MISMATCH"):
                load_and_verify_certificate_artifact(one_path, two)
            descriptor = load_and_verify_certificate_artifact(CERTIFICATE, two)
            self.assertEqual(descriptor["flight_energy_multiplier"], 2.0)
            self.assertEqual(descriptor["energy_version"], "synthetic-mission-energy-v2-x2")
            two.close()


class ReplayAndWarmStartTests(unittest.TestCase):
    def setUp(self):
        self.environment = PersistentEnergyNavigationEnv(
            max_episode_steps=10,
            energy_config=EnergyNavigationConfig(flight_energy_multiplier=2.0),
        )
        self.model = SAC(
            "MlpPolicy",
            self.environment,
            buffer_size=100,
            learning_starts=5,
            batch_size=4,
            seed=4,
            device="cpu",
        )

    def tearDown(self):
        self.environment.close()

    def test_standard_sb3_replay_prefill_fields(self):
        observation, _ = self.environment.reset(seed=4)
        action = np.array([0.2, -0.1, 0.3], dtype=np.float32)
        next_observation, reward, terminated, truncated, info = self.environment.step(action)
        transition = TeacherTransition(observation, action, reward, next_observation, terminated, truncated, info, "frozen_kappa")
        self.assertEqual(prefill_sb3_replay_buffer(self.model, [transition]), 1)
        self.assertEqual(self.model.replay_buffer.size(), 1)
        np.testing.assert_allclose(self.model.replay_buffer.observations[0, 0], observation)
        np.testing.assert_allclose(self.model.replay_buffer.next_observations[0, 0], next_observation)
        np.testing.assert_allclose(self.model.replay_buffer.actions[0, 0], action)
        self.assertAlmostEqual(float(self.model.replay_buffer.rewards[0, 0]), reward, places=5)

    def test_supervised_warm_start_changes_only_actor(self):
        observations = np.stack([self.environment.observation_space.sample() for _ in range(12)]).astype(np.float32)
        actions = np.tile(np.array([0.8, -0.6, 0.4], dtype=np.float32), (12, 1))
        actor_before = {key: value.detach().clone() for key, value in self.model.actor.state_dict().items()}
        critic_before = {key: value.detach().clone() for key, value in self.model.critic.state_dict().items()}
        target_before = {key: value.detach().clone() for key, value in self.model.critic_target.state_dict().items()}
        entropy_before = self.model.log_ent_coef.detach().clone()
        losses = supervised_actor_warm_start(
            self.model,
            observations,
            actions,
            epochs=2,
            batch_size=4,
            learning_rate=1e-3,
            seed=5,
        )
        self.assertEqual(len(losses), 2)
        self.assertTrue(any(not torch.equal(value, self.model.actor.state_dict()[key]) for key, value in actor_before.items()))
        self.assertTrue(all(torch.equal(value, self.model.critic.state_dict()[key]) for key, value in critic_before.items()))
        self.assertTrue(all(torch.equal(value, self.model.critic_target.state_dict()[key]) for key, value in target_before.items()))
        self.assertTrue(torch.equal(entropy_before, self.model.log_ent_coef.detach()))


@unittest.skipUnless(NAVIGATION_TEACHER.is_file(), "solved navigation teacher artifact is unavailable")
class CompleteRecoveryTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oracle = CertifiedRecoveryOracle("random_persistent_open.json", 2.0, CERTIFICATE)
        cls.navigation_teacher = SAC.load(NAVIGATION_TEACHER, device="cpu")

    @classmethod
    def tearDownClass(cls):
        cls.oracle.environment.close()

    def test_complete_cycle_preserves_and_finishes_original_goal(self):
        environment = PersistentEnergyNavigationEnv(
            max_episode_steps=5000,
            energy_config=EnergyNavigationConfig(flight_energy_multiplier=2.0),
        )
        teacher = CertifiedRecoveryRolloutTeacher(self.navigation_teacher, self.oracle)
        result = teacher.generate_cycle(environment, seed=70_000, max_steps=5000)
        environment.close()
        self.assertTrue(result.success, result.failure_reason)
        self.assertGreater(result.recovery_transitions, 0)
        self.assertGreater(result.charging_steps, 0)
        self.assertEqual(
            result.charging_steps,
            sum(transition.info["charging"] for transition in result.transitions),
        )
        self.assertEqual(result.resume_successes, 1)
        self.assertEqual(result.original_goal_completions, 1)
        original_goal_id = result.transitions[0].info["goal_id_before"]
        self.assertTrue(all(item.info["goal_id_before"] == original_goal_id for item in result.transitions))
        self.assertTrue(result.transitions[-1].info["task_completed_now"])
        first_station = next(index for index, item in enumerate(result.transitions) if item.info["charging"])
        self.assertLess(first_station, len(result.transitions) - 1)
        self.assertFalse(result.transitions[first_station].info["task_completed_now"])
        self.assertTrue(all(np.all(np.abs(action) <= 1.0 + 1e-7) for action in result.kappa_actions))

    def test_real_system_recovery_action_uses_certified_frozen_kappa(self):
        environment = PersistentEnergyNavigationEnv(
            max_episode_steps=20,
            energy_config=EnergyNavigationConfig(flight_energy_multiplier=2.0),
        )
        start = self.oracle.atlas.sample_initial_state(91, 30.0)
        environment.reset(seed=91, options={
            "start_position": start.position,
            "start_velocity": start.velocity,
            "goal_position": self.oracle.atlas.sample_goal(np.random.default_rng(92), start.position, 0.6),
            "initial_energy_fraction": start.energy / 30.0,
        })
        self.oracle.begin_recovery()
        normalized, context = self.oracle.recovery_action(environment)
        self.assertTrue(context.recovery.certified)
        np.testing.assert_allclose(
            environment.normalized_to_physical_action(normalized),
            np.asarray(context.recovery.action),
            atol=1e-8,
        )
        environment.close()


class PolicyOnlyEvaluationTests(unittest.TestCase):
    def test_policy_only_has_no_kappa_or_hold_override(self):
        environment = PersistentEnergyNavigationEnv(max_episode_steps=2)
        model = SAC("MlpPolicy", environment, buffer_size=20, learning_starts=2, batch_size=2, device="cpu")
        args = SimpleNamespace(
            distance_potential_scale=0.25,
            gamma=0.99,
            velocity_reward_weight=0.1,
            time_cost=0.01,
            completion_reward=10.0,
            collision_penalty=1.2,
            energy_cost_weight=0.01,
            backup_intervention_cost=0.1,
            battery_capacity=30.0,
            charging_rate=2.0,
            charging_radius=0.18,
            charging_velocity_limit=[0.05, 0.05, 0.04],
            initial_energy_fraction_min=0.30,
            initial_energy_fraction_max=1.0,
            flight_energy_multiplier=2.0,
            scenario="random_persistent_open.json",
            max_episode_steps=2,
            evaluation_steps=2,
            goal_radius=0.20,
            minimum_goal_separation=0.60,
            sampling_margin=0.20,
            heldout_seeds=[100],
            evaluation_seed_base=1_000,
            seed=0,
        )
        result = evaluate_energy_model(
            model,
            args,
            0,
            0,
            evaluation_mode="deterministic",
            soc_group="full",
            execution_mode="POLICY_ONLY",
        )
        row = result["seed_results"][0]
        self.assertEqual(row["kappa_intervention_count"], 0)
        self.assertEqual(row["charger_hold_steps"], 0)
        environment.close()


if __name__ == "__main__":
    unittest.main()
