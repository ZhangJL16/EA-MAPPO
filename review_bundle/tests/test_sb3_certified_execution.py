from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
from envs.certified_uav import EnergyNavigationConfig, PersistentEnergyNavigationEnv
from scripts.sb3_certified_execution import CertifiedExecutionTrainingEnv


ROOT = Path(__file__).resolve().parents[1]
CERTIFICATE = ROOT / "certificates/random_persistent_open_energy_x2_certified_execution.json"


class CertifiedExecutionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CERTIFICATE.is_file():
            raise unittest.SkipTest(f"current certified execution artifact missing: {CERTIFICATE}")
        clean = PersistentEnergyNavigationEnv(
            max_episode_steps=200,
            energy_config=EnergyNavigationConfig(flight_energy_multiplier=2.0),
        )
        cls.environment = CertifiedExecutionTrainingEnv(
            clean,
            scenario="random_persistent_open.json",
            flight_energy_multiplier=2.0,
            certificate_artifact=CERTIFICATE,
            training_seed=17,
        )

    @classmethod
    def tearDownClass(cls):
        cls.environment.close()

    def test_clean_observation_contract_is_preserved(self):
        observation, _ = self.environment.reset(seed=17)
        clean = self.environment.actor_environment
        self.assertEqual(observation.shape, clean.observation_space.shape)
        forbidden = {
            "energy_margin",
            "required_return_energy",
            "goal_delta",
            "station_delta",
            "recovery_mode",
            "teacher_mode",
        }
        self.assertTrue(forbidden.isdisjoint(clean.observation_fields))

    def test_normal_action_maps_into_certified_support_and_preserves_goal(self):
        self.environment.reset(seed=21)
        goal_before = self.environment.actor_environment.goal.copy()
        proposed = np.array([1.0, -1.0, 1.0], dtype=np.float32)
        _, _, terminated, _, info = self.environment.step(proposed)
        self.assertFalse(terminated)
        self.assertEqual(info["execution_authority"], "POLICY_CERTIFIED_RUN_SUPPORT")
        self.assertFalse(info["kappa_takeover_started"])
        self.assertFalse(np.allclose(info["proposed_normalized_action"], info["executed_normalized_action"]))
        np.testing.assert_array_equal(self.environment.actor_environment.goal, goal_before)

    def test_departure_requires_both_gate_and_certified_action_support(self):
        station = self.environment.actor_environment.scenario.station_position
        self.environment.reset(
            seed=31,
            options={
                "start_position": station,
                "start_velocity": [0.0, 0.0, 0.0],
                "goal_position": [3.0, 3.0, 1.0],
                "initial_energy_fraction": 1.0,
            },
        )
        self.assertTrue(self.environment.oracle.departure_allowed(self.environment.actor_environment))
        self.assertTrue(
            self.environment.oracle.certified_departure_action(
                self.environment.actor_environment,
                np.array([0.5, 0.0, 0.0], dtype=np.float32),
            )
        )
        self.assertFalse(
            self.environment.oracle.certified_departure_action(
                self.environment.actor_environment,
                np.ones(3, dtype=np.float32),
            )
        )

    def test_training_source_has_no_one_shot_teacher_initialization(self):
        source = (ROOT / "scripts/train_sb3_energy_sac_certified_execution.py").read_text(encoding="utf-8")
        self.assertNotIn("prefill_sb3_replay_buffer", source)
        self.assertNotIn("supervised_actor_warm_start", source)
        self.assertIn('"teacher_replay_prefill": False', source)
        self.assertIn('"actor_supervised_warm_start": False', source)

    def test_existing_formal_artifact_roots_are_distinct(self):
        formal_root = ROOT / "artifacts/phase2_sb3_sac_energy_open_1m_2x_certified_execution"
        protected = {
            ROOT / "artifacts/phase1_sb3_sac_1m_gpu",
            ROOT / "artifacts/phase2_sb3_sac_energy_open_1m",
            ROOT / "artifacts/phase2_sb3_sac_energy_open_1m_2x_unguided",
            ROOT / "artifacts/phase2_sb3_sac_energy_open_1m_2x_recovery_guided",
        }
        self.assertNotIn(formal_root, protected)


if __name__ == "__main__":
    unittest.main()
