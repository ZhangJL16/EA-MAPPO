"""The Gym adapter exposes identical action and feature shapes in both modes."""

import unittest

from dual_constraint_2d.gym_adapter import DualConstraintGym


class GymAdapterTest(unittest.TestCase):
    def test_training_and_evaluation_modes_share_interface(self) -> None:
        for shielded in (False, True):
            env = DualConstraintGym((0,), 85.39121788182787, 29.9, shielded=shielded, horizon_s=5)
            observation, info = env.reset(seed=17)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertEqual(info["map_id"], 0)
            observation, reward, terminated, truncated, info = env.step(4)
            self.assertEqual(observation.shape, env.observation_space.shape)
            self.assertGreater(info["time_s"], 0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertIsInstance(reward, float)


if __name__ == "__main__":
    unittest.main()
