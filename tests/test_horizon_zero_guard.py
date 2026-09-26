"""No planner may certify a zero-length block at the horizon boundary."""

from pathlib import Path
import pickle
import unittest

from dual_constraint_2d.action_adapter import ActionAdapter
from dual_constraint_2d.environment import DualConstraintEnv
from dual_constraint_2d.mpc import FullMapMPC
from gpu2d.rollout_mpc import BatchedRolloutMPC


class HorizonZeroGuardTests(unittest.TestCase):
    def test_cpu_and_gpu_policy_return_horizon_action(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        env.mode = "flight"
        env.time_s = env.horizon_s - 0.01
        self.assertEqual(env._flight_step_limit(), 0)
        self.assertEqual(FullMapMPC(env.case, depth=2).act(env), 9)
        gpu_policy = object.__new__(BatchedRolloutMPC)
        self.assertEqual(gpu_policy.act(env), 9)
        _, _, done, _ = env.step(ActionAdapter(env.case).decode(9, env.observe()))
        self.assertTrue(done)
        self.assertEqual(env.time_s, env.horizon_s)

    def test_failing_cpu_checkpoint_finishes_without_trace_change(self) -> None:
        checkpoint = Path(
            "/home/zjl/uav_learning_research/artifacts/"
            "dual_constraint_2d_static_matrix_v1_20260926/validation/"
            "mpc_h1/map_034/checkpoint.pkl"
        )
        if not checkpoint.exists():
            self.skipTest("archived failing checkpoint unavailable")
        saved = pickle.loads(checkpoint.read_bytes())
        wrapper, policy = saved["wrapper"], saved["policy"]
        before = len(saved["events"])
        self.assertEqual(before, 380)
        action = policy.act(wrapper.env)
        wrapper.step(action)
        self.assertEqual(wrapper.env._flight_step_limit(), 0)
        action = policy.act(wrapper.env)
        self.assertEqual(action, 9)
        _, _, terminated, truncated, _ = wrapper.step(action)
        self.assertTrue(terminated or truncated)
        self.assertTrue(wrapper.env.done)
        self.assertEqual(wrapper.env.time_s, 600.0)


if __name__ == "__main__":
    unittest.main()
