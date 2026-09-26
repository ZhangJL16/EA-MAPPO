"""GPU proposal search cannot bypass the exact safety filter."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from dual_constraint_2d.environment import DualConstraintEnv
from gpu2d.evaluate import evaluate
from gpu2d.rollout_mpc import BatchedRolloutMPC


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
class GPURolloutMPCTests(unittest.TestCase):
    def test_gpu_search_uses_exact_shortlist(self) -> None:
        env = DualConstraintEnv(0, 85.39121788182787, 29.9)
        policy = BatchedRolloutMPC(env.case, horizon_blocks=8, exact_shortlist=3)
        env.mode = "flight"
        action = policy.act(env)
        self.assertIn(action, range(10))
        self.assertEqual(policy.stats.gpu_candidates, 324)
        self.assertLessEqual(policy.stats.exact_candidates, 3)

    def test_evaluation_checkpoint_resumes(self) -> None:
        with TemporaryDirectory() as directory:
            output = Path(directory)
            first = evaluate(output, map_id=0, horizon_blocks=4,
                             max_new_decisions=1, checkpoint_every=1)
            second = evaluate(output, map_id=0, horizon_blocks=4,
                              max_new_decisions=1, checkpoint_every=1)
            self.assertEqual(first["policy_decisions"], 1)
            self.assertEqual(second["policy_decisions"], 2)
            self.assertTrue((output / "checkpoint.pkl").exists())
            self.assertEqual(second["collision_count"], 0)


if __name__ == "__main__":
    unittest.main()
