"""Checkpointing must not change a paired evaluation trajectory."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dual_constraint_2d.evaluate_matrix import evaluate


class EvaluateMatrixTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_and_rejects_mixed_config(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            interrupted = root / "interrupted"
            continuous = root / "continuous"
            first = evaluate(interrupted, algorithm="route_full", map_id=0,
                             max_new_decisions=1, checkpoint_every=1)
            self.assertEqual(first["policy_decisions"], 1)
            second = evaluate(interrupted, algorithm="route_full", map_id=0,
                              max_new_decisions=1, checkpoint_every=1)
            whole = evaluate(continuous, algorithm="route_full", map_id=0,
                             max_new_decisions=2, checkpoint_every=1)
            self.assertEqual(second["policy_decisions"], 2)
            self.assertEqual(second["completed_targets"], whole["completed_targets"])
            self.assertEqual(second["final_energy"], whole["final_energy"])
            self.assertEqual((interrupted / "events.jsonl").read_bytes(),
                             (continuous / "events.jsonl").read_bytes())
            with self.assertRaises(RuntimeError):
                evaluate(interrupted, algorithm="route_partial", map_id=0,
                         max_new_decisions=1)


if __name__ == "__main__":
    unittest.main()
