"""New checkpoint path needs a tiny resume-equivalence check."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dual_constraint_2d.baseline_runner import run


class BaselineRunnerTest(unittest.TestCase):
    def test_one_plus_two_decisions_match_three_direct_decisions(self) -> None:
        with TemporaryDirectory() as temp:
            direct = Path(temp) / "direct"
            resumed = Path(temp) / "resumed"
            run(direct, horizon_s=5.0, max_new_decisions=3, checkpoint_every=2)
            first = run(resumed, horizon_s=5.0, max_new_decisions=1)
            self.assertEqual(first["decision_count"], 1)
            run(resumed, horizon_s=5.0, max_new_decisions=2)
            self.assertEqual(
                (direct / "events.jsonl").read_bytes(),
                (resumed / "events.jsonl").read_bytes(),
            )
            self.assertEqual(
                json.loads((direct / "status.json").read_text()),
                json.loads((resumed / "status.json").read_text()),
            )

    def test_manifest_rejects_changed_configuration(self) -> None:
        with TemporaryDirectory() as temp:
            output = Path(temp)
            run(output, horizon_s=5.0)
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                run(output, horizon_s=6.0)


if __name__ == "__main__":
    unittest.main()
