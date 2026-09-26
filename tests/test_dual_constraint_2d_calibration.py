"""Focused calibration persistence check for a newly resumable execution path."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from dual_constraint_2d.calibration import run


class CalibrationResumeTest(unittest.TestCase):
    def test_one_row_then_resume_preserves_first_result(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            first = run(output, max_new_jobs=1)
            self.assertEqual((first["completed_jobs"], first["new_jobs"]), (1, 1))
            original = (output / "results/m000_c000.json").read_bytes()
            resumed = run(output, max_new_jobs=1)
            self.assertEqual((resumed["completed_jobs"], resumed["new_jobs"]), (2, 1))
            self.assertEqual((output / "results/m000_c000.json").read_bytes(), original)
            self.assertTrue((output / "results/m000_c001.json").exists())


if __name__ == "__main__":
    unittest.main()
