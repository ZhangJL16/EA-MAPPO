import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sac_native_qualification import run, source_hash, MODEL_SHA


class NativeSACCheckpointTests(unittest.TestCase):
    def test_checkpoint_resume_has_no_duplicate_results(self):
        jobs = [{"job_id": f"j{i}"} for i in range(4)]
        manifest = {"protocol": "test", "source_sha256": source_hash(),
                    "model_sha256": MODEL_SHA, "checkpoint_jobs": 2, "jobs": jobs}
        calls = []

        def fake_execute(_manifest, job):
            calls.append(job["job_id"])
            return {"job_id": job["job_id"], "outcome": "arrived"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            self.assertEqual(run(output, manifest_override=manifest, stop_after_checkpoint=True,
                                 execute=fake_execute)["completed"], 2)
            self.assertEqual(run(output, resume=True, execute=fake_execute)["completed"], 4)
            self.assertEqual(calls, [job["job_id"] for job in jobs])
            self.assertEqual(len(list((output / "results").glob("*.json"))), 4)
            saved = json.loads((output / "manifest.json").read_text())
            saved["source_sha256"] = "wrong"
            (output / "manifest.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, "source or model"):
                run(output, resume=True, execute=fake_execute)


if __name__ == "__main__":
    unittest.main()
