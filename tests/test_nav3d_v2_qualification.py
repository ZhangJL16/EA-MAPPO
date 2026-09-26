import json
from pathlib import Path
import tempfile
import unittest

from nav3d.qualify import config, decode_world
from nav3d_v2.qualify import build_manifest, run, source_hash


class V2QualificationTests(unittest.TestCase):
    def test_heldout_manifest_is_deterministic_and_strata_match_geometry(self):
        manifest = build_manifest("holdout", map_count=2, per_stratum=2)
        self.assertEqual(manifest, build_manifest("holdout", map_count=2, per_stratum=2))
        self.assertEqual(len(manifest["jobs"]), 8)
        for job in manifest["jobs"]:
            world = decode_world(manifest["worlds"][job["map_id"]])
            self.assertEqual(world.segment_clear(job["start"], job["goal"], config().inflation), job["stratum"] == "clear")

    def test_checkpoint_resume_skips_completed_jobs(self):
        original = build_manifest("holdout", map_count=1, per_stratum=2)
        calls = []

        def fake_execute(_output, _manifest, job):
            calls.append(job["job_id"])
            return {"job_id": job["job_id"], "outcome": "no_route"}

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial"
            self.assertEqual(run(output, suite="holdout", manifest_override=original,
                                 stop_after_checkpoint=True, execute=fake_execute)["completed"], 2)
            self.assertEqual(run(output, suite="holdout", resume=True, execute=fake_execute)["completed"], 4)
            self.assertEqual(calls, [job["job_id"] for job in original["jobs"]])
            self.assertEqual(len(list((output / "results").glob("*.json"))), 4)
            saved = json.loads((output / "manifest.json").read_text())
            saved["source_sha256"] = "wrong"
            (output / "manifest.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, "suite/source"):
                run(output, suite="holdout", resume=True, execute=fake_execute)
            self.assertEqual(original["source_sha256"], source_hash())


if __name__ == "__main__":
    unittest.main()
