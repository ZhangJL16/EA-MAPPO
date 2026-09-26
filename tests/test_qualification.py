import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nav3d.qualify import build_manifest, run, source_sha256


class QualificationTests(unittest.TestCase):
    def test_fixed_manifest_has_both_geometry_strata_and_valid_source_hash(self):
        manifest = build_manifest(map_count=2, per_stratum=2)
        self.assertEqual(len(manifest["jobs"]), 8)
        self.assertEqual(manifest["source_sha256"], source_sha256())
        for map_id in range(2):
            labels = [job["stratum"] for job in manifest["jobs"] if job["map_id"] == map_id]
            self.assertEqual(labels, ["blocked", "clear", "blocked", "clear"])
        self.assertEqual(manifest, build_manifest(map_count=2, per_stratum=2))

    def test_checkpoint_resume_skips_saved_jobs_and_rejects_source_mismatch(self):
        manifest = build_manifest(map_count=1, per_stratum=2)
        calls = []

        def fake_job(_output, _manifest, job):
            calls.append(job["job_id"])
            return {"job_id": job["job_id"], "outcome": "no_route", "map_id": job["map_id"]}

        with tempfile.TemporaryDirectory() as directory, patch("nav3d.qualify.execute_job", side_effect=fake_job):
            output = Path(directory) / "qualification"
            first = run(output, manifest_override=manifest, stop_after_checkpoint=True)
            self.assertEqual((first["completed"], first["total"]), (2, 4))
            self.assertEqual(len(list((output / "results").glob("*.json"))), 2)
            finished = run(output, resume=True)
            self.assertEqual((finished["completed"], finished["total"]), (4, 4))
            self.assertEqual(calls, [job["job_id"] for job in manifest["jobs"]])
            self.assertEqual(len(list((output / "results").glob("*.json"))), 4)
            saved = json.loads((output / "manifest.json").read_text())
            saved["source_sha256"] = "invalid"
            (output / "manifest.json").write_text(json.dumps(saved))
            with self.assertRaisesRegex(RuntimeError, "source hash"):
                run(output, resume=True)


if __name__ == "__main__":
    unittest.main()
