"""Cohort selection must be fixed before looking at new-map results."""

import unittest

from delivery_1km.confirmation_qualify import MAP_IDS, STRATA, build_manifest, cohort_hash


class ConfirmationQualificationTests(unittest.TestCase):
    def test_unseen_map_manifest_has_two_fixed_routes_per_map(self) -> None:
        manifest = build_manifest()
        self.assertEqual(len(manifest["jobs"]), 32)
        self.assertEqual({job["map_id"] for job in manifest["jobs"]}, set(MAP_IDS))
        self.assertEqual({job["stratum"] for job in manifest["jobs"]}, set(STRATA))
        self.assertTrue(all(job["repeat"] == 0 for job in manifest["jobs"]))
        self.assertEqual(manifest["cohort_source_sha256"], cohort_hash())
        self.assertEqual(manifest["checkpoint_jobs"], 2)


if __name__ == "__main__":
    unittest.main()
