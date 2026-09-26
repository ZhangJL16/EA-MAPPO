"""The staged matrix must preserve paired routes and input hashes."""

import unittest

from delivery_1km.return_matrix import NAVIGATION, build_manifest, source_hash


class ReturnMatrixTests(unittest.TestCase):
    def test_two_map_stage_keeps_both_route_types_and_all_inputs(self) -> None:
        manifest = build_manifest(NAVIGATION, (112, 113))
        self.assertEqual(len(manifest["jobs"]), 4)
        self.assertEqual({job["map_id"] for job in manifest["jobs"]}, {112, 113})
        self.assertEqual({job["stratum"] for job in manifest["jobs"]},
                         {"station_to_ground", "station_to_roof"})
        self.assertTrue(all(job["navigation_outcome"] == "arrived" for job in manifest["jobs"]))
        self.assertEqual(manifest["source_sha256"], source_hash())
        self.assertEqual(manifest["disk_budget_bytes"], 4 * 1024**3)

    def test_rejects_duplicate_or_out_of_cohort_maps(self) -> None:
        with self.assertRaises(ValueError):
            build_manifest(NAVIGATION, (112, 112))
        with self.assertRaises(ValueError):
            build_manifest(NAVIGATION, (100,))


if __name__ == "__main__":
    unittest.main()
