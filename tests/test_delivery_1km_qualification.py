"""Qualification manifest and checkpoint continuation without costly flights."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from nav3d.geometry import CylinderObstacle, World3D
from delivery_1km.navigation import DeliveryRoutePlanner
from delivery_1km.qualify import build_manifest, run


class QualificationTests(unittest.TestCase):
    def test_default_manifest_covers_unseen_maps_and_four_route_types(self) -> None:
        manifest = build_manifest()
        self.assertEqual(len(manifest["jobs"]), 96)
        self.assertEqual(len({job["job_id"] for job in manifest["jobs"]}), 96)
        self.assertEqual({job["map_id"] for job in manifest["jobs"]}, set(range(100, 112)))
        self.assertEqual({job["stratum"] for job in manifest["jobs"]}, {
            "station_to_ground", "station_to_roof", "ground_to_roof", "roof_to_station",
        })
        self.assertTrue(all("start_m" in job and "goal_m" in job for job in manifest["jobs"]))

    def test_overflight_fallback_checks_every_leg_against_full_map(self) -> None:
        world = World3D((100, 100, 50), (
            CylinderObstacle(np.array([50.0, 50.0]), 5.0, 0.0, 30.0),
        ))
        planner = DeliveryRoutePlanner(world, body_radius=0.4)
        route = planner.plan((2.0, 50.0, 2.0), (98.0, 50.0, 2.0))
        self.assertEqual(len(route.waypoints), 4)
        self.assertTrue(all(world.segment_clear(a, b, planner.inflation)
                            for a, b in zip(route.waypoints, route.waypoints[1:])))
        self.assertGreater(route.waypoints[1][2], 30.0 + planner.inflation)

    def test_resume_preserves_completed_job_results(self) -> None:
        manifest = build_manifest(map_count=1, per_stratum=1)
        called = []

        def fake_execute(_output: Path, _manifest: dict, job: dict) -> dict:
            called.append(job["job_id"])
            return {"job_id": job["job_id"], "outcome": "mocked"}

        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "qualification"
            with patch("delivery_1km.qualify.build_manifest", return_value=manifest), patch(
                "delivery_1km.qualify.execute_job", side_effect=fake_execute,
            ):
                first = run(output, stop_after_checkpoint=True)
                self.assertEqual((first["completed"], first["total"]), (2, 4))
                saved_first = (output / "results" / f"{manifest['jobs'][0]['job_id']}.json").read_bytes()
                second = run(output, resume=True, stop_after_checkpoint=True)
            self.assertEqual((second["completed"], second["total"]), (4, 4))
            self.assertEqual(called, [job["job_id"] for job in manifest["jobs"]])
            self.assertEqual((output / "results" / f"{manifest['jobs'][0]['job_id']}.json").read_bytes(), saved_first)
            self.assertEqual(json.loads((output / "checkpoint.json").read_text())["completed"], 4)


if __name__ == "__main__":
    unittest.main()
