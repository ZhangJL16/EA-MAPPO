import json
from pathlib import Path
import unittest

import numpy as np

from nav3d.qualify import config, decode_world
from nav3d_v2 import SegmentTrackingController, fly_segment_route


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "artifacts/nav3d_qualification_20260923/manifest.json").read_text())
JOBS = {job["job_id"]: job for job in MANIFEST["jobs"]}


class SegmentTrackingTests(unittest.TestCase):
    def test_nominal_acceleration_stays_on_planned_ascent_or_descent(self):
        job = JOBS["m00_clear_01"]
        world = decode_world(MANIFEST["worlds"][job["map_id"]])
        controller = SegmentTrackingController(world, config())
        initial = np.asarray(job["start"])
        target = np.asarray(job["goal"])
        acceleration = controller.nominal_acceleration(initial, np.zeros(3), target)
        direction = (target - initial) / np.linalg.norm(target - initial)
        self.assertLess(np.linalg.norm(np.cross(acceleration, direction)), 1e-10)
        self.assertLess(acceleration[2], 0.0)

    def test_original_clear_overflight_stays_above_wall_and_arrives(self):
        job = JOBS["m00_clear_01"]
        world = decode_world(MANIFEST["worlds"][job["map_id"]])
        flight = fly_segment_route(world, job["start"], job["goal"], config=config())
        self.assertTrue(flight.arrived)
        self.assertEqual(flight.final_state.collision_count, 0)
        self.assertEqual(flight.controller_infeasible_steps, 0)
        near_wall = flight.positions[(flight.positions[:, 0] >= 26.4) & (flight.positions[:, 0] <= 33.6)]
        self.assertGreater(near_wall.shape[0], 0)
        self.assertGreater(float(np.min(near_wall[:, 2])), float(world.obstacles[0].high[2] + config().inflation))


if __name__ == "__main__":
    unittest.main()
