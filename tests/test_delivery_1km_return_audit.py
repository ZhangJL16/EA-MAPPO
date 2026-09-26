"""Focused checks for the moving-state return evidence boundary."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from delivery_1km.return_audit import NAVIGATION, build_manifest, run, simulate_return
from delivery_1km.scenario import make_map


class ReturnAuditTests(unittest.TestCase):
    def test_moving_inside_station_radius_requires_executed_braking(self) -> None:
        case = make_map(100)
        initial = np.asarray(case.station_m) + np.array((0.1, 0.0, 0.0))
        backup = simulate_return(case.world, initial, np.array((4.0, 0.0, 0.0)), case.station_m)
        self.assertGreater(backup["steps"], 0)
        if backup["arrived"]:
            self.assertLessEqual(np.linalg.norm(backup["velocities"][-1]), 0.5)

    def test_manifest_binds_to_frozen_navigation_inputs(self) -> None:
        manifest = build_manifest(NAVIGATION, "m100_station_to_ground_00")
        self.assertGreater(manifest["states"], 2)
        self.assertEqual(len(manifest["profiles"]), 19)
        self.assertEqual(manifest["payloads_kg"], [0.0, 0.25, 0.5])
        self.assertEqual(len(manifest["state_indices"]), manifest["states"])
        self.assertEqual(set(manifest["state_indices"]), set(range(1, manifest["states"] + 1)))
        self.assertGreater(manifest["state_indices"][0], 1)
        with self.assertRaises(ValueError):
            build_manifest(NAVIGATION, "m100_ground_to_roof_00")

    def test_checkpoint_resume_rejects_changed_source(self) -> None:
        manifest = build_manifest(NAVIGATION, "m100_station_to_ground_00")
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            with patch("delivery_1km.return_audit.source_hash", return_value="changed"):
                with self.assertRaisesRegex(RuntimeError, "source changed"):
                    run(output, navigation=NAVIGATION, resume=True)

    def test_checkpoint_saves_verifiable_executed_return_traces(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "return_audit"
            checkpoint = run(output, navigation=NAVIGATION, stop_after_checkpoint=True)
            self.assertEqual(checkpoint["completed"], 2)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            for index in manifest["state_indices"][:2]:
                result = json.loads((output / "results" / f"step_{index:05d}.json").read_text(encoding="utf-8"))
                trace_path = output / result["return_trace"]
                with np.load(trace_path, allow_pickle=False) as trace:
                    self.assertEqual(len(trace["positions"]), result["return_steps"] + 1)
                    self.assertEqual(len(trace["accelerations"]), result["return_steps"])
                    self.assertLessEqual(np.linalg.norm(trace["positions"][-1] - np.asarray(manifest["job"]["station_m"])), 0.8)
            trace_path.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "return trace changed"):
                run(output, navigation=NAVIGATION, resume=True, stop_after_checkpoint=True)


if __name__ == "__main__":
    unittest.main()
