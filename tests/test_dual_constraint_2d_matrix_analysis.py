"""Frozen-map inference must pair on maps and choose MPC on validation only."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from dual_constraint_2d.matrix_analysis import analyze
from dual_constraint_2d.train_ppo import TEST_MAP_IDS, TRAIN_SEEDS, VALIDATION_MAP_IDS


class MatrixAnalysisTests(unittest.TestCase):
    def test_validation_comparator_and_map_pairing(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split, maps in (("validation", VALIDATION_MAP_IDS),
                                ("test", TEST_MAP_IDS)):
                for map_id in maps:
                    methods = ("route_full", "route_partial", "mpc_h1", "mpc_h2",
                               *(f"ppo_seed_{seed}" for seed in TRAIN_SEEDS))
                    for method in methods:
                        score = 1
                        if method == "mpc_h1":
                            score = 3 if split == "validation" else 2
                        if method == "mpc_h2":
                            score = 2 if split == "validation" else 10
                        if method.startswith("ppo_"):
                            score = 4
                        row = {
                            "map_id": map_id, "done": True,
                            "completed_targets": score, "collision_count": 0,
                            "safety_cost": 0, "charge_events": 1,
                            "return_takeovers": 0, "energy_filter_events": 0,
                            "collision_intervention_steps": 0,
                            "policy_decisions": 5, "plant_decisions": 5,
                            "simulated_seconds": 600.0, "actual_depletion": False,
                            "failure_reason": None,
                        }
                        path = root / split / method / f"map_{map_id:03d}" / "summary.json"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(row))
            result = analyze(root)
            self.assertEqual(result["validation_selected_comparator"], "mpc_h1")
            self.assertEqual(result["ppo_minus_selected_mpc_completed"]["mean"], 2.0)
            self.assertTrue(result["c1_empirically_supported_under_prespecified_rule"])
            self.assertTrue((root / "RESULT.md").exists())


if __name__ == "__main__":
    unittest.main()
