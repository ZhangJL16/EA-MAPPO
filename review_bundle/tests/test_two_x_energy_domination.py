from __future__ import annotations

import unittest

import numpy as np

from envs.certified_uav import make_random_persistent_uav_env


class TwoXEnergyDominationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.environment = make_random_persistent_uav_env(
            "random_persistent_open.json",
            seed=0,
            timing_mode="functional",
            flight_energy_multiplier=2.0,
        )

    def test_analytic_box_upper_attains_the_separable_plant_maximum(self):
        model = self.environment.plant.energy_model
        velocity = np.array((0.3, 0.2, 0.1), dtype=np.float64)
        action = np.array((0.18, 0.10, 0.08), dtype=np.float64)
        upper = model.analytic_box_upper(velocity, action, self.environment.runtime.config.dt)

        class State:
            pass

        state = State()
        state.velocity = velocity
        realized = model.realized_cost(state, action, self.environment.runtime.config.dt)
        self.assertGreaterEqual(upper.total_cost_upper, realized)
        self.assertLess(upper.total_cost_upper - realized, 1e-14)

    def test_every_two_x_cell_dominates_full_velocity_and_tracking_action_box(self):
        environment = self.environment
        tracking = np.asarray(environment.runtime.config.tracking_error_bound, dtype=np.float64)
        minimum_slack = float("inf")
        worst_cell = None
        checked = 0
        for cell in environment.atlas.manifest.cells:
            if cell.level == 0:
                self.assertEqual(cell.one_step_energy_upper, 0.0)
                self.assertEqual(cell.energy_upper, 0.0)
                continue
            velocity = np.maximum(
                np.abs(np.asarray(cell.state_bounds.velocity.low, dtype=np.float64)),
                np.abs(np.asarray(cell.state_bounds.velocity.high, dtype=np.float64)),
            )
            command = np.maximum(
                np.abs(np.asarray(cell.action_low, dtype=np.float64)),
                np.abs(np.asarray(cell.action_high, dtype=np.float64)),
            )
            report = environment.plant.energy_model.audit_certified_upper(
                cell.one_step_energy_upper,
                velocity,
                command + tracking,
                environment.runtime.config.dt,
            )
            checked += 1
            if report.slack < minimum_slack:
                minimum_slack = report.slack
                worst_cell = cell.cell_id
            self.assertTrue(
                report.verified,
                msg=(
                    f"2x one-step energy underbound in {cell.cell_id}: "
                    f"certificate={report.certified_upper:.17g}, "
                    f"realized_box={report.realized_box_upper:.17g}, "
                    f"slack={report.slack:.17g}"
                ),
            )
        self.assertGreater(checked, 0)
        self.assertIsNotNone(worst_cell)
        self.assertGreaterEqual(minimum_slack, 0.0)

    def test_every_two_x_cell_satisfies_outward_cumulative_recursion(self):
        cells = {cell.cell_id: cell for cell in self.environment.atlas.manifest.cells}
        checked = 0
        for cell in cells.values():
            if cell.level == 0:
                self.assertIsNone(cell.successor_target_cell)
                self.assertEqual(cell.successor_energy_upper, 0.0)
                continue
            successor = cells[cell.successor_target_cell]
            self.assertLess(successor.level, cell.level)
            self.assertEqual(cell.successor_energy_upper, successor.energy_upper)
            self.assertGreaterEqual(
                cell.energy_upper,
                cell.one_step_energy_upper + successor.energy_upper,
            )
            self.assertGreaterEqual(cell.e3_residual, 0.0)
            checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
