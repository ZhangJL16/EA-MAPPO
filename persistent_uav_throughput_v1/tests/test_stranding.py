"""Offline accounting tests: no simulation, model inference, or runtime imports."""
import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/analyze_b5_stranding.py'
SPEC = importlib.util.spec_from_file_location('analyze_b5_stranding', SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class StrandingTests(unittest.TestCase):
    def test_task_error_sufficient_and_both_identities(self):
        d = analysis.decompose(10., 4., 3., 2., 4., 2., 4.)
        self.assertEqual(d['m_pre'], 3.)
        self.assertEqual(d['task_energy_error'], 4.)
        self.assertEqual(d['endpoint_return_shift'], 1.)
        self.assertEqual(d['m_post'], -2.)
        self.assertEqual(analysis.return_category(d, False), 'task_underestimation_sufficient_for_margin_flip')
        self.assertEqual(d['task_identity_error'], 0.)
        self.assertEqual(d['return_identity_error'], 0.)

    def test_endpoint_only_and_joint_are_not_task_only(self):
        endpoint = analysis.decompose(10., 4., 3., 6., 7., 6., 7.)
        self.assertEqual(analysis.return_category(endpoint, False), 'endpoint_shift_sufficient_for_margin_flip')
        joint = analysis.decompose(10., 4., 3., 4., 5., 4., 5.)
        self.assertEqual(analysis.return_category(joint, False), 'combined_task_and_endpoint_margin_flip')

    def test_waiting_loss_without_task_error(self):
        d = analysis.decompose(100., 20., 20., 80., 20., 10., 20.)
        self.assertEqual(d['task_energy_error'], 0.)
        self.assertEqual(d['endpoint_return_shift'], 0.)
        self.assertEqual(d['post_task_battery_loss'], 70.)
        self.assertEqual(analysis.return_category(d, False), 'post_task_waiting_margin_loss')

    def test_fallback_precedence_keeps_overlapping_wait_visible(self):
        d = analysis.decompose(10., 6., 6., 7., 6., 4., 6.)
        self.assertLess(d['m_pre'], 0.)
        self.assertGreater(d['m_post'], 0.)
        self.assertLess(d['m_return'], 0.)
        self.assertEqual(analysis.return_category(d, True), 'fallback_predicted_infeasible_before_task')

    def test_return_false_safe_is_departure_specific(self):
        d = analysis.decompose(20., 4., 5., 15., 5., 8., 5.)
        self.assertEqual(d['m_return'], 3.)
        self.assertEqual(analysis.return_category(d, False), 'return_prediction_false_safe')

    def test_depleted_leg_energy_is_censored_not_completed_cost(self):
        d = analysis.failed_energy(10., 0., 3.)
        self.assertEqual(d['completion_energy_lower_bound'], 10.)
        self.assertEqual(d['prediction_error_lower_bound'], 7.)
        self.assertIsNone(d['full_leg_energy'])
        self.assertTrue(d['full_leg_energy_censored'])

    def test_goal_radius_zero_estimate_boundary(self):
        model = dict(goal_radius=5., energy_coefficients=[1., 2., 3.])
        self.assertEqual(analysis.predict_energy(model, [0., 0., 0.], [3., 4., 0.]), 0.)
        self.assertAlmostEqual(analysis.predict_energy(model, [0., 0., 0.], [6., 0., 2.]), 19.)


if __name__ == '__main__':
    unittest.main()
