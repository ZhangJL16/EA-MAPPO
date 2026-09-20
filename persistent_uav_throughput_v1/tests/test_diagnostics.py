"""Telemetry checks use engineering fixtures, not research trajectories."""
import copy
import tempfile
import unittest

from test_core import TestNavigator, task
from persistent_uav.baselines import Action, Scheduler
from persistent_uav.config import Config
from persistent_uav.diagnostics import audited_step, reserve_prediction, run_diagnostics
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.estimates import EstimateModel
from persistent_uav.navigation import StepResult
from persistent_uav.storage import restore, snapshot


class TimeoutNavigator(TestNavigator):
    def advance_flight(self, max_duration):
        result = super().advance_flight(max_duration)
        return StepResult(result.duration, result.energy_used, timeout=True)


class DiagnosticsTests(unittest.TestCase):
    model = EstimateModel((0., .1, 0.), (0., .1, 0.), goal_radius=1e-8)

    def env(self, capacity=10., tasks=None, navigator=None):
        tasks = [task(0, 10)] if tasks is None else tasks
        config = Config(capacity, 2., .2, 5., initial_tasks=len(tasks))
        return PersistentUAVThroughput(config, navigator or TestNavigator(capacity), tasks)

    def finish(self, env, action):
        audited_step(env, action, self.model)
        while not env.done and not env.decision_required:
            audited_step(env, None, self.model)
        return run_diagnostics(env.events, env.summary())

    def test_strict_margin_and_full_station_fallback(self):
        env = self.env(capacity=2.)
        action = Scheduler('reserve_sjf', self.model).choose(env.observe())
        self.assertEqual(action.reason, 'full_station_infeasible_estimate_fallback')
        audit = reserve_prediction(env.observe(), action, self.model)
        self.assertEqual(audit['selected']['predicted_task_energy'], 1.)
        self.assertEqual(audit['selected']['predicted_return_energy'], 1.)
        self.assertEqual(audit['selected']['predicted_reserve_margin'], 0.)
        self.assertFalse(audit['selected']['estimated_feasible'])

    def test_passive_telemetry_and_disk_resume_match(self):
        plain = self.env(tasks=[task(0, 10), task(1, 30)])
        logged = copy.deepcopy(plain)
        scheduler = Scheduler('reserve_sjf', self.model)
        action = scheduler.choose(plain.observe())
        plain.step(action)
        audited_step(logged, action, self.model)
        with tempfile.TemporaryDirectory() as folder:
            snapshot(folder, logged)
            resumed = restore(folder)
        while not plain.done:
            action = scheduler.choose(plain.observe()) if plain.decision_required else None
            for env in (logged, resumed):
                self.assertEqual(env.observe(), plain.observe())
            plain.step(action)
            audited_step(logged, action, self.model)
            audited_step(resumed, action, self.model)
        self.assertEqual(plain.summary(), logged.summary())
        self.assertEqual(logged.events, resumed.events)
        stripped = copy.deepcopy(logged.events)
        for event in stripped:
            event.pop('reserve_prediction', None)
            event.pop('failure_phase', None)
        self.assertEqual(plain.events, stripped)
        self.assertEqual(run_diagnostics(logged.events, logged.summary()),
                         run_diagnostics(resumed.events, resumed.summary()))

    def test_task_depletion_attribution(self):
        flags, decisions = self.finish(self.env(capacity=.5), Action('serve', 0))
        self.assertEqual(flags['depletion_phase'], 'task')
        self.assertFalse(flags['navigation_failure'])
        self.assertEqual(decisions[0]['actual_leg_energy'], .5)
        self.assertEqual(decisions[0]['leg_failure'], 'energy_depletion')

    def test_return_depletion_attribution(self):
        env = self.env()
        env.nav.position[0] = 10.
        env.nav.energy = .5
        flags, decisions = self.finish(env, Action('recharge'))
        self.assertEqual(flags['depletion_phase'], 'return')
        self.assertTrue(flags['recharge_failure'])
        self.assertEqual(decisions[0]['predicted_direct_return_energy'], 1.)
        self.assertIsNone(decisions[0]['selected'])

    def test_waiting_depletion_not_mislabeled_as_task(self):
        env = self.env(capacity=.5, tasks=[])
        env.nav.position[0] = 10.
        flags, _ = self.finish(env, Action('idle'))
        self.assertEqual(flags['depletion_phase'], 'waiting')

    def test_navigation_timeout_is_not_depletion(self):
        env = self.env(navigator=TimeoutNavigator())
        flags, decisions = self.finish(env, Action('serve', 0))
        self.assertTrue(flags['navigation_failure'])
        self.assertFalse(flags['depletion'])
        self.assertEqual(flags['failure_phase'], 'task')
        self.assertIsNone(flags['depletion_phase'])
        self.assertEqual(decisions[0]['leg_failure'], 'navigation_failure')

    def test_completed_leg_not_assigned_later_return_failure(self):
        env = self.env()
        self.finish(env, Action('serve', 0))
        env.nav.energy = .5
        flags, decisions = self.finish(env, Action('recharge'))
        self.assertEqual(decisions[0]['leg_outcome'], 'task_completed')
        self.assertIsNone(decisions[0]['leg_failure'])
        self.assertAlmostEqual(decisions[0]['actual_leg_energy'], 1.)
        self.assertEqual(decisions[0]['run_outcome']['depletion_phase'], 'return')
        self.assertEqual(decisions[1]['leg_failure_phase'], 'return')


if __name__ == '__main__':
    unittest.main()
