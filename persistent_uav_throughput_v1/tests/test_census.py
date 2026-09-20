"""Bounded branch mechanics and reconstruction tests with an engineering plant."""
import importlib.util
from pathlib import Path
import pickle
import tempfile
import unittest

from test_core import TestNavigator, task
from test_diagnostics import TimeoutNavigator
from persistent_uav.config import Config
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.storage import snapshot, restore

SPEC = importlib.util.spec_from_file_location('oracle_census', Path(__file__).resolve().parents[1] / 'scripts/oracle_recoverability_census.py')
census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(census)


class CensusTests(unittest.TestCase):
    def env(self, capacity=10., cutoff=10., navigator=None, goal=10., limit=4000):
        return PersistentUAVThroughput(Config(capacity, 2., .2, cutoff, initial_tasks=1, option_step_limit=limit),
                                       navigator or TestNavigator(capacity), [task(0, goal)])

    def root(self, env):
        return dict(env=env, rng=census.rng_state())

    def finish(self, branch):
        for _ in range(200):
            result = census.advance_branch(branch)
            if result is not None:
                return result
        self.fail('Engineering branch failed to terminate')

    def test_task_then_return_success_stops_before_charging(self):
        env = self.env()
        result = self.finish(census.new_branch(self.root(env), dict(kind='task_return', task_id=0)))
        self.assertTrue(result['safe'])
        self.assertTrue(result['task_completed'])
        self.assertTrue(result['reached_charger'])
        self.assertAlmostEqual(result['consumed_energy'], 2.)
        self.assertAlmostEqual(result['remaining_battery'], 8.)
        self.assertFalse(any(e['event'] == 'charge_complete' for e in result['events']))

    def test_task_and_return_depletion_are_distinct(self):
        for capacity, phase in ((.5, 'task'), (1.5, 'return')):
            result = self.finish(census.new_branch(self.root(self.env(capacity)), dict(kind='task_return', task_id=0)))
            self.assertFalse(result['safe'])
            self.assertEqual(result['failure_phase'], phase)
            self.assertTrue(result['depletion'])
            self.assertIsNone(result['full_mission_energy'])
            self.assertTrue(result['energy_censored'])

    def test_navigation_timeout_not_energy_infeasibility(self):
        result = self.finish(census.new_branch(self.root(self.env(navigator=TimeoutNavigator())), dict(kind='task_return', task_id=0)))
        self.assertFalse(result['safe'])
        self.assertTrue(result['navigation_failure'])
        self.assertFalse(result['depletion'])

    def test_horizon_censor_is_unknown_not_failure(self):
        result = self.finish(census.new_branch(self.root(self.env(cutoff=.1)), dict(kind='task_return', task_id=0)))
        self.assertIsNone(result['safe'])
        self.assertTrue(result['horizon_censored'])
        self.assertFalse(result['depletion'])

    def test_full_station_does_not_get_illegal_return_branch(self):
        env = self.env()
        specs = census.branch_specs(env.observe(), 'A')
        self.assertEqual(specs, [dict(kind='task_return', task_id=0)])
        env.nav.position[0] = 3.
        self.assertEqual(census.branch_specs(env.observe(), 'B'), [dict(kind='direct_return', task_id=None)])

    def test_clone_isolation_and_midflight_resume(self):
        env = self.env()
        root = self.root(env)
        before = pickle.dumps(root)
        branch = census.new_branch(root, dict(kind='task_return', task_id=0))
        self.assertIsNone(census.advance_branch(branch))
        with tempfile.TemporaryDirectory() as directory:
            snapshot(directory, dict(branch=branch, rng=census.rng_state()))
            resumed = restore(directory)
        left = self.finish(branch)
        census.set_rng(resumed['rng'])
        right = self.finish(resumed['branch'])
        self.assertEqual(census.normalized(left), census.normalized(right))
        self.assertEqual(pickle.dumps(root), before)

    def test_task_already_at_charger_needs_no_illegal_recharge(self):
        result = self.finish(census.new_branch(self.root(self.env(goal=0.)), dict(kind='task_return', task_id=0)))
        self.assertTrue(result['safe'])
        self.assertEqual(result['consumed_energy'], 0.)

    def test_unmatched_root_is_rejected(self):
        env = self.env()
        source = dict(events=census.normalized(env.events))
        target = dict(decision_event_index=len(env.events), observation=census.normalized(env.observe()))
        self.assertTrue(census.at_target(env, target, source))
        target['observation']['battery'] -= 1.
        with self.assertRaisesRegex(AssertionError, 'historical replay mismatch'):
            census.at_target(env, target, source)

    def test_policy_step_watchdog_is_not_silently_unsafe(self):
        branch = census.new_branch(self.root(self.env(limit=1)), dict(kind='task_return', task_id=0))
        with self.assertRaisesRegex(RuntimeError, 'policy-step bound'):
            self.finish(branch)


if __name__ == '__main__':
    unittest.main()
