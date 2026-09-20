"""Focused semantic checks with a deterministic test plant, not research evidence."""
import pickle
import tempfile
import unittest

import numpy as np

from persistent_uav.baselines import Action, Scheduler
from persistent_uav.config import Config, METHODS, THRESHOLDS, VALIDATION_SEEDS
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.estimates import EstimateModel
from persistent_uav.navigation import StepResult
from persistent_uav.storage import restore, snapshot
from persistent_uav.streams import Task, stream_hash, workload


class TestNavigator:
    """Constant-speed, exactly metered test fixture, never usable via research CLI."""
    goal_radius = 1e-8
    dt = .05

    def __init__(self, capacity=10., speed=10., power=1.):
        self.capacity = capacity
        self.energy = capacity
        self.speed, self.power = speed, power
        self.time = 0.
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.station = np.zeros(3)
        self.goal = np.zeros(3)
        self.contacts = self.policy_steps = 0
        self.reset_count = 1

    def start_leg(self, goal):
        self.goal = np.array(goal, dtype=float)

    def reached(self):
        return bool(np.linalg.norm(self.goal - self.position) <= self.goal_radius)

    def stop_at_service(self):
        self.velocity[:] = 0.

    def advance_flight(self, max_duration):
        delta = self.goal - self.position
        distance = np.linalg.norm(delta)
        duration = min(.2, max_duration, distance / self.speed, self.energy / self.power)
        used = duration * self.power
        self.position += delta / max(distance, 1e-12) * self.speed * duration
        self.time = round(self.time + duration, 10)
        self.energy = max(0., self.energy - used)
        self.policy_steps += 1
        return StepResult(duration, used, reached=self.reached(), depleted=self.energy <= 1e-9)

    def advance_stationary(self, duration, *, recharge_rate=None):
        if recharge_rate is None:
            power = self.power if np.linalg.norm(self.position - self.station) > self.goal_radius else 0.
            if power:
                duration = min(duration, self.energy / power)
            used = duration * power
            self.energy = max(0., self.energy - used)
        else:
            used = 0.
            self.energy = min(self.capacity, self.energy + duration * recharge_rate)
        self.time = round(self.time + duration, 10)
        return StepResult(duration, used, depleted=self.energy <= 1e-9)

    def absorb_until(self, cutoff):
        self.time = cutoff


def task(i, x, arrival=0.):
    return Task(i, arrival, arrival, (float(x), 0., 0.))


class EnvironmentTests(unittest.TestCase):
    def env(self, tasks, cutoff=5., capacity=10., recharge=2., queue_capacity=5):
        config = Config(capacity, recharge, .2, cutoff, queue_capacity=queue_capacity,
                        initial_tasks=min(queue_capacity, sum(t.arrival == 0 for t in tasks)))
        return PersistentUAVThroughput(config, TestNavigator(capacity), tasks)

    def test_arrivals_during_service_queue_capacity_and_no_redraw(self):
        env = self.env([task(0, 10), task(1, 20, .05), task(2, 30, .1), task(3, 40, .15)], queue_capacity=2)
        env.step(Action('serve', 0))
        self.assertEqual([t.id for t in env.queue], [1, 2])
        self.assertEqual(env.overflow, 1)
        while env.mode == 'SERVING':
            env.step()
        self.assertEqual(env.completed, 1)
        self.assertEqual(env.cursor, 4)
        self.assertEqual(env.nav.reset_count, 1)
        self.assertAlmostEqual(sum(env.queue_time), env.time)

    def test_no_wait_action_and_no_preemption(self):
        env = self.env([task(0, 10)])
        with self.assertRaises(ValueError):
            env.step(Action('wait'))
        env.step(Action('serve', 0))
        with self.assertRaises(ValueError):
            env.step(Action('recharge'))

    def test_empty_queue_forced_wait_until_arrival(self):
        env = self.env([task(0, 10, 2.)])
        self.assertFalse(env.decision_required)
        with self.assertRaises(ValueError):
            env.step(Action('recharge'))
        env.step()
        self.assertAlmostEqual(env.time, 2.)
        self.assertAlmostEqual(env.nav.energy, 10.)
        self.assertTrue(env.decision_required)

    def test_full_station_recharge_rejected(self):
        env = self.env([task(0, 10)])
        with self.assertRaises(ValueError):
            env.step(Action('recharge'))

    def test_charging_consumes_time_and_arrivals_continue(self):
        env = self.env([task(0, 10), task(1, 20, .5)], cutoff=5.)
        env.nav.energy = 6.
        env.step(Action('recharge'))
        self.assertEqual(env.mode, 'CHARGING')
        self.assertEqual(env.nav.energy, 6.)
        env.step()
        self.assertEqual(env.time, 2.)
        self.assertEqual(env.nav.energy, 10.)
        self.assertEqual(len(env.queue), 2)
        self.assertEqual(env.completed_recharges, 1)

    def test_cutoff_midcharge_does_not_refill(self):
        env = self.env([task(0, 10)], cutoff=1.)
        env.nav.energy = 2.
        env.step(Action('recharge'))
        env.step()
        self.assertTrue(env.done)
        self.assertEqual(env.nav.energy, 4.)
        self.assertTrue(env.summary()['recharge_censored'])
        self.assertFalse(env.summary()['recharge_failure'])

    def test_goal_exactly_at_cutoff_counts_without_return_requirement(self):
        env = self.env([task(0, 10)], cutoff=1.)
        env.step(Action('serve', 0))
        while not env.done:
            env.step()
        self.assertEqual(env.completed, 1)
        self.assertEqual(env.time, 1.)
        self.assertFalse(env.summary()['depletion'])

    def test_depletion_precedes_goal_and_absorbs_to_T(self):
        env = self.env([task(0, 10), task(1, 20, 2.)], capacity=1.)
        env.step(Action('serve', 0))
        while not env.done:
            env.step()
        self.assertEqual(env.completed, 0)
        self.assertTrue(env.summary()['depletion'])
        self.assertAlmostEqual(env.failure_time, 1.)
        self.assertEqual(env.time, 5.)
        self.assertEqual(env.nav.reset_count, 1)
        self.assertAlmostEqual(sum(env.time_by_mode.values()), 5.)

    def test_away_idle_can_deplete_and_absorbs_to_cutoff(self):
        env = self.env([], capacity=1.)
        env.nav.position[0] = 5.
        env.step(Action('idle'))
        self.assertTrue(env.done)
        self.assertEqual(env.failure, 'energy_depletion')
        self.assertFalse(env.recharge_failure)
        self.assertEqual(env.failure_time, 1.)
        self.assertEqual(env.time, 5.)

    def test_partial_dock_idle_preserves_energy_without_charging(self):
        env = self.env([task(0, 10, 2.)])
        env.nav.energy = 7.
        env.step(Action('idle'))
        self.assertEqual(env.nav.energy, 7.)
        self.assertEqual(env.time, 2.)
        self.assertEqual([t.id for t in env.queue], [0])
        self.assertEqual(env.completed_recharges, 0)
        self.assertEqual(env.time_by_mode['waiting'], 2.)

    def test_full_dock_idle_to_cutoff_never_depletes(self):
        env = self.env([], capacity=1.)
        env.step()
        self.assertTrue(env.done)
        self.assertIsNone(env.failure)
        self.assertEqual(env.nav.energy, 1.)
        self.assertEqual(env.time_by_mode['waiting'], 5.)

    def test_away_idle_until_arrival_retains_hover_cost(self):
        env = self.env([task(0, 10, 2.)])
        env.nav.position[0] = 5.
        env.step(Action('idle'))
        self.assertEqual(env.nav.energy, 8.)
        self.assertEqual(env.time, 2.)
        self.assertIsNone(env.failure)

    def test_empty_queue_can_recharge_away_from_station(self):
        env = self.env([])
        env.nav.position[0] = 5.
        env.nav.energy = 2.
        self.assertTrue(env.decision_required)
        env.step(Action('recharge'))
        while env.mode in ('RETURNING', 'CHARGING'):
            env.step()
        self.assertEqual(env.completed_recharges, 1)
        self.assertEqual(env.nav.energy, 10.)

    def test_empty_queue_can_idle_but_no_idle_with_backlog(self):
        env = self.env([task(0, 10, 1.)])
        env.nav.energy = 5.
        env.step(Action('idle'))
        self.assertEqual(env.time, 1.)
        with self.assertRaises(ValueError):
            env.step(Action('idle'))

    def test_empty_station_partial_battery_can_charge(self):
        env = self.env([])
        env.nav.energy = 8.
        self.assertTrue(env.can_recharge)
        env.step(Action('recharge'))
        env.step()
        self.assertEqual(env.nav.energy, 10.)
        self.assertFalse(env.decision_required)

    def test_full_station_empty_queue_forces_idle(self):
        env = self.env([task(0, 10, 1.)])
        self.assertFalse(env.decision_required)
        env.step()
        self.assertEqual(env.time, 1.)
        self.assertTrue(env.decision_required)

    def test_midflight_disk_resume_matches_uninterrupted(self):
        env = self.env([task(0, 10), task(1, 20, .5)])
        env.step(Action('serve', 0))
        with tempfile.TemporaryDirectory() as directory:
            snapshot(directory, env)
            other = restore(directory)
            for instance in (env, other):
                while instance.mode == 'SERVING':
                    instance.step()
            self.assertEqual(env.summary(), other.summary())
            self.assertEqual(env.events, other.events)

    def test_policy_view_has_no_map_future_or_random_state(self):
        env = self.env([task(0, 10), task(1, 20, 4.)])
        o = env.observe()
        self.assertEqual([t['id'] for t in o['queue']], [0])
        self.assertFalse({'layout', 'map', 'seed', 'stream', 'cursor', 'rng'} & set(o))


class BaselineAndStreamTests(unittest.TestCase):
    def setUp(self):
        self.model = EstimateModel((0., 1., 10.), (0., 2., 1.))

    def view(self):
        return dict(position=[0., 0., 0.], battery=100., capacity=100.,
                    charger_position=[0., 0., 0.], decision_required=True, can_recharge=False,
                    queue=[dict(id=0, arrival=0., position=[20., 0., 0.]),
                           dict(id=1, arrival=1., position=[0., 0., 10.])])

    def test_rankings_are_not_all_the_same(self):
        expected = dict(fifo=0, nearest=1, shortest_time=0, energy_greedy=1, threshold_sjf=0, reserve_sjf=0)
        for method in METHODS:
            action = Scheduler(method, self.model).choose(self.view())
            self.assertEqual(action.task_id, expected[method], method)

    def test_reserve_uses_public_estimate_and_recharge_when_low(self):
        o = self.view()
        o.update(battery=1., can_recharge=True)
        a = Scheduler('reserve_sjf', self.model).choose(o)
        self.assertEqual(a.kind, 'recharge')
        o['can_recharge'] = False
        a = Scheduler('reserve_sjf', self.model).choose(o)
        self.assertEqual(a.kind, 'serve')
        self.assertIn('fallback', a.reason)

    def test_exogenous_stream_reproducible_independent_of_scheduler(self):
        c = Config(10., 1., 2., 5.)
        left, right = workload(123, c, []), workload(123, c, [])
        self.assertEqual(stream_hash(left), stream_hash(right))
        self.assertEqual(left[:3], right[:3])
        self.assertTrue(all(t.arrival >= t.original_arrival for t in left))
        self.assertTrue(all(100 <= t.position[0] <= 3900 for t in left))

    def test_frozen_regimes_depend_only_on_physical_statistics(self):
        from persistent_uav.calibration import freeze_results
        rows = [dict(job_id=i, start=[0., 0., 0.], goal=[10., 0., 0.], success=True,
                     timeout=False, duration=10., energy_used=2., collision_count=0) for i in range(1000)]
        _, frozen = freeze_results(dict(provenance={}, layout=[]), rows)
        self.assertEqual(len(frozen['regimes']), 27)
        for row in frozen['regimes']:
            c = row['config']
            self.assertEqual(c['capacity'], row['battery_tilde'] * 2.)
            self.assertAlmostEqual(c['capacity'] / c['recharge_rate'], row['charge_tilde'] * 10.)
            self.assertAlmostEqual(c['arrival_rate'] * 10., row['eta'])
            self.assertEqual(c['cutoff'], 1000.)

    def test_validation_never_selects_infeasible_threshold(self):
        from persistent_uav.evaluation import choose_thresholds
        rows = [dict(regime=0, method='threshold_sjf', threshold=t, seed=s,
                     completed=100, depletion=True, recharge_failure=False, navigation_failure=False,
                     arrivals=1, overflow=0, terminal_residual_battery=0.)
                for t in THRESHOLDS for s in VALIDATION_SEEDS]
        self.assertIsNone(choose_thresholds(rows, [0])['0'])


class CalibrationCompatibilityTests(unittest.TestCase):
    def test_compatibility_requires_exact_calibration_and_current_source(self):
        import copy
        from pathlib import Path
        from persistent_uav.evaluation import calibration_compatibility
        from persistent_uav.provenance import provenance
        from persistent_uav.storage import sha, write_json
        current = provenance()
        historical = copy.deepcopy(current)
        historical['sources']['historical_source'] = 'old'
        frozen = dict(provenance=historical)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frozen.json'
            record_path = Path(directory) / 'compatibility.json'
            write_json(path, frozen)
            record = dict(calibration_sha256=sha(path), calibration_provenance=historical,
                          diagnostic_provenance=current)
            write_json(record_path, record)
            self.assertEqual(calibration_compatibility(path, frozen, record_path), record)
            with self.assertRaises(RuntimeError):
                calibration_compatibility(path, frozen, None)
            record['diagnostic_provenance']['sources']['unexpected_change'] = 'new'
            write_json(record_path, record)
            with self.assertRaises(RuntimeError):
                calibration_compatibility(path, frozen, record_path)
            write_json(path, dict(provenance=historical, changed_regimes=True))
            with self.assertRaises(ValueError):
                calibration_compatibility(path, frozen, record_path)


if __name__ == '__main__':
    unittest.main()
