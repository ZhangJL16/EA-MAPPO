"""Focused regression for the fixed-map reset bug and unchanged valid-start flight."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np

from persistent_uav import navigation

ROOT = Path(__file__).resolve().parents[1]


class DockIdleRegression(unittest.TestCase):
    def setUp(self):
        self.nav = navigation.FrozenNavigator(100.)
        self.addCleanup(self.nav.close)

    def test_full_and_partial_dock_idle_preserve_energy_and_telemetry(self):
        nav = self.nav
        for energy in (100., 70.):
            with self.subTest(energy=energy):
                nav.base.agent.energy = energy
                before_time = nav.time
                before_cost = nav.base.cumulative_virtual_energy
                before_position = nav.position
                outcome = nav.advance_stationary(10.)
                self.assertEqual(outcome.energy_used, 0.)
                self.assertFalse(outcome.depleted)
                self.assertEqual(nav.energy, energy)
                self.assertAlmostEqual(nav.time - before_time, 10.)
                self.assertEqual(nav.base.cumulative_virtual_energy, before_cost)
                np.testing.assert_array_equal(nav.position, before_position)
                self.assertEqual(nav.reset_count, 1)

    def test_dock_tolerance_boundary_and_away_hover_metering(self):
        nav = self.nav
        nav.base.agent.pos = nav.station.copy() + [nav.goal_radius, 0., 0.]
        self.assertEqual(nav.advance_stationary(1.).energy_used, 0.)
        nav.base.agent.pos = nav.station.copy() + [nav.goal_radius + 1., 0., 0.]
        before = nav.energy
        cost = nav.base._realized_energy_cost(np.zeros(3), nav.dt, velocity=np.zeros(3))
        outcome = nav.advance_stationary(1.)
        self.assertGreater(outcome.energy_used, 0.)
        self.assertAlmostEqual(outcome.energy_used, cost / nav.dt)
        self.assertAlmostEqual(nav.energy, before - outcome.energy_used)

    def test_only_explicit_recharge_increases_docked_energy(self):
        nav = self.nav
        nav.base.agent.energy = 70.
        nav.advance_stationary(10.)
        self.assertEqual(nav.energy, 70.)
        nav.advance_stationary(10., recharge_rate=2.)
        self.assertEqual(nav.energy, 90.)
        nav.advance_stationary(10., recharge_rate=2.)
        self.assertEqual(nav.energy, 100.)

    def test_stationary_accepts_roundoff_on_both_sides_of_grid(self):
        for duration in (146.70000000011078, 146.69999999988924):
            with self.subTest(duration=duration):
                nav = self.nav
                nav.base.simulation_time = 2618.1999999998893
                nav.base.agent.energy = 70.
                outcome = nav.advance_stationary(duration)
                self.assertAlmostEqual(outcome.duration, 146.7)
                self.assertAlmostEqual(nav.time, 2764.9)
                self.assertEqual(nav.energy, 70.)

    def test_off_grid_and_nonfinite_wait_rejected_without_mutation(self):
        nav = self.nav
        for duration in (.025, 146.70001, -.05, float('nan'), float('inf')):
            with self.subTest(duration=duration):
                before = (nav.time, nav.energy)
                with self.assertRaisesRegex(ValueError, 'physics grid'):
                    nav.advance_stationary(duration)
                self.assertEqual((nav.time, nav.energy), before)

    def test_rounded_interval_still_meters_hover_and_explicit_charge(self):
        nav = self.nav
        nav.base.agent.energy = 70.
        nav.advance_stationary(1.00000000011, recharge_rate=2.)
        self.assertEqual(nav.energy, 72.)
        nav.base.agent.pos = nav.station.copy() + [nav.goal_radius + 1., 0., 0.]
        cost = nav.base._realized_energy_cost(np.zeros(3), nav.dt, velocity=np.zeros(3))
        outcome = nav.advance_stationary(1.00000000011)
        self.assertEqual(outcome.duration, 1.)
        self.assertAlmostEqual(outcome.energy_used, cost / nav.dt)


class NavigationRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        bare = Path(cls.temp.name) / 'old.git'
        subprocess.run(['git', 'clone', '--bare', '--quiet',
                        str(ROOT / 'evidence/original_v1_1/source.bundle'), str(bare)], check=True)
        source = subprocess.check_output(['git', '--git-dir', str(bare), 'show',
                                          '1a13bf0:persistent_uav/navigation.py'], text=True)
        cls.old = types.ModuleType('persistent_uav._old_navigation_regression')
        cls.old.__file__ = str(ROOT / 'persistent_uav/navigation.py')
        sys.modules[cls.old.__name__] = cls.old
        exec(compile(source, cls.old.__file__, 'exec'), cls.old.__dict__)
        cls.manifest = json.loads((ROOT / 'evidence/original_v1_1/manifest.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()
        del sys.modules[cls.old.__name__]

    def test_protected_generation_halo_does_not_reject_legal_fixed_map_start(self):
        layout = self.manifest['layout']
        # First pair whose legal sampled start lies within the 20m generation
        # halo. All input pairs were committed before calibration outcomes.
        candidates = [p for p in self.manifest['pairs'] if any(
            np.linalg.norm(np.asarray(p['start'])[:2] - o['position']) < o['radius'] + 20.5
            for o in layout)]
        pair = candidates[0]
        self.assertEqual(pair['job_id'], 94)
        with self.assertRaisesRegex(ValueError, 'protected reset position'):
            self.old.FrozenNavigator(1e9, layout=layout, start=pair['start'])
        nav = navigation.FrozenNavigator(1e9, layout=layout, start=pair['start'])
        nav.start_leg(pair['goal'])
        np.testing.assert_allclose(nav.position, pair['start'], atol=1e-4)
        np.testing.assert_array_equal(nav.base._lidar_packet.origin, nav.position)
        self.assertTrue(np.isfinite(nav.observation()).all())
        self.assertEqual(nav.layout, layout)
        nav.close()

    def test_valid_start_observation_and_physical_step_match_original_source(self):
        pair = self.manifest['pairs'][0]
        new = navigation.FrozenNavigator(1e9, layout=self.manifest['layout'], start=pair['start'])
        old = self.old.FrozenNavigator(1e9, layout=self.manifest['layout'], start=pair['start'])
        for nav in (new, old):
            nav.start_leg(pair['goal'])
        np.testing.assert_array_equal(new.observation(), old.observation())
        self.old._ACTOR = navigation.actor_for(new)
        new_result, old_result = new.advance_flight(.2), old.advance_flight(.2)
        self.assertEqual(new_result.__dict__, old_result.__dict__)
        np.testing.assert_array_equal(new.observation(), old.observation())
        self.assertEqual(new.energy, old.energy)
        for nav in (new, old):
            nav.close()
