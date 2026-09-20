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
