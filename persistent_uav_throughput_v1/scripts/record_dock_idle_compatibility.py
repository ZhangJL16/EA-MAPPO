"""Audit reuse of the unchanged flight calibration after the dock-idle fix."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistent_uav.provenance import provenance
from persistent_uav.storage import sha, write_json

BASE = '15fe98e14a84e109964d0b5570f53d024a8e54d3'
frozen_path = ROOT / 'evidence/v1_2/calibration/frozen_regimes.json'
frozen = json.loads(frozen_path.read_text())
old = frozen['provenance']
current = provenance()
changed = {}
assert set(old['sources']) == set(current['sources'])
for key, digest in old['sources'].items():
    if current['sources'][key] != digest:
        relative = Path(key).relative_to(ROOT).as_posix()
        assert relative in {'persistent_uav/navigation.py', 'persistent_uav/evaluation.py',
                            'persistent_uav/cli.py', 'persistent_uav/continuing_smoke.py'}, relative
        changed[relative] = dict(before=digest, after=current['sources'][key])
for key in old.keys() - {'sources'}:
    assert old[key] == current[key], key
old_nav = subprocess.check_output(['git', '-C', str(ROOT.parent), 'show',
                                   f'{BASE}:persistent_uav_throughput_v1/persistent_uav/navigation.py'], text=True)
assert hashlib.sha256(old_nav.encode()).hexdigest() == old['sources'][str(ROOT / 'persistent_uav/navigation.py')]
expected = old_nav.replace('elif actual > 0:',
    'elif actual > 0 and np.linalg.norm(self.position - self.station) > self.goal_radius:')
new_nav = (ROOT / 'persistent_uav/navigation.py').read_text()
assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(new_nav))
# Verify frozen artifact bytes against the completed evidence commit, not just a
# new digest of potentially modified input.
old_frozen = subprocess.check_output(['git', '-C', str(ROOT.parent), 'show',
    '15aba0d:persistent_uav_throughput_v1/evidence/v1_2/calibration/frozen_regimes.json'])
assert old_frozen == frozen_path.read_bytes()
write_json(ROOT / 'evidence/v1_3/calibration_compatibility.json', dict(
    reason='User-authorized dock idle consumes zero energy; flight calibration is unchanged.',
    historical_source_commit=BASE, calibration_sha256=sha(frozen_path),
    calibration_provenance=old, diagnostic_provenance=current, changed_sources=changed,
    checks=dict(navigation_ast_only_adds_dock_guard=True, flight_and_calibration_engine_unchanged=True,
                legacy_model_and_dependencies_unchanged=True, frozen_artifact_byte_identical=True),
    calibration_rerun=False, regimes_retuned=False,
    authorization='2026-09-20 user: navigation PASS for scheduling diagnostics; B0–B5 GO after dock idle fix.',
    limits='Diagnostic only; no neural training, MPC, Oracle or final 98% kill test.'))
print(json.dumps(dict(compatibility_verified=True, changed_sources=list(changed),
                      frozen_sha256=sha(frozen_path))))
