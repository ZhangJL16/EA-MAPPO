"""Record v1.4 local compatibility without rewriting historical calibration."""
import ast
import json
from pathlib import Path
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from persistent_uav.provenance import provenance
from persistent_uav.storage import write_json, sha

current = provenance()
prior = json.loads((ROOT / 'evidence/v1_3/calibration_compatibility.json').read_text())
old = prior['diagnostic_provenance']
assert old.keys() == current.keys()
changed = {}
for k in old.keys() - {'sources'}:
    assert old[k] == current[k], k
assert old['sources'].keys() == current['sources'].keys()
for path, digest in old['sources'].items():
    if digest != current['sources'][path]:
        rel = Path(path).relative_to(ROOT).as_posix()
        assert rel in ('persistent_uav/navigation.py', 'persistent_uav/evaluation.py'), rel
        changed[rel] = dict(before=digest, after=current['sources'][path])
old_source = subprocess.check_output(['git', '-C', str(ROOT.parent), 'show',
    'b8cb673:persistent_uav_throughput_v1/persistent_uav/navigation.py'], text=True)
def nonstationary(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'FrozenNavigator':
            node.body = [n for n in node.body if not (isinstance(n, ast.FunctionDef) and n.name == 'advance_stationary')]
    return ast.dump(tree)
assert nonstationary(old_source) == nonstationary((ROOT / 'persistent_uav/navigation.py').read_text())
frozen_path = ROOT / 'evidence/v1_2/calibration/frozen_regimes.json'
assert sha(frozen_path) == prior['calibration_sha256']
write_json(ROOT / 'evidence/v1_4/calibration_compatibility.json', dict(
    calibration_sha256=prior['calibration_sha256'], calibration_provenance=prior['calibration_provenance'],
    diagnostic_provenance=current, prior_compatibility_sha256=sha(ROOT / 'evidence/v1_3/calibration_compatibility.json'),
    changed_since_v1_3=changed, flight_code_unchanged=True, legacy_model_and_dependencies_unchanged=True,
    calibration_rerun=False, regimes_retuned=False,
    reason='Nearest-grid stationary validation and honest error status; frozen flight calibration unchanged.',
    platform_scope='Local x86_64 provenance only; ARM host must extend its audited migration record, not copy this provenance.',
    limits='No local production resume; ARM full validation in a new output, never merge local 382 rows.'))
print('v1.4 compatibility verified; historical calibration untouched')
