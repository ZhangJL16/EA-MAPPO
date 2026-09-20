"""One explicit, audited migration of the original fixed-map start-validator failure.

Does not relax normal --resume checks. The original manifest/checkpoint stays intact.
"""
import hashlib
import json
from pathlib import Path
import subprocess

from .provenance import ROOT, provenance
from .storage import restore, sha, snapshot, write_json

ORIGINAL_COMMIT = '1a13bf0'


def migrate(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() and any(target.iterdir()):
        raise FileExistsError('migration target must be empty')
    original = json.loads((source / 'manifest.json').read_text())
    error = json.loads((source / 'error.json').read_text())
    if error['error'] != "ValueError('static obstacle overlaps a protected reset position')":
        raise ValueError('this migration only covers the documented fixed-map reset validation bug')
    current = provenance()
    old_provenance = original['provenance']
    if current['checkpoint_sha256'] != old_provenance['checkpoint_sha256'] or current['packages'] != old_provenance['packages']:
        raise ValueError('model or dependency change is not permitted by this migration')
    git_dir = ROOT / 'artifacts/source_history.git'
    old_root = Path('/home/zjl/persistent_uav_throughput_v1')
    for source_path, expected in old_provenance['sources'].items():
        p = Path(source_path)
        if p.is_relative_to(old_root):
            content = subprocess.check_output(['git', '--git-dir', str(git_dir), 'show',
                                               ORIGINAL_COMMIT + ':' + str(p.relative_to(old_root))])
            actual = hashlib.sha256(content).hexdigest()
        else:
            actual = sha(p)
        if actual != expected:
            raise ValueError('original source evidence or unchanged legacy dependency mismatch: ' + source_path)
    state = restore(source)
    if state['manifest'] != original or state['kind'] != 'calibration':
        raise ValueError('snapshot and original manifest mismatch')
    retained = len(state['rows'])
    if [r['job_id'] for r in state['rows']] != list(range(retained)):
        raise ValueError('non-prefix completed jobs')
    pair_hash = hashlib.sha256(json.dumps(original['pairs'], sort_keys=True).encode()).hexdigest()
    migration = dict(
        original_source_commit=ORIGINAL_COMMIT, original_manifest_sha256=sha(source / 'manifest.json'),
        original_resume_ref=json.loads((source / 'resume.json').read_text()),
        error=error, preserved_completed_jobs=retained,
        resumed_active_job=retained if state['nav'] is not None else None,
        pair_list_sha256=pair_hash, map_unchanged=True, pairs_unchanged=True,
        explanation='Fixed map initialized at its protected station, then physically legal x is placed at reset only. No map, sample, collision or flight change.',
        software_failure_not_navigation_failure=True,
        uncheckpointed_tail='Re-executed from last valid checkpoint; completed prefix retained byte-for-byte in JSON.')
    revised = dict(original, provenance=current, migration=migration)
    state['manifest'] = revised
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'manifest.json', revised)
    write_json(target / 'migration.json', migration)
    snapshot(target, state)
    write_json(target / 'status.json', dict(status='ready_to_resume', completed_jobs=retained,
                                          planned_jobs=1000, training_updates=0, scheduler_runs=0))
    print(json.dumps(migration, indent=2))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    a = p.parse_args()
    migrate(a.source, a.target)
