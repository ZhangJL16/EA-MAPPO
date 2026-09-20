"""Copy completed pilot evidence and verify retained prefix; never starts experiments."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from persistent_uav.provenance import verify_provenance
from persistent_uav.qualification import analyze
from persistent_uav.storage import restore, write_json


def main():
    source = ROOT / 'artifacts/calibration_20260920_repaired'
    target = ROOT / 'evidence/v1_2/calibration'
    status = json.loads((source / 'status.json').read_text())
    if status['status'] != 'complete':
        raise RuntimeError('completed 1000-job pilot required; do not publish partial results as complete')
    manifest = json.loads((source / 'manifest.json').read_text())
    verify_provenance(manifest['provenance'])
    rows = json.loads((source / 'jobs.json').read_text())
    old = restore(ROOT / 'evidence/original_v1_1')
    assert rows[:len(old['rows'])] == old['rows']
    assert manifest['pairs'] == old['manifest']['pairs']
    assert manifest['layout'] == old['manifest']['layout']
    assert sorted(r['job_id'] for r in rows) == list(range(1000))
    target.mkdir(parents=True, exist_ok=True)
    for name in ('manifest.json', 'migration.json', 'status.json', 'jobs.json', 'statistics.json', 'frozen_regimes.json'):
        shutil.copy2(source / name, target / name)
    result = analyze(source, target)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir() if p.is_file()}
    write_json(target / 'integrity.json', dict(
        completed_jobs=1000, planned_jobs=1000, unique_job_ids=True,
        original_completed_prefix_preserved=len(old['rows']),
        map_unchanged=True, pair_list_unchanged=True,
        source_commit='15fe98e14a84e109964d0b5570f53d024a8e54d3',
        original_source_commit='1a13bf0', evidence_sha256=hashes,
        training_updates=0, scheduler_runs=0, software_error_count=1,
        software_error='fixed-map reset generation-halo validation; repaired without dropping a pair',
        qualification_status=result['status']))
    print(json.dumps(dict(status=result['status'], overall=result['overall'], statistics=result['statistics']), indent=2))


if __name__ == '__main__':
    main()
