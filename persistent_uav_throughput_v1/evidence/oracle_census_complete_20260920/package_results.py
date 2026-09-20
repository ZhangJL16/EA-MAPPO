"""Package completed census evidence and derive state partitions; no simulation."""
from pathlib import Path
import collections
import hashlib
import json
import shutil
import sys
import tarfile

source = Path(sys.argv[1]).resolve()
out = Path(__file__).resolve().parent

def dump(name, obj):
    (out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = json.loads((source / 'aggregate/results.json').read_text())
assert len(rows) == len({r['target_id'] for r in rows}) == 205
assert sum(len(r['branches']) for r in rows) == 561
for name in ('results.json', 'summary.json', 'integrity.json'):
    shutil.copyfile(source / 'aggregate' / name, out / name)
analysis = {}
for group in 'ABC':
    rr = [r for r in rows if r['group'] == group]
    categories = collections.defaultdict(list)
    for r in rr:
        if r['any_safe_task'] is True:
            label = 'safe_alternative_task' if r['original_task_safe'] is False else 'original_task_immediate_return_safe'
        elif r['direct_return_safe'] is True:
            label = 'empty_queue_return_safe' if group == 'B' else 'direct_return_only_safe'
        elif r['any_tested_safe_continuation'] is False:
            label = 'no_tested_safe_continuation'
        else:
            label = 'unresolved'
        categories[label].append(r['target_id'])
    bad = [r for r in rr if r['original_task_safe'] is False]
    analysis[group] = {
        'states': len(rr),
        'partition': {k: {'count': len(v), 'target_ids': v} for k, v in categories.items()},
        'original_task_unsafe_states': len(bad),
        'alternative_safe_given_original_unsafe': sum(r['alternative_safe_task'] is True for r in bad),
        'safe_task_count_distribution': dict(sorted(collections.Counter(len(r['safe_task_ids']) for r in rr).items())),
        'branch_outcomes': dict(collections.Counter(b['outcome'] for r in rr for b in r['branches'])),
    }
dump('state_analysis.json', analysis)
# Include all inputs, exact root snapshots, branch traces, final and preceding
# worker checkpoints, manifests/status and logs. Exclude empty process lock files.
files = sorted(p for p in source.rglob('*') if p.is_file() and not p.name.endswith('.lock'))
inventory = [{'path': str(p.relative_to(source)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
archive = out / 'raw_census.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for p in files:
        tar.add(p, arcname=str(p.relative_to(source)), recursive=False)
with tarfile.open(archive, 'r:gz') as tar:
    members = tar.getmembers()
    assert [m.name for m in members] == [r['path'] for r in inventory]
    for member, record in zip(members, inventory):
        assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == record['sha256']
dump('raw_inventory.json', inventory)
dump('archive_verification.json', {'passed': True, 'members_verified': len(files), 'archive_bytes': archive.stat().st_size, 'archive_sha256': sha(archive), 'excluded': 'empty process lock files only', 'new_simulations': 0})
(out / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.name}\n' for p in sorted(out.iterdir()) if p.is_file() and p.name != 'SHA256SUMS'))
print(json.dumps({'groups': analysis, 'archive_bytes': archive.stat().st_size, 'members': len(files)}))
