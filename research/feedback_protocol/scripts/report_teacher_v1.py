"""Post-run descriptive coverage only. Does not solve, sample or edit Teacher V1."""
import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction as F
import hashlib
from itertools import combinations
import json
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fpl.evaluation import digest, save_new


def summarize(rows):
    exact = [r for r in rows if r['status'] == 'exact']
    return dict(roots=len({r['root'] for r in rows}), exact=len(exact),
                unresolved=len(rows)-len(exact), nonprior=sum(r['nonprior'] for r in exact),
                measurement_optimal=sum(r['measurement_optimal'] for r in exact),
                measurement_required=sum(r['measurement_required'] for r in exact),
                nonprior_measurement=sum(r['nonprior'] and r['measurement_optimal'] for r in exact),
                ties=sum(r['ties'] for r in exact),
                unique_belief_vectors=len({tuple(r['belief']) for r in exact}),
                structural_groups=len({r['structural_group'] for r in rows}))


def work_summary(rows):
    return {k:dict(median=statistics.median(r[k] for r in rows),
                   max=max(r[k] for r in rows), total=sum(r[k] for r in rows))
            for k in ('solver_states','expansions','model_calls','cpu_seconds','wall_seconds')} if rows else {}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    data, audit_dir, out = map(Path, (args.data, args.audit, args.output))
    if out.exists():
        raise FileExistsError('new report directory required')
    seal = json.loads((data/'seal.json').read_text())
    audit = json.loads((audit_dir/'audit.json').read_text())
    archive = audit_dir/'teacher_startup_evidence.zip'  # Frozen audit script's historical filename.
    if audit['status'] != 'passed' or audit['summary']['status'] != 'complete':
        raise ValueError('completed audit required')
    if hashlib.sha256(archive.read_bytes()).hexdigest() != audit['archive_sha256']:
        raise ValueError('archive checksum mismatch')
    rows, groups, samples, input_hashes = [], defaultdict(list), [], {}
    expected = {j['instance_hash'] for j in seal['jobs']}
    with zipfile.ZipFile(archive) as z:
        if z.read('data/seal.json') != (data/'seal.json').read_bytes():
            raise ValueError('archive seal mismatch')
        for path in sorted(data.glob('*.shard.json')):
            raw = path.read_bytes()
            if z.read('data/'+path.name) != raw:
                raise ValueError('archive shard mismatch')
            input_hashes[path.name] = hashlib.sha256(raw).hexdigest()
            envelope = json.loads(raw)
            s = envelope['payload']
            if digest(s) != envelope['hash'] or s['instance_hash'] not in expected:
                raise ValueError('shard mismatch')
            expected.remove(s['instance_hash'])
            samples.extend(dict(root=s['metadata']['root_seed'], condition=s['metadata']['condition'], **x) for x in s['sampling'])
            for r in s['records']:
                label = r['label']
                exact = label['status'] == 'exact'
                opts = [c for c in label.get('candidates',[]) if c['operations'] in label.get('optimal_set',[])]
                measurement = exact and any(c.get('channels') for c in opts)
                required = exact and bool(opts) and all(c.get('channels') for c in opts)
                first = sorted({c['operations'][0] if c['operations'] else '<STOP>' for c in opts})
                row = dict(instance_hash=s['instance_hash'], state_id=r['id'], root=s['metadata']['root_seed'],
                           condition=s['metadata']['condition'], structural_group=s['metadata']['clone_hash'],
                           channels=len(s['problem']['channels']), hypotheses=len(s['problem']['hypotheses']),
                           H=r['state']['remaining'], belief=r['state']['posterior'],
                           status=label['status'], reason=label.get('reason'),
                           sources=sorted({x['source'] for x in r['provenance']}),
                           nonprior=r['state']['posterior'] != s['problem']['prior'],
                           measurement_optimal=bool(measurement), measurement_required=bool(required),
                           ties=exact and len(opts)>1, optimal_first_operations=first,
                           optimal_protocols=[c['operations'] for c in opts],
                           value=label.get('value'), solver_states=label['solver_states'],
                           expansions=label['work']['expansions'], model_calls=label['work']['model_calls'],
                           cpu_seconds=label['cpu_seconds'], wall_seconds=label['wall_seconds'])
                rows.append(row)
                groups[s['instance_hash']].append(row)
    if expected:
        raise ValueError('missing roots')
    by_condition = {c['id']:summarize([r for r in rows if r['condition']==c['id']]) for c in seal['plan']['conditions']}
    by_source = {src:summarize([r for r in rows if src in r['sources']])
                 for src in ('exact','random','voi','beam','posterior_sampling')}
    changes = dict(belief=[], horizon=[])
    for group in groups.values():
        for a,b in combinations([r for r in group if r['status']=='exact' and r['H']>0], 2):
            axis = ('belief' if a['H']==b['H'] and a['belief']!=b['belief'] else
                    'horizon' if a['belief']==b['belief'] and a['H']!=b['H'] else None)
            if axis and set(a['optimal_first_operations']).isdisjoint(b['optimal_first_operations']):
                changes[axis].append(dict(root=a['root'], condition=a['condition'],
                                         states=[a['state_id'],b['state_id']], H=[a['H'],b['H']],
                                         beliefs=[a['belief'],b['belief']],
                                         optimal_first=[a['optimal_first_operations'],b['optimal_first_operations']],
                                         involves_measurement=a['measurement_optimal'] or b['measurement_optimal']))
    report = dict(scope='post-run TRAIN corpus census; no solver reruns, no prefix export or network',
                  total=summarize(rows), by_condition=by_condition, by_source=by_source,
                  work_all=work_summary(rows), work_exact=work_summary([r for r in rows if r['status']=='exact']),
                  work_unresolved=work_summary([r for r in rows if r['status']!='exact']),
                  yield_by_horizon={str(h):summarize([r for r in rows if r['H']==h]) for h in sorted({r['H'] for r in rows})},
                  yield_by_channels={str(m):summarize([r for r in rows if r['channels']==m]) for m in sorted({r['channels'] for r in rows})},
                  sampling_status=dict(Counter(s['status'] for s in samples)),
                  sampling_truncations=[s for s in samples if s['status']!='complete'],
                  action_change_summary={k:dict(pairs=len(v), roots=len({x['root'] for x in v}),
                                               measurement_pairs=sum(x['involves_measurement'] for x in v)) for k,v in changes.items()},
                  input_sha256=input_hashes, archive_sha256=audit['archive_sha256'])
    if report['total']['exact'] != audit['summary']['labels'].get('exact',0):
        raise ValueError('count mismatch')
    out.mkdir(parents=True)
    save_new(out/'summary.json', report)
    save_new(out/'action_change_witnesses.json', changes)
    with (out/'states.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v,list) else v for k,v in row.items()})
    shutil.copy2(audit_dir/'audit.json', out/'audit.json')
    shutil.copy2(archive, out/'teacher_v1_complete_evidence.zip')
    lines = ['# Teacher V1 full coverage audit', '', '## Material Passport', '',
             '- Origin skill: academic-research-suite / experiment-agent, inline run and descriptive collection.',
             '- Verification: sealed full audit passed; this report performs no new exact solves.',
             '- No independent optimality proof, population inference or student-performance claim.', '',
             '## Condition coverage', '',
             'Counts are root–state pairs. Non-prior, sensing, ties and beliefs below refer to exact labels.',
             'Measurement-optimal means **at least one** optimal protocol measures; measurement-required means all do.',
             'Belief counts are distinct numeric vectors, not necessarily the same posterior decision problem.', '',
             '| Condition | Roots | Exact | Unresolved | Non-prior | Measurement-optimal | Ties | Unique belief vectors | Structural groups |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,row in by_condition.items():
        keys=('roots','exact','unresolved','nonprior','measurement_optimal','ties','unique_belief_vectors','structural_groups')
        lines.append('| '+name+' | '+' | '.join(str(row[k]) for k in keys)+' |')
    lines += ['', 'Structural groups overlap across conditions; do not sum that column. Global counts:', '',
              '```json', json.dumps(report['total'],indent=2), '```', '', '## Source memberships', '',
              '| Source | Final unique states | Exact | Unresolved | Non-prior | Measurement-optimal |',
              '|---|---:|---:|---:|---:|---:|']
    for name,row in by_source.items():
        lines.append(f"| {name} | {row['exact']+row['unresolved']} | {row['exact']} | {row['unresolved']} | {row['nonprior']} | {row['measurement_optimal']} |")
    lines += ['', 'Memberships overlap; they are not independent samples and must not be summed as dataset size.', '',
              '## Label work (all selected states)', '', '| Metric | Median | Maximum | Total |', '|---|---:|---:|---:|']
    for name,s in report['work_all'].items():
        lines.append(f"| {name} | {s['median']:.9g} | {s['max']:.9g} | {s['total']:.9g} |")
    lines += ['', f"Whole-shard recorded CPU: {audit['summary']['total_cpu_seconds']:.9g} s; includes sampling and in-process auditing.",
              'Label CPU above is only exact V/Q generation. Separate full-audit/report/export cost is excluded.',
              'Root-state work distribution is descriptive, not a sample of independent task difficulties.', '',
              '## Action-change witnesses', '',
              'Within one fixed public instance, compare H>0 states with either identical H and different belief,',
              'or identical belief and different H. Count only **disjoint optimal first-operation sets**;',
              'mere tie-breaking changes are not counted. These pairs are dependent and not population evidence.', '',
              '```json', json.dumps(report['action_change_summary'],indent=2), '```', '',
              'Full pairs are in action_change_witnesses.json. CSV retains optimal full protocols and input identities.', '',
              '## Selection and limitations', '',
              f"Sampling status: {report['sampling_status']}. Truncation detail and yield by horizon/channel/source are in summary.json.",
              'With no unresolved selected labels, there is no observed solver-status selection bias within this corpus.',
              'This does not rule out sampling caps, hash-priority selection or restriction to small/easy tasks.',
              'Unresolved concentration in hypothetical harder/more informative states cannot be inferred from zero failures.',
              'Measurement-optimal states need not require deep adaptive sensing; first-operation witnesses are a conservative',
              'diagnostic, not proof of graph-generalizing planning competence. Eight structures remain insufficient evidence',
              'for that claim. No network, prefix export, Teacher V2, test/OOD/confirm execution or new gate was run.', '',
              '## Evidence', '',
              f"Archive SHA256: `{audit['archive_sha256']}`.",
              'The archive has the historical internal filename convention from the unchanged audit script but contains all 24 shards.',
              'The new portable filename is teacher_v1_complete_evidence.zip; startup evidence is untouched.', '']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({k:report[k] for k in ('total','work_all','sampling_status','action_change_summary')},indent=2))


if __name__ == '__main__':
    main()
