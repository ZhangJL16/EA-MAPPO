"""Full sizing audit and portable evidence; no new optimization or labels."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
from statistics import median
from time import monotonic, process_time
from unittest.mock import patch
import zipfile
from fpl.evaluation import digest, save_new
from fpl.teacher_data import audit_label
from fpl_v2.generator import from_json
from fpl_v2.decoder import ScalableFeasibilityMask
from fpl_v2.runtime import closure

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('prefix_export', ROOT/'scripts/export_teacher_targets.py')
prefix_export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix_export)


def category(p, record):
    label = record['label']
    exact = label['status'] == 'exact'
    nonprior = tuple(map(F, record['state']['posterior'])) != p.prior
    required = optimal = tied = False
    if exact:
        channels = {o.name: o.channels for o in p.operations}
        measured = [bool(path) and any(channels[n] for n in path) for path in label['optimal_set']]
        required, optimal = all(measured), any(measured)
        tied = len(measured) > 1
    return dict(selected=1, exact=int(exact), unresolved=int(not exact),
                nonprior=int(nonprior and exact), measurement_required=int(required),
                measurement_optimal=int(optimal), nonprior_measurement_required=int(nonprior and required),
                nonprior_measurement_optimal=int(nonprior and optimal), ties=int(tied))


def mask_regression(shards, replay=False):
    result = Counter()
    for shard in shards:
        p = from_json(shard['problem'])
        mask = ScalableFeasibilityMask(p)
        for record in shard['records']:
            if replay:
                audit_label(p, record)
                result['arithmetic_replayed'] += 1
            if record['label']['status'] != 'exact':
                result['unresolved_no_exhaustive_targets'] += 1
                continue
            rows = prefix_export.derive_prefixes(shard['problem'], record)
            prefix_export.verify_prefixes(shard['problem'], record, rows)
            with patch('fpl.protocols.enumerate_protocols', side_effect=AssertionError('mask requested catalogue')):
                for row in rows:
                    actual = mask.legal_next(row['prefix'], record['state']['remaining'])
                    if actual != row['legal_next']:
                        raise AssertionError(dict(instance=p.instance_id, state=record['id'], prefix=row['prefix'],
                                                  actual=actual, expected=row['legal_next']))
                    result['prefixes'] += 1
                    result['decision_prefixes'] += int(row['loss_mask'])
            result['states'] += 1
        result['dp_states'] += mask.states
        result['edge_checks'] += mask.edge_checks
    result['mismatches'] = 0
    return dict(result)


def census(shards):
    counts, groups, variants, adaptive = defaultdict(Counter), defaultdict(set), defaultdict(set), defaultdict(set)
    source_counts, source_groups = defaultdict(Counter), defaultdict(set)
    source_status = Counter()
    work_rows, sources_work, state_rows = [], [], []
    for s in shards:
        p, meta = from_json(s['problem']), s['metadata']
        keys = [('ALL','ALL'), (meta['family'],'ALL'), ('ALL',meta['fold']), (meta['family'],meta['fold'])]
        for key in keys:
            groups[key].add(meta['clone_hash']); variants[key].add(p.instance_id)
        for log in s['sampling']:
            source_status[(log['source'],log['status'],log['reason'])] += 1
            sources_work.append(dict(family=meta['family'], root=meta['root_seed'], source=log['source'],
                                     status=log['status'], **log['work'], cpu_seconds=log['cpu_seconds']))
        for r in s['records']:
            cat = category(p,r)
            for key in keys:
                counts[key].update(cat)
                if cat['nonprior_measurement_required']:adaptive[key].add(meta['clone_hash'])
            memberships = sorted({o['source'] for o in r['provenance']})
            for source in memberships:
                for fold in ('ALL',meta['fold']):
                    source_counts[(source,fold)].update(cat)
                    if cat['nonprior_measurement_required']:source_groups[(source,fold)].add(meta['clone_hash'])
                if memberships == [source]:source_counts[(source,'ALL')]['exclusive_states'] += 1
            label=r['label']
            work_rows.append(dict(family=meta['family'], root=meta['root_seed'], state=r['id'],
                                  status=label['status'], solver_states=label['solver_states'],
                                  cpu_seconds=label['cpu_seconds'], **label['work']))
            state_rows.append(dict(family=meta['family'], fold=meta['fold'], root=meta['root_seed'],
                                   clone_hash=meta['clone_hash'], instance=p.instance_id, state=r['id'],
                                   remaining=r['state']['remaining'], posterior=r['state']['posterior'],
                                   sources=memberships, **cat))
    tables = [dict(family=k[0],fold=k[1],groups=len(groups[k]),variants=len(variants[k]),
                   **counts[k],adaptive_groups=len(adaptive[k])) for k in sorted(counts)]
    sources = [dict(source=k[0],fold=k[1],**source_counts[k],adaptive_groups=len(source_groups[k])) for k in sorted(source_counts)]
    work = {}
    for name, rows in [('labels',work_rows),('collection',sources_work)]:
        work[name] = {key:dict(total=sum(r[key] for r in rows),median=median(r[key] for r in rows),max=max(r[key] for r in rows))
                      for key in ('expansions','model_calls','cpu_seconds','seconds')}
    work['labels']['solver_states'] = dict(median=median(r['solver_states'] for r in work_rows),max=max(r['solver_states'] for r in work_rows))
    return dict(by_family_fold=tables, by_source_fold=sources,
                source_status=[dict(source=k[0],status=k[1],reason=k[2],count=v) for k,v in sorted(source_status.items(),key=str)],
                work=work), state_rows, work_rows, sources_work


def audit(admission, data, output):
    start,cpu=monotonic(),process_time()
    admission,data,out=Path(admission),Path(data),Path(output)
    if out.exists():raise FileExistsError(out)
    manifest=json.loads((admission/'admission.json').read_text())
    seal=json.loads((data/'seal.json').read_text())
    if seal['admission_hash'] != digest(manifest) or seal['source_hashes'] != closure():
        raise ValueError('source/admission seal changed')
    expected={Path(j['problem_file']).stem:j for j in manifest['jobs'] if j['pilot']}
    if len(expected)!=32:raise ValueError('wrong pilot size')
    shards=[]; files={}
    for path in sorted(data.glob('*.shard.json')):
        stem=path.name.removesuffix('.shard.json')
        if stem not in expected:raise ValueError('nonpilot label encountered')
        packed=json.loads(path.read_text()); shard=packed['payload']
        if packed['hash']!=digest(shard) or shard['metadata']!=expected[stem] or shard['seal_hash']!=digest(seal):
            raise ValueError('shard content/identity mismatch')
        selected=json.loads((data/(stem+'.states.json')).read_text())
        if digest(selected)!=shard['selection_hash'] or selected['seal_hash']!=digest(seal):
            raise ValueError('prelabel selection seal mismatch')
        if selected['records'] != [{k:v for k,v in r.items() if k!='label'} for r in shard['records']]:
            raise ValueError('state selection changed after labels')
        if from_json(shard['problem']).instance_hash!=expected[stem]['instance_hash']:
            raise ValueError('public problem changed')
        shards.append(shard);files[path.name]=sha256(path.read_bytes()).hexdigest()
    if len(shards)!=32:raise ValueError('pilot incomplete; use startup check instead of full audit')
    pilot_mask=mask_regression(shards,replay=True)
    v1=[]
    v1path=ROOT/'provenance/teacher_v1_complete/teacher_v1_complete_evidence.zip'
    with zipfile.ZipFile(v1path) as z:
        for name in z.namelist():
            if name.endswith('.shard.json'):
                packed=json.loads(z.read(name))
                if packed['hash']!=digest(packed['payload']):raise ValueError('V1 shard hash mismatch')
                v1.append(packed['payload'])
    v1_mask=mask_regression(v1)
    summary,rows,work,collection=census(shards)
    summary.update(schema='teacher-v2-sizing-audit-v1', completed_variants=len(shards),
                   admitted_groups=len(manifest['groups']), admitted_variants=len(manifest['jobs']),
                   unlabelled_nonpilot_variants=96, mask_regression=dict(v1=v1_mask,v2_pilot=pilot_mask),
                   admission_hash=digest(manifest),seal_hash=digest(seal),shard_hashes=files,
                   new_optimality_solves=0, new_environment_calls=0,
                   optimality_scope='Arithmetic/reachability and V/Q replay; not an independent optimality proof',
                   auto_full_run=False,training_started=False,private_confirm_read=False,
                   audit_cpu_seconds=process_time()-cpu,audit_wall_seconds=monotonic()-start)
    out.mkdir(parents=True)
    for name,payload in [('summary.json',summary),('states.json',rows),('label_work.json',work),('collection_work.json',collection)]:
        save_new(out/name,payload)
    allrow=next(x for x in summary['by_family_fold'] if x['family']==x['fold']=='ALL')
    lines=['# Teacher V2 sizing pilot','', '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
           '- Origin Date: 2026-09-19', '- Verification Status: arithmetic/reachability and mask regression checked',
           '- Version Label: teacher_v2_sizing_v1','',
           '32/32 pilot variants completed; 32 identities / 128 variants admitted before labels. '
           'The remaining 24 structures / 96 variants were not labeled. No neural training or DEV/CONFIRM access.','',
           '## Coverage (exact-state categories; unresolved kept)','',
           '| Family | Fold | Groups | Variants | Selected | Exact | Unresolved | Non-prior | Required | Non-prior + required | Adaptive groups |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['by_family_fold']:
        fields=['family','fold','groups','variants','selected','exact','unresolved','nonprior','measurement_required','nonprior_measurement_required','adaptive_groups']
        lines.append('| '+' | '.join(str(r.get(k,0)) for k in fields)+' |')
    lines += ['', '## Source membership (overlaps, not independent samples)','',
              '| Source | Fold | Selected | Exact | Non-prior | Non-prior + required | Adaptive groups |',
              '|---|---|---:|---:|---:|---:|---:|']
    for r in summary['by_source_fold']:
        lines.append('| '+' | '.join(str(r.get(k,0)) for k in ['source','fold','selected','exact','nonprior','nonprior_measurement_required','adaptive_groups'])+' |')
    lines += ['', '## Mask and audit', '',
              f"V1: {v1_mask['prefixes']} prefixes; V2 pilot: {pilot_mask['prefixes']} prefixes; **0 mismatches**. "
              'Catalogue calls are patched to fail during mask checks. Only audit/reference paths enumerate.',
              '', 'Joint DP avoids complete path materialization, not worst-case combinatorial complexity. '
              'Work/state caps are explicit unresolved conditions, not false infeasibility.',
              '', 'The exact label audit replays public histories, likelihoods, candidate completeness and Q arithmetic. '
              'It does not independently prove successor optimality.', '', '## Execution limits and interpretation','',
              f"Exact: {allrow['exact']}/{allrow['selected']}; unresolved: {allrow['unresolved']}; "
              f"adaptive structures: {allrow['adaptive_groups']}/8. No cap increases, root replacements or retries.",
              '', 'Full-corpus targets 6 total / 4 train / 2 validation are not mechanically applied to this sizing pilot. '
              'Full run requires a separate decision; no automatic continuation. See source_status and work in summary.json '
              'for collection truncation, solve cost and selected-state coverage. Source truncation and label failure are distinct.',
              '', 'Targets remain Bayes values. Hierarchical Bernoulli-family coverage is not general experimental-planning evidence.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();result=audit(a.admission,a.data,a.output)
    print(json.dumps({k:result[k] for k in ('completed_variants','mask_regression','by_family_fold','source_status','work')},indent=2))
