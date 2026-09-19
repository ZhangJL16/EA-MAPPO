"""Post-run descriptive diagnosis and portable census evidence, no new solves."""
import argparse
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from fpl.evaluation import digest

ROOT=Path(__file__).resolve().parents[1]


def report(data,audit,output):
    data,audit,out=map(Path,(data,audit,output))
    if out.exists():raise FileExistsError(out)
    summary=json.loads((audit/'summary.json').read_text())
    witnesses=json.loads((audit/'adaptive_witnesses.json').read_text())
    seal=json.loads((data/'seal.json').read_text())
    if digest(seal)!=summary['seal_hash']:raise ValueError('wrong audit')
    margins=[]
    for file in data.glob('*.census.json'):
        packed=json.loads(file.read_text());s=packed['payload']
        if packed['hash']!=digest(s) or sha256(file.read_bytes()).hexdigest()!=summary['census_hashes'][file.name]:
            raise ValueError('census hash changed')
        by={o['name']:o['channels'] for o in s['problem']['operations']}
        ids={w['state'] for w in witnesses if w['instance']==s['problem']['instance_id']}
        for r in s['records']:
            if r['id'] not in ids:continue
            label=r['label']
            best_no_measure=max(F(c['q']) for c in label['candidates']
                                if not c['operations'] or not any(by[n] for n in c['operations']))
            gap=F(label['value'])-best_no_measure
            if gap<=0:raise AssertionError('required sensing lacks strict gap')
            margins.append(dict(instance=s['problem']['instance_id'],state=r['id'],
                                value=label['value'],best_nonmeasuring_first_Q=str(best_no_measure),
                                strict_first_decision_gap=str(gap),
                                execute_forever_value=str(r['state']['remaining']*max(map(F,r['state']['posterior'])))))
    total=next(r for r in summary['by_family_fold'] if r['family']==r['fold']=='ALL')
    lines=['# T5.1d — depth-two adaptive-state census','', '## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
           '- Origin Date: 2026-09-19', '- Verification Status: bounded closure and label arithmetic checked',
           '- Version Label: teacher_v2_census_v1','',
           '## Decision','',
           '**A collector gap is demonstrated; broad adaptive coverage is not.** '
           'Do not kill the V2 family on an absence claim. Prioritize a versioned '
           'measurement-closure collection design (V2.1), not an immediate V3 redesign. '
           'No next-stage collection or training is executed here.','',
           'The old collector found 1 of the 6 depth-two adaptive states; it missed 5. '
           'They span 3 structures / 2 families, but only 1 training structure. '
           'This is qualified C1 evidence, not the hypothesized discovery of dozens '
           'of adaptive states, nor proof that task economics is unproblematic.','',
           '## Exact census','',
           '| Family | Fold | Groups | States | Non-prior | Required sensing | Adaptive | Adaptive groups | Missed adaptive |',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary['by_family_fold']:
        lines.append('| '+' | '.join(str(r[k]) for k in ('family','fold','groups','states','nonprior','required','adaptive','adaptive_groups','missed_adaptive'))+' |')
    lines += ['', '32/32 closures complete; 1,171/1,171 exact labels; zero unresolved. '
              'All states were selected before labeling. No roots, objective, fold, limits or mask changed.',
              '', 'Depth 1: 238 states, 6 adaptive. Depth 2: 909 states, zero required sensing. '
              'Depth 0: 32 roots, 13 required sensing. Depth memberships overlap: '
              'their sum is not the unique-state total.',
              '', 'Old collector/census intersection: 164 states. Census adds 1,007. '
              'The census is not a superset of all 591 collector states, because '
              'the latter also includes task-interleaved and longer histories.',
              '', '## Adaptive witnesses','',
              'All six occur after one measuring batch at initial H=12. Four '
              'records are two capacity-paired copies each at roots 300001 and '
              '300129: specialist feedback is followed by required coarse sensing. '
              'The other two occur at root 300130/B5: coarse or specialist feedback '
              'is followed by another specialist. Six records are not six independent '
              'graphs or six independent mechanisms.',
              '', 'Every witness has a strictly positive Q gap against the best '
              'nonmeasuring **first** action, whose Q itself allows optimal later '
              'continuation. Thus this comparison is stronger than merely '
              'beating an execute-forever heuristic. Exact fractions and full '
              'positive-probability histories are in adaptive_margins.json and '
              'adaptive_witnesses.json.',
              '', '## Boundaries','',
              'These are set counts, not occupancy probabilities or expected policy '
              'performance. The census exhausts at most two measuring-only batches '
              'including joint routes; it does not exhaust arbitrary task-interleaved '
              'or longer histories, and says nothing about unrun roots. It therefore '
              'cannot prove global family impossibility or uniquely identify a '
              'causal economics failure.',
              '', 'Closure reconstruction uses a separate normalized-belief BFS, '
              'sharing the public model/likelihood implementation. Label audits '
              'check candidate completeness, reachability and rational Q arithmetic; '
              'they are not an external independent proof of optimality.',
              '', '## Work','',
              f"Closure CPU: {summary['work']['closure_cpu_seconds']:.3f} s; "
              f"label CPU: {summary['work']['label_cpu_seconds']:.3f} s. "
              f"Label model calls: {summary['work']['label_model_calls']:,}. "
              'Single local run; no hardware-independent latency claim. '
              '59 focused tests passed, including DFS/BFS equality, joint batches, '
              'cross-depth deduplication, caps and resume. No neural/DEV/test/CONFIRM run.']
    out.mkdir(parents=True)
    for name in ('summary.json','adaptive_witnesses.json','states.json'):
        (out/name).write_bytes((audit/name).read_bytes())
    (out/'adaptive_margins.json').write_text(json.dumps(margins,sort_keys=True,indent=2)+'\n')
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    entries={}
    for folder,base in [('data',data),('audit',out)]:
        for p in sorted(base.iterdir()):
            if p.is_file() and p.suffix in ('.json','.md'):entries[f'{folder}/{p.name}']=p.read_bytes()
    for package in ('fpl','fpl_v2'):
        for p in (ROOT/'src'/package).rglob('*.py'):entries['runtime/'+str(p.relative_to(ROOT/'src'))]=p.read_bytes()
    for name in ('scripts/census_teacher_v2.py','scripts/report_teacher_census.py','tests/test_teacher_census.py',
                 'configs/teacher_v2_census_v1.json','CONTRACT_T51D_CENSUS.md'):
        entries['source/'+name]=(ROOT/name).read_bytes()
    hashes={n:sha256(b).hexdigest() for n,b in entries.items()}
    entries['FILES_SHA256.json']=json.dumps(hashes,sort_keys=True,indent=2).encode()
    archive=out/'teacher_v2_census_evidence.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n,b in sorted(entries.items()):
            info=zipfile.ZipInfo(n,(2026,9,19,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,b)
    receipt=dict(archive=archive.name,sha256=sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size,
                 source_archive_sha256=seal['plan']['source_archive_sha256'],seal_hash=digest(seal),files=hashes)
    (out/'manifest.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ('archive','sha256','bytes')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('data','audit','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();report(a.data,a.audit,a.output)
