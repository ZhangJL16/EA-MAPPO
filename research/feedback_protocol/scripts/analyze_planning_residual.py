"""Exact descriptive attribution. No significance threshold or training."""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
import zipfile
from fpl.evaluation import digest, source_hashes

ROOT=Path(__file__).resolve().parents[1]


def dimensions(r,meta):
    return dict(prior='prior' if r['prior_state'] else 'nonprior',decision_class=r['decision_class'],
                H=str(r['H']),B=str(r['B']),family=meta['family'],structure=meta['clone_hash'],fold=meta['fold'],
                optimal_types='+'.join(r['optimal_types']),chosen_type=r['chosen_type'],
                limit='limit_hit' if r.get('work',{}).get('limit_reason') else 'no_limit',
                autopilot='autopilot' if r.get('work',{}).get('autopilot') else 'normal')


def analyze(data,out):
    data,out=Path(data),Path(out)
    if out.exists():raise FileExistsError(out)
    seal=json.loads((data/'seal.json').read_text())
    if seal['source_hashes']!=source_hashes() or seal['script_sha256']!=sha256((ROOT/'scripts/planning_residual.py').read_bytes()).hexdigest():
        raise ValueError('runtime changed')
    files=sorted(data.glob('*.residual.json'))
    if len(files)!=32:raise ValueError('study incomplete')
    allrows=[];episodes=[];comparisons=[];episode_comparisons=[];supplemental=[]
    probes=defaultdict(Counter);probe_groups=defaultdict(set)
    occ=defaultdict(lambda:defaultdict(F));occgroups=defaultdict(set);occvariants=defaultdict(set)
    attribution=defaultdict(lambda:defaultdict(F));uniform=defaultdict(lambda:defaultdict(F))
    concentration=defaultdict(lambda:defaultdict(F));paired=defaultdict(lambda:defaultdict(F));wallcpu=0
    for file in files:
        packed=json.loads(file.read_text());s=packed['payload'];meta=s['metadata'];instance=s['problem']['instance_id']
        if packed['hash']!=digest(s) or s['seal_hash']!=digest(seal):raise ValueError('receipt mismatch')
        if meta not in seal['schedule']:raise ValueError('unexpected instance')
        index={};wallcpu+=s['cpu_seconds']
        for r in s['state_residuals']:
            row=dict(r,instance=instance,**meta);allrows.append(row)
            key=(r['method'],r['tier']);probes[key]['rows']+=1
            if r['status']!='exact':probes[key]['unresolved']+=1;continue
            delta=F(r['residual']);index[(r['state_id'],*key)]=r
            probes[key]['nonzero']+=int(delta>0)
            probes[key]['residual_sum']+=delta
            if delta>0:probe_groups[key].add(meta['clone_hash'])
            for dim,value in dimensions(r,meta).items():
                uniform[(*key,dim,value)]['states']+=1
                uniform[(*key,dim,value)]['nonzero']+=int(delta>0)
                uniform[(*key,dim,value)]['residual_sum']+=delta
        for method in seal['plan']['methods']:
            for sid in {k[0] for k in index if k[1]==method}:
                small=index.get((sid,method,'small'));large=index.get((sid,method,'large'))
                if small is None or large is None:continue
                delta=F(small['residual'])-F(large['residual'])
                comparisons.append(dict(instance=instance,**meta,state_id=sid,method=method,
                    small_residual=small['residual'],large_residual=large['residual'],delta_compute=str(delta),
                    small_action=small['action'],large_action=large['action'],
                    categories=dimensions(small,meta)))
        emap={}
        for e in s['occupancy']:
            episodes.append(dict(e,instance=instance,**meta));key=(e['method'],e['tier']);emap[key]=e
            if e['status']!='exact':occ[key]['unresolved_episodes']+=1;continue
            gap=F(e['gap']);rows=e['rows']
            if sum(F(r['weighted_residual']) for r in rows)!=gap or F(e['value_star'])-F(e['value_policy'])!=gap:
                raise AssertionError('receipt identity failed')
            occ[key]['episodes']+=1;occ[key]['positive_gap_episodes']+=int(gap>0);occ[key]['gap_sum']+=gap
            occ[key]['expected_model_calls_sum']+=F(e['expected_model_calls']);occ[key]['expected_expansions_sum']+=F(e['expected_expansions'])
            if gap>0:occgroups[key].add(meta['clone_hash']);occvariants[key].add(instance)
            for r in rows:
                mass=F(r['occupancy']);delta=F(r['residual']);weighted=mass*delta
                if weighted!=F(r['weighted_residual']):raise AssertionError('wrong contribution')
                occ[key]['context_rows']+=1;occ[key]['positive_contexts']+=int(delta>0)
                occ[key]['expected_visits_sum']+=mass;occ[key]['expected_bad_visits_sum']+=mass*int(delta>0)
                concentration[key][(instance,r['state_id'])]+=weighted
                for dim,value in dimensions(r,meta).items():
                    attribution[(*key,dim,value)]['gap_contribution']+=weighted
                    attribution[(*key,dim,value)]['expected_visits']+=mass
                    attribution[(*key,dim,value)]['positive_contexts']+=int(delta>0)
        for method in seal['plan']['methods']:
            small,large=emap[(method,'small')],emap[(method,'large')]
            if small['status']==large['status']=='exact':
                improvement=F(small['gap'])-F(large['gap'])
                episode_comparisons.append(dict(instance=instance,**meta,method=method,
                    small_gap=small['gap'],large_gap=large['gap'],gain_from_larger_budget=str(improvement)))
        supplemental.extend(dict(r,instance=instance) for r in s['supplemental_labels'])
    occtable=[]
    for key,values in sorted(occ.items()):
        gap=values['gap_sum'];ranked=sorted((v for v in concentration[key].values() if v>0),reverse=True)
        occtable.append(dict(method=key[0],tier=key[1],**{k:str(v) for k,v in values.items()},
            positive_gap_groups=len(occgroups[key]),positive_sufficient_states=len(ranked),
            top_state_share=str(ranked[0]/gap) if gap else None,
            mean_gap=str(gap/values['episodes']) if values['episodes'] else None,
            concentration_curve=[dict(rank=i,cumulative_gap=str(sum(ranked[:i])),share=str(sum(ranked[:i])/gap)) for i in range(1,len(ranked)+1)]))
    atable=[]
    for (method,tier,dim,value),v in sorted(attribution.items()):
        total=occ[(method,tier)]['gap_sum']
        atable.append(dict(method=method,tier=tier,dimension=dim,category=value,
            **{k:str(x) for k,x in v.items()},gap_share=str(v['gap_contribution']/total) if total else None))
    # Each exclusive dimension must separately account for the WHOLE gap.
    for key,values in occ.items():
        for dim in {a['dimension'] for a in atable}:
            if sum(F(a['gap_contribution']) for a in atable if (a['method'],a['tier'],a['dimension'])==(*key,dim))!=values['gap_sum']:
                raise AssertionError('category partition failed')
    summary=dict(schema='planning-residual-summary-v1',seal_hash=digest(seal),
        census_rows=len(allrows),episodes=len(episodes),supplemental_states=len(supplemental),
        supplemental_unresolved=sum(r['label']['status']!='exact' for r in supplemental),
        fresh_probe_summary=[dict(method=k[0],tier=k[1],**{n:str(v) for n,v in c.items()},nonzero_groups=len(probe_groups[k])) for k,c in sorted(probes.items())],
        occupancy_summary=occtable,occupancy_attribution=atable,
        uniform_state_attribution=[dict(method=k[0],tier=k[1],dimension=k[2],category=k[3],**{n:str(v) for n,v in c.items()}) for k,c in sorted(uniform.items())],
        study_cpu_seconds=wallcpu,all_equalities_exact=all(e.get('identity_exact',False) for e in episodes),
        no_new_roots=True,no_training=True)
    out.mkdir(parents=True)
    with (out/'state_residuals.jsonl').open('x') as f:
        for row in allrows:f.write(json.dumps(row,sort_keys=True)+'\n')
    for name,value in [('occupancy_decomposition.json',episodes),('small_vs_large.json',dict(fresh_state_pairs=comparisons,episode_pairs=episode_comparisons)),
                       ('supplemental_labels.json',supplemental),('summary.json',summary)]:
        (out/name).write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('census_rows','episodes','supplemental_states','all_equalities_exact','fresh_probe_summary')},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='concentration_curve'} for r in occtable],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.data,a.output)
