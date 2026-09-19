"""Read-only raw provenance/invariant audit, independent of local-value fitting."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from research.regenerative_control.census_p2a1 import physical,digest
from research.regenerative_control.qualify import ROOT,MODEL,EXPECTED,sha,atomic


def no_rng(value):
    if isinstance(value,dict):
        assert 'task_rng' not in value
        for v in value.values():no_rng(v)
    elif isinstance(value,list):
        for v in value:no_rng(v)


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args();root=args.input
    census=json.loads((root/'census.json').read_text());contract=json.loads((root/'contract.json').read_text())
    assert sha(MODEL)==EXPECTED==contract['checkpoint_sha256']
    for path,h in contract['source_hashes'].items():assert sha(ROOT/path)==h,path
    provenance=json.loads((root/'collection_provenance.json').read_text());old=ROOT/provenance['collection_root']
    assert sha(old/'contract.json')==provenance['contract_sha256']
    old_contract=json.loads((old/'contract.json').read_text())
    assert sha(old/'collector_source_v1.txt')==old_contract['source_hashes']['research/regenerative_control/census_p2a1.py']
    for path,h in old_contract['source_hashes'].items():
        if path!='research/regenerative_control/census_p2a1.py':assert sha(ROOT/path)==h
    counts=dict(states=0,core_branches=0,core_rng_draw_violations=0,diagnostic_missions=0,
        diagnostic_successes=0,diagnostic_navigation_timeouts=0,diagnostic_depletions=0)
    overshoots=[]
    for meta in census['states']:
        raw=(root/meta['snapshot']).read_bytes();assert hashlib.sha256(raw).hexdigest()==meta['snapshot_sha256']
        e=pickle.loads(raw);assert physical(e)==meta['state'];assert digest(physical(e))==meta['state_id']
        assert e.phase=='decision' and e.task is None and not e.failed and e.base.reset_calls==1
        source=json.loads((root/'sources'/meta['trajectory_id']/'trajectory.json').read_text())
        assert meta in source['states'] and meta['natural_occupancy'];e.close()
        env=json.loads((root/'branches'/f"{meta['state_id']}.json").read_text());r=env['row'];assert digest(r)==env['sha256']
        assert r['state']==meta and r['task_probabilities']==[1/3]*3 and not r['task_rng_read'];no_rng(r)
        allmacros=[]
        for name,b in r['branches'].items():
            q=b['row'];s=q['start_state'];n=q['next_macro_state'];counts['core_branches']+=1
            assert s['energy']==meta['state']['energy'] and s['position']==meta['state']['position'] and s['velocity']==meta['state']['velocity']
            assert s['task'] is None and s['phase']=='decision' and s['map_id']==meta['state']['map_id']
            assert n['task_draw_count']-s['task_draw_count']==int(name!='R')
            if name=='R' and q['success']:
                assert n['phase']=='home' and n['task'] is None and n['energy']==60 and n['velocity']==[0,0,0] and n['position']==[2880,2000,200]
            if name!='R' and q['success']:assert n['phase']=='decision' and n['task'] is None
            allmacros.append(q)
        for m in r['missions'].values():
            a=m['actual'];assert a['success']==bool(a['task']['success'] and a['return_after_task'] and a['return_after_task']['success'])
            if a['return_after_task']:allmacros.append(a['return_after_task'])
            d=m['diagnostic']
            if d:
                counts['diagnostic_missions']+=1
                assert not a['success'] and not d['natural_occupancy'] and d['initial_energy']==d['capacity']==10000
                assert m['requirement']['counterfactual_energy_intervention']
                steps=[d['task']]+([d['return_after_task']] if d['return_after_task'] else [])
                counts['diagnostic_successes']+=int(all(x['success'] for x in steps) and len(steps)==2)
                counts['diagnostic_navigation_timeouts']+=int(any(x['timeout'] for x in steps))
                counts['diagnostic_depletions']+=int(any(x['terminal_reason']=='energy_depletion' for x in steps))
                allmacros+=steps
            else:
                assert a['success'] and not m['requirement']['counterfactual_energy_intervention']
                assert abs(m['requirement']['energy']-sum(a[k]['energy_used'] for k in ['task','return_after_task']))<1e-9
        for q in allmacros:
            assert q['elapsed_time']>0 and q['energy_used']>=0
            assert q['next_macro_state']['reset_calls']==1
            assert q['collision']==(q['collision_count']>0)
            for leg in q['legs']:
                residual=leg['energy_used']-(leg['start_energy']-leg['end_energy'])
                if leg['terminal_reason']=='energy_depletion':overshoots.append(residual)
                else:assert abs(residual)<1e-7
        counts['states']+=1
    counts.update(pass_invariants=True,checkpoint_unchanged=True,neural_updates=0,
        actual_battery_interventions=0,diagnostic_capacity_correction=True,
        navigation_depletion_raw_energy_overrun_max=max(overshoots,default=0),
        depletion_accounting_note='Frozen physics logs full final integration-step demand but clips battery at zero; no new dynamics correction.')
    atomic(root/'raw_audit.json',counts);print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
