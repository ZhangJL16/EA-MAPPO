"""Evidence audit for P1; no rollout, model fit or downstream experiment."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'artifacts/regenerative_p1_20260919_v2'
f=json.loads((p/'contract.json').read_text())
for name,digest in f['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
model=ROOT/'artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip'
assert hashlib.sha256(model.read_bytes()).hexdigest()==f['checkpoint_sha256']
rows=[]
for group in ['near_light','far_dense']:
    for option in ['task','return']:
        for i in range(f['n']):
            envelope=json.loads((p/f'{group}_{option}_{i:03d}.json').read_text())
            r=envelope['row']
            assert hashlib.sha256(json.dumps(r,sort_keys=True).encode()).hexdigest()==envelope['sha256']
            assert (r['state_group'],r['option'],r['replicate'])==(group,option,i)
            assert r['audit']['reset_calls']==1
            assert abs(r['audit']['energy_balance_residual'])<1e-8
            assert r['elapsed_time']>0 and r['energy_used']>=0
            rows.append(r)
t=json.loads((p/'trajectory.json').read_text())
assert t['audit']['reset_calls']==1
assert abs(t['audit']['energy_balance_residual'])<1e-8
assert all(e['reset_calls']==1 for e in t['events'])
assert all(a['time']<=b['time'] for a,b in zip(t['events'],t['events'][1:]))
for a,b in zip(t['records'],t['records'][1:]):
    assert a['next_macro_state']==b['start_state'], 'macro continuity'
for r in t['records']:
    assert r['collision_count']==sum(leg['collision_count'] for leg in r['legs'])
    for a,b in zip(r['legs'],r['legs'][1:]):
        for k in ['position','velocity','energy','time','collision_count']:
            assert a['end_state'][k]==b['start_state'][k], ('leg continuity',k)
    for leg in r['legs']:
        assert abs(leg['start_energy']-leg['end_energy']-leg['energy_used'])<1e-8
reg=[e for e in t['events'] if e['event']=='recharge_complete']
assert reg
for e in reg:
    assert e['energy']==60 and e['position']==[2880,2000,200]
    assert e['velocity']==[0,0,0] and e['task'] is None and not e['previous_contact']
for i,e in enumerate(t['events']):
    if e['event']=='recharge_complete':
        assert t['events'][i+1]['event']=='task_available'
    if e['event']=='docking_start':
        nxt=t['events'][i+1]
        assert nxt['event']=='recharge_start'
        distance=np.linalg.norm(np.array(e['position'])-np.array(nxt['position']))
        assert abs(nxt['time']-e['time']-distance)<1e-5
        assert abs(e['energy']-nxt['energy']-.02*distance)<1e-6
summary=dict(passed=True,raw_macro_rows=len(rows),trace_macro_transitions=len(t['records']),
    trace_reset_calls=1,complete_regenerations=len(reg),source_and_checkpoint_hashes_match=True,
    continuous_time_battery_position_velocity=True,paid_docking_verified=True,
    canonical_regeneration_verified=True,task_rng_independence='by IID generator construction; not inferred from six cycles',
    p2_kernel_complete=False,neural_updates=0)
(p/'audit_receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
