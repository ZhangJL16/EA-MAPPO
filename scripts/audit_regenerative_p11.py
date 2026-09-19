"""Audit P1.1 task-draw ordering and persistent physical state from raw traces."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'artifacts/regenerative_p11_20260919'
contract=json.loads((p/'contract.json').read_text())
for name,digest in contract['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
model=ROOT/'artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip'
assert hashlib.sha256(model.read_bytes()).hexdigest()==contract['checkpoint_sha256']
sequences=[]; results=[]
for policy in contract['policies']:
    t=json.loads((p/f'{policy}.json').read_text())
    a=t['audit']; rows=t['records']; events=t['events']
    assert a['reset_calls']==1 and a['task_rejections']==0
    assert 'abandoned' not in a
    assert all(e['reset_calls']==1 for e in events)
    assert all(x['time']<=y['time'] for x,y in zip(events,events[1:]))
    for x,y in zip(rows,rows[1:]): assert x['next_macro_state']==y['start_state']
    draws=0
    for row in rows:
        start,end=row['start_state'],row['next_macro_state']
        assert start['task'] is None
        assert row['action'] in ('forced_task','C','R')
        if row['action']=='forced_task': assert start['phase']=='home'
        else: assert start['phase']=='decision'
        if row['action']=='R':
            assert end['task_draw_count']==start['task_draw_count']
            assert end['task_rng']==start['task_rng']
            assert end['task'] is None
            if row['success']: assert end['phase']=='home' and end['energy']==60
        else:
            draws+=1
            assert end['task_draw_count']==start['task_draw_count']+1
            if row['success']: assert end['phase']=='decision' and end['task'] is None
        for x,y in zip(row['legs'],row['legs'][1:]):
            for k in ('position','velocity','energy','time','collision_count'):
                assert x['end_state'][k]==y['start_state'][k]
    assert draws==a['task_draw_count']
    for i,event in enumerate(events):
        if event['event']=='task_draw':
            assert events[i-1]['event'] in ('decision_C','forced_first_task')
        if event['event']=='delivery_complete':
            assert event['task'] is None and event['phase']=='decision'
        if event['event']=='recharge_complete':
            assert event['task'] is None and event['energy']==60
            assert event['position']==[2880,2000,200] and event['velocity']==[0,0,0]
            assert not event['previous_contact']
            if i+1<len(events): assert events[i+1]['event']=='forced_first_task'
    # Preserve the inherited full-substep demand residual on depletion; do not
    # confuse it with unaccounted reset/refill or silently change flight physics.
    if not a['energy_failure']: assert abs(a['energy_balance_residual'])<1e-8
    sequences.append([e['task']['index'] for e in events if e['event']=='task_draw'])
    results.append(dict(policy=policy,macros=len(rows),draws=draws,reset_calls=1,
                        recharge=a['cycle_count'],energy_balance_residual=a['energy_balance_residual']))
for seq in sequences:
    n=min(map(len,sequences))
    assert seq[:n]==sequences[0][:n], 'policy-dependent task skipping'
receipt=dict(passed=True,decision_has_no_pending_task=True,R_never_draws_or_discards=True,
    H_has_no_CR_choice=True,draws_only_after_C_or_forced_H=True,
    task_sequence_common_prefix_preserved=True,canonical_recharge=True,
    source_and_checkpoint_hashes_match=True,policies=results,
    P2_started=False,neural_updates=0)
(p/'audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
