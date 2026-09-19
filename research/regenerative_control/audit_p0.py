"""Summarize saved qualification evidence and inspect retained energy artifacts."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from research.regenerative_control.qualify import ROOT, MODEL, EXPECTED, atomic

out=ROOT/'artifacts/regenerative_p0_20260919'
rows=[json.loads(p.read_text()) for p in sorted((out/'qualification').glob('route_*.json'))]
manifest=json.loads((out/'qualification/freeze.json').read_text())
assert len(rows)==18 and [r['id'] for r in rows]==list(range(18))
for row, job in zip(rows,manifest['jobs'],strict=True):
    assert all(row[k]==v for k,v in job.items())
    assert row['checkpoint_unchanged'] and row['result']['collision_count']>=0
assert hashlib.sha256(MODEL.read_bytes()).hexdigest()==EXPECTED
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest
summary={}
for role in ['pickup','dropoff','return']:
    group=[r for r in rows if r['role']==role]
    summary[role]=dict(n=len(group),reached=sum(r['result']['goal_reached'] for r in group),
        contact_routes=sum(r['result']['collision_count']>0 for r in group),
        collision_steps=sum(r['result']['collision_count'] for r in group),
        policy_steps=sum(r['result']['steps'] for r in group),
        mean_seconds=float(np.mean([r['elapsed_seconds'] for r in group])),
        mean_energy=float(np.mean([r['result']['energy'] for r in group])))
head_path=ROOT/'artifacts/new_navigation_energy_heads_20260908_v1/latest.pt'
head=torch.load(head_path,map_location='cpu',weights_only=False)
repair=json.loads((ROOT/'artifacts/new_navigation_energy_global_scale_repair_20260908_v1/repair.json').read_text())
assert hashlib.sha256(head_path.read_bytes()).hexdigest()==repair['head_sha256']
assert head['contract']['model_sha256']==EXPECTED
inventory=dict(head_sha256=repair['head_sha256'],models=list(head['models']),
    contract={k:head['contract'][k] for k in ['protocol','arms','quantile','normalization','navigation_updates']},
    return_label_source='experiments/directional_navigation/return_energy_data.py:suffix_labels',
    return_label_semantics='suffix energy and collision-free arrival; not mission+return labels',
    runtime_default_estimator=None,
    four_quantile_checkpoint_identified=False,mission_quantile_checkpoint_identified=False,
    return_coverage=None,mission_coverage=None,
    note='Missing verified coverage is not an observed miscalibration rate. No training or historical held-out outcome analysis performed.')
atomic(out/'energy_inventory.json',inventory)
atomic(out/'summary.json',dict(n=18,by_role=summary,gate='P0_NOT_PASSED_INCOMPLETE_ENERGY_QUALIFICATION',
    navigation='18/18 local navigation legs reached, 0 contact routes; one scene per cell, no reliability certification',
    coverage={'return':None,'task_plus_return':None},
    p1='NOT_STARTED',p2='NOT_STARTED',p3='NOT_STARTED',neural_updates=0,
    all_source_hashes_match=True,checkpoint_sha256=EXPECTED))
print(json.dumps(summary,indent=2))
