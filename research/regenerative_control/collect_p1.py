"""Bounded P1 smoke collection, source-bound atomic resume, no P2 solver."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import pickle

import cloudpickle
import numpy as np
import torch

from research.regenerative_control.continuing import Config, ContinuingUAV
from research.regenerative_control.qualify import ROOT, EXPECTED, MODEL, actor_for, atomic, sha


def dump(path, value):
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        cloudpickle.dump(value, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def contract(n, cutoff):
    files = [ROOT/'envs/UAVEnergyDeliverySAC.py']
    for directory in ['research/regenerative_control', 'experiments/directional_navigation',
                      'experiments/jacobian_energy_bridge', 'runtime_support/review_bundle']:
        files.extend(sorted((ROOT/directory).rglob('*.py')))
    return dict(version='P1_v2_cutoff', checkpoint_sha256=EXPECTED, n=n, trace_cutoff=cutoff,
        collector_distribution='fixed layout per group; independent initial vx/vy U(-1,1), vz U(-.2,.2); deterministic frozen actor',
        task_process='IID uniform distances 100/300/600; always available; value 1',
        config=asdict(Config()), source_hashes={str(p.relative_to(ROOT)):sha(p) for p in files})


def fixture(group, replicate):
    far = group == 'far_dense'
    env = ContinuingUAV(Config(obstacles=24 if far else 4),
        layout_seed=319190002 if far else 319190001, stream_seed=519190001+replicate)
    d = 600 if far else 100
    home=env.base.charger_position
    env.task=dict(index=2 if far else 0, value=1., pickup=(home-[d,0,0]).tolist(),
                  dropoff=(home+[-d,d,0]).tolist())
    rng=np.random.default_rng(np.random.SeedSequence([619190001, int(far), replicate]))
    velocity=rng.uniform([-1,-1,-.2],[1,1,.2]).astype(np.float32)
    return env, velocity


def collect_one(group, option, replicate, actor):
    env, velocity = fixture(group, replicate)
    if option == 'return':
        # Independent conditional macro trial, not a reset within a trajectory.
        env.base.agent.pos=np.array(env.task['dropoff'],np.float32)
        env.base.agent.prev_pos=env.base.agent.pos.copy()
        env.base.agent.last_pos=env.base.agent.pos.copy()
    env.base.agent.vel=velocity
    env.base._update_lidar()
    start=env.state(full=True)
    row=env.execute('C' if option=='task' else 'R',actor)
    row.update(state_id=hashlib.sha256(json.dumps(start,sort_keys=True).encode()).hexdigest(),
               state_group=group, option=option, replicate=replicate,
               audit=env.audit(),
               raw_state_visibility='privileged audit; actor gets only frozen 2056 sensor/goal vector')
    env.close()
    return row


def run_trace(path, actor, cutoff):
    if path.exists():
        with path.open('rb') as f: saved=pickle.load(f)
        env, records=saved['env'],saved['records']
    else:
        env=ContinuingUAV(Config(cutoff=cutoff,obstacles=24),layout_seed=319190002,
                          stream_seed=719190001)
        records=[]
    while not (env.failed or env.truncated):
        # Fixed diagnostic schedule, NOT a proposed controller or baseline search.
        since=env.completed-env.cycle_start_tasks
        action='R' if since>=2 or env.energy<30 else 'C'
        records.append(env.execute(action,actor))
        dump(path,dict(env=env,records=records))
    return dict(audit=env.audit(),records=records,events=env.events,cycles=env.cycles)


def summarize(rows):
    table=[]
    for group in ('near_light','far_dense'):
        for option in ('task','return'):
            selected=[r for r in rows if r['state_group']==group and r['option']==option]
            if not selected: continue
            # Failed/censored expenditure is never a completed-option quantile.
            good=[r for r in selected if r['success']]
            values=[r['energy_used'] for r in good]
            table.append(dict(state_group=group,option=option,n=len(selected),
                success=sum(r['success'] for r in selected),
                failures=sum(r['audit']['failed'] for r in selected),
                contact_trials=sum(r['collision'] for r in selected),
                mean_energy_all_observed=float(np.mean([r['energy_used'] for r in selected])),
                mean_energy_success=float(np.mean(values)) if values else None,
                q95_energy_success=float(np.quantile(values,.95,method='higher')) if values else None,
                q95_unconditional_completion_energy=(float(np.quantile(values,.95,method='higher'))
                                                      if len(good)==len(selected) else None),
                mean_time_all_observed=float(np.mean([r['elapsed_time'] for r in selected])),
                mean_navigation_time=float(np.mean([sum(x['elapsed_time'] for x in r['legs']) for r in selected]))))
    return table


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--n',type=int,default=32)
    parser.add_argument('--trace-cutoff',type=float,default=1000.)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-new-rows',type=int)
    args=parser.parse_args()
    if args.n<=0 or args.trace_cutoff<=0: raise ValueError('positive bounded smoke sizes required')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    freeze=contract(args.n,args.trace_cutoff)
    if sha(MODEL)!=EXPECTED: raise RuntimeError('navigation checkpoint changed')
    manifest=args.output/'contract.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())!=freeze:
            raise RuntimeError('resume requires identical contract')
    else: atomic(manifest,freeze)
    template=ContinuingUAV(Config(obstacles=4))
    actor=actor_for(template)
    template.close()
    rows=[]
    new=0
    # Interleave cells so the first checkpoint contains all four strata.
    for i in range(args.n):
        for group in ('near_light','far_dense'):
            for option in ('task','return'):
                path=args.output/f'{group}_{option}_{i:03d}.json'
                if path.exists():
                    envelope=json.loads(path.read_text())
                    row=envelope['row']
                    if hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()!=envelope['sha256']:
                        raise RuntimeError('corrupt committed row')
                    if (row['state_group'],row['option'],row['replicate'])!=(group,option,i):
                        raise RuntimeError('row identity mismatch')
                else:
                    if args.max_new_rows is not None and new>=args.max_new_rows:
                        atomic(args.output/'status.json',dict(status='paused',rows=len(rows),planned=4*args.n))
                        return
                    row=collect_one(group,option,i,actor)
                    if abs(row['audit']['energy_balance_residual'])>1e-8:
                        raise RuntimeError('energy accounting mismatch')
                    atomic(path,dict(row=row,sha256=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()))
                    new+=1
                    print(f"saved {group}/{option}/{i}: success={row['success']}",flush=True)
                rows.append(row)
                atomic(args.output/'startup_health.json',dict(rows=len(rows),finite=True,
                    checkpoint_unchanged=sha(MODEL)==EXPECTED,reset_calls=row['audit']['reset_calls'],
                    energy_balance_residual=row['audit']['energy_balance_residual'],neural_updates=0))
    table=summarize(rows)
    atomic(args.output/'smoke_table.json',table)
    # Key execution/energy failures stop before the continuing demonstration.
    if any(r['audit']['failed'] or r['censored'] for r in rows):
        atomic(args.output/'status.json',dict(status='STOP_macro_failure',rows=len(rows),table=table))
        return
    trace=run_trace(args.output/'trace_checkpoint.pkl',actor,args.trace_cutoff)
    atomic(args.output/'trajectory.json',trace)
    if sha(MODEL)!=EXPECTED: raise RuntimeError('navigator was modified')
    atomic(args.output/'status.json',dict(status='STOP_trace_failure' if trace['audit']['failed'] else 'P1_smoke_complete',
        rows=len(rows),trace_audit=trace['audit'],p2_started=False,training_updates=0))


if __name__=='__main__': main()
