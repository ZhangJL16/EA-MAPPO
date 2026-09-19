"""Only three predeclared 1000s semantic sanity traces; no oracle or census."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle

import torch

from research.regenerative_control.continuing import Config
from research.regenerative_control.continuing_p11 import PostDeliveryUAV
from research.regenerative_control.collect_p1 import contract, dump
from research.regenerative_control.qualify import MODEL, EXPECTED, actor_for, atomic, sha

POLICIES=('after_1','after_2','threshold_30')


def decision(env, policy):
    if env.phase!='decision' or env.task is not None:
        raise ValueError('policy may only act at task-free D-state')
    if policy=='threshold_30': return 'C' if env.energy>30 else 'R'
    count=1 if policy=='after_1' else 2
    return 'R' if env.completed-env.cycle_start_tasks>=count else 'C'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--max-new-transitions',type=int)
    args=parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    freeze=contract(0,1000.)
    freeze.pop('n'); freeze.pop('collector_distribution')
    freeze.update(version='P1.1_post_delivery',policies=list(POLICIES),
        task_process='draw IID uniform 100/300/600 only after C or forced H departure',
        layout_seed=319190002,stream_seed=719190001,
        scope='three bounded 1000s sanity traces; no P2',
        config=vars(Config(obstacles=24,cutoff=1000.)))
    digest=hashlib.sha256(json.dumps(freeze,sort_keys=True).encode()).hexdigest()
    manifest=args.output/'contract.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())!=freeze:
            raise RuntimeError('resume needs identical P1.1 contract')
    else: atomic(manifest,freeze)
    if sha(MODEL)!=EXPECTED: raise RuntimeError('frozen SAC mismatch')
    template=PostDeliveryUAV(Config(obstacles=0)); actor=actor_for(template); template.close()
    new=0; table=[]
    for policy in POLICIES:
        path=args.output/f'{policy}.pkl'
        if path.exists():
            with path.open('rb') as f: saved=pickle.load(f)
            if saved['contract']!=digest: raise RuntimeError('checkpoint contract mismatch')
            env,records=saved['env'],saved['records']
        else:
            env=PostDeliveryUAV(Config(obstacles=24,cutoff=1000.),
                               layout_seed=319190002,stream_seed=719190001)
            records=[]
        while not (env.failed or env.truncated):
            if args.max_new_transitions is not None and new>=args.max_new_transitions:
                atomic(args.output/'status.json',dict(status='paused',policy=policy,
                       completed_macros=len(records),P2_started=False))
                return
            r=(env.advance_home(actor) if env.phase=='home'
               else env.execute(decision(env,policy),actor))
            records.append(r); new+=1
            dump(path,dict(contract=digest,env=env,records=records))
            atomic(args.output/'startup_health.json',dict(policy=policy,macros=len(records),
                reset_calls=env.base.reset_calls,checkpoint_unchanged=sha(MODEL)==EXPECTED,
                audit=env.audit(),neural_updates=0))
        trace=dict(policy=policy,audit=env.audit(),records=records,events=env.events,cycles=env.cycles)
        atomic(args.output/f'{policy}.json',trace)
        table.append(dict(policy=policy,deliveries=env.completed,recharge=len(env.cycles),
            depletion=int(env.failure_reason=='energy_depletion'),failure_reason=env.failure_reason,
            collision_count=env.contacts,elapsed_time=env.time,cutoff=1000.,
            task_draws=env.task_draw_count,reset_calls=env.base.reset_calls))
        atomic(args.output/'sanity_table.json',table)
        print(json.dumps(table[-1]),flush=True)
        env.close()
    assert sha(MODEL)==EXPECTED
    atomic(args.output/'status.json',dict(status='P1.1_sanity_complete',P2_started=False,
           neural_updates=0,table=table))


if __name__=='__main__': main()
