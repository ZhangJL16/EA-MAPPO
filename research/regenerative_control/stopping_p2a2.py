"""P2-A2 exact safe task-sequence tree, resumable at node boundaries.

No random census, state bins, RNG lookahead, learned model or new training.
A STOP file requests a graceful stop after the current worker batch.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import pickle
import signal

import torch

from research.regenerative_control.census_p2a1 import compact,enumerated,physical,digest,make_contract
from research.regenerative_control.continuing import Config
from research.regenerative_control.continuing_p11 import PostDeliveryUAV
from research.regenerative_control.qualify import ROOT,MODEL,EXPECTED,sha,atomic,actor_for
from research.regenerative_control.stopping_dp import solve

TASKS=(100,300,600)
CANDIDATE='e5352ba4f4a9a467a1ba50cbf4b747ba522fdf1cacad4be12d036f02d62d300a'
OLD=ROOT/'artifacts/regenerative_p2a1_20260919_v2'
ACTOR=None
STOP=False


def init_worker():
    global ACTOR
    torch.set_num_threads(1)
    e=PostDeliveryUAV(Config(obstacles=24),layout_seed=319190002)
    ACTOR=actor_for(e);e.close()


def write_bytes(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,path)


def envelope(path,row):
    atomic(path,dict(row=row,sha256=digest(row)))


def read_envelope(path):
    e=json.loads(path.read_text())
    if digest(e['row'])!=e['sha256']:raise RuntimeError(f'corrupt receipt: {path}')
    return e['row']


def descriptor(root,raw,origin,sequence,return_receipt=None):
    key=digest(dict(origin=origin,sequence=sequence))
    path=root/'snapshots'/f'{key}.pkl'
    write_bytes(path,raw)
    e=pickle.loads(raw);s=physical(e);e.close()
    return dict(id=key,origin=origin,sequence=sequence,snapshot=str(path.relative_to(root)),
        snapshot_sha256=hashlib.sha256(raw).hexdigest(),physical=s,return_receipt=return_receipt)


def payload(root,desc):
    raw=(root/desc['snapshot']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=desc['snapshot_sha256']:raise RuntimeError('snapshot mismatch')
    return raw


def audited_macro(row,action):
    if row['censored'] or row['elapsed_time']<=0:raise RuntimeError('incomplete/invalid macro duration')
    s,n=row['start_state'],row['next_macro_state']
    if row['success'] and row['reward']!=int(action!='R'):
        raise RuntimeError('unexpected task reward')
    if s['task'] is not None:raise RuntimeError('pending task at decision')
    if n['task_draw_count']-s['task_draw_count']!=int(action!='R'):raise RuntimeError('task RNG/draw semantics')
    if s['reset_calls']!=1 or n['reset_calls']!=1:raise RuntimeError('hidden reset')
    if row['success'] and action!='R':
        if n['phase']!='decision' or n['task'] is not None or n['energy']>=s['energy']-1e-9:
            raise RuntimeError('successful task must strictly consume energy and end in D')
    def check(v):
        if isinstance(v,dict):
            if 'task_rng' in v:raise RuntimeError('RNG feature leak')
            for w in v.values():check(w)
        elif isinstance(v,list):
            for w in v:check(w)
    check(row)


def return_branch(raw,canonical):
    e=enumerated(raw);row=e.execute('R',ACTOR);audited_macro(row,'R')
    state=physical(e)
    if row['success'] and state!=canonical:raise RuntimeError('noncanonical physical H')
    e.close();return dict(row=row,end_physical=state)


def mission_diagnostic(raw,d,actual_task,actual_return):
    """Cause attribution only; no diagnostic data enter admissibility or DP."""
    observed=[actual_task]+([actual_return] if actual_return else [])
    energy=any(x['terminal_reason']=='energy_depletion' for x in observed)
    nav=any(x['timeout'] for x in observed)
    diagnostic=None
    if energy:
        u=enumerated(raw,d,energy_intervention=True)
        c=u.execute('C',ACTOR);audited_macro(c,'C')
        r=u.execute('R',ACTOR) if c['success'] else None
        if r:audited_macro(r,'R')
        nav=nav or c['timeout'] or bool(r and r['timeout'])
        if c['terminal_reason']=='energy_depletion' or (r and r['terminal_reason']=='energy_depletion'):
            raise RuntimeError('diagnostic energy budget exhausted; cause unresolved')
        diagnostic=dict(initial_energy=10000.,capacity=10000.,intervention=True,
            used_for_safe_set=False,task=c,return_after_task=r)
        u.close()
    cause='both' if energy and nav else ('energy' if energy else ('navigation' if nav else 'unknown'))
    if cause=='unknown':raise RuntimeError('unclassified mission failure')
    return dict(cause=cause,observed_terminal_reasons=[x['terminal_reason'] for x in observed],diagnostic=diagnostic)


def robust_continue(cases):
    return len(cases)==3 and all(x['task']['success'] and x['return_after_task'] is not None
                               and x['return_after_task']['row']['success'] for x in cases)


def expand(job):
    root,desc,canonical=job;root=Path(root);path=root/'nodes'/f"{desc['id']}.json"
    if path.exists():
        row=read_envelope(path)
        if row['descriptor']!=desc:raise RuntimeError('node descriptor changed')
        payload(root,desc)
        for child in row['children']:payload(root,child)
        return row
    raw=payload(root,desc)
    R=desc['return_receipt'] or return_branch(raw,canonical)
    audited_macro(R['row'],'R')
    if not R['row']['success']:raise RuntimeError('reachable safe-tree D cannot return; gate failed')
    cases=[];children=[]
    for d in TASKS:
        e=enumerated(raw,d);c=e.execute('C',ACTOR);audited_macro(c,'C')
        child=None;r=None
        if c['success']:
            childraw=compact(e)
            r=return_branch(childraw,canonical)
            child=descriptor(root,childraw,desc['origin'],desc['sequence']+[d],r)
            children.append(child)
        cause=None
        if not (c['success'] and r and r['row']['success']):
            cause=mission_diagnostic(raw,d,c,r['row'] if r else None)
        cases.append(dict(task_type=d,probability=1/3,task=c,child_id=child['id'] if child else None,
                          return_after_task=r,infeasibility=cause))
        e.close()
    safe=robust_continue(cases)
    row=dict(descriptor=desc,R=R,cases=cases,safe_C=safe,children=children if safe else [],
        unexpanded_children_reason=None if safe else 'C illegal: at least one task+return fails',
        model=dict(R_safe=True,tau_R=R['row']['elapsed_time'],safe_C=safe,
            children=[dict(id=c['child_id'],probability=1/3,time=c['task']['elapsed_time']) for c in cases] if safe else []))
    envelope(path,row)
    return row


def prepare(root):
    path=root/'root.json'
    if path.exists():
        r=read_envelope(path)
        for d in r['descriptors']:payload(root,d)
        return r
    # Use an actual paid recharge completion, not a manually reset H state.
    e=PostDeliveryUAV(Config(obstacles=24),layout_seed=319190002)
    w=enumerated(compact(e),100);e.close()
    warm_task=w.advance_home(ACTOR);audited_macro(warm_task,'forced_task')
    if not warm_task['success']:raise RuntimeError('canonical-H setup task failed')
    warm_return=w.execute('R',ACTOR);audited_macro(warm_return,'R')
    if not warm_return['success']:raise RuntimeError('canonical-H setup recharge failed')
    canonical=physical(w);hraw=compact(w);write_bytes(root/'canonical_H.pkl',hraw);w.close()
    roots=[];descs=[];first=[]
    for d in TASKS:
        e=enumerated(hraw,d);c=e.advance_home(ACTOR);audited_macro(c,'forced_task')
        if not c['success']:raise RuntimeError('forced H task fails: no robust regenerative policy')
        raw=compact(e);r=return_branch(raw,canonical)
        if not r['row']['success']:raise RuntimeError('forced H task cannot return safely')
        desc=descriptor(root,raw,'cycle',[d],r)
        roots.append(dict(id=desc['id'],probability=1/3,time=c['elapsed_time']))
        descs.append(desc);first.append(dict(task_type=d,row=c));e.close()
    census=json.loads((OLD/'census.json').read_text())
    m=next(s for s in census['states'] if s['state_id']==CANDIDATE)
    raw=(OLD/m['snapshot']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=m['snapshot_sha256']:raise RuntimeError('candidate snapshot corrupt')
    # A separate exact prefix tree checks the nominated state; its weight in
    # canonical-H renewal optimization is zero. No physical-key merging.
    candidate=descriptor(root,raw,'candidate',[],return_branch(raw,canonical))
    descs.append(candidate)
    r=dict(canonical=canonical,canonical_snapshot_sha256=hashlib.sha256(hraw).hexdigest(),
        setup_excluded_from_cycle=dict(task=warm_task,return_home=warm_return),
        roots=roots,forced_first_tasks=first,descriptors=descs,candidate_id=candidate['id'],
        candidate_source=m,task_rng_read=False)
    envelope(path,r);return r


def protocol():
    hashes=make_contract()['source_hashes']
    for name in ['stopping_p2a2.py','stopping_dp.py']:
        path=ROOT/'research/regenerative_control'/name;hashes[str(path.relative_to(ROOT))]=sha(path)
    return dict(version='P2-A2-v1',checkpoint_sha256=EXPECTED,source_hashes=hashes,
        layout_seed=319190002,capacity=60,task_types=list(TASKS),task_probabilities=[1/3]*3,
        safety='C legal iff every original-energy task then immediate R succeeds; R must reach canonical H',
        oracle='fully enumerated task-prefix tree, ratio DP; no bins/NN/physical-key merging',
        baseline='C at every robust-feasible D; otherwise R',kill_ratio=.98,
        candidate=CANDIDATE,candidate_source_census_sha256=sha(OLD/'census.json'),
        diagnostic='energy and capacity10000 only for failure cause; excluded from safe set and DP',
        incomplete='never infer terminal R at an unexplored safe successor; no ratio on incomplete trees',
        node_work_budget=10000,energy_decrease_check=1e-9,solver_excess_tolerance=1e-11,
        safe_return_strict_advantage=1e-9,neural_updates=0)


def stop_signal(signum,frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--max-new-nodes',type=int)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    lock=open(root/'run.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for sub in ['nodes','snapshots']:(root/sub).mkdir(exist_ok=True)
    contract=protocol();path=root/'contract.json'
    if path.exists():
        if json.loads(path.read_text())!=contract:raise RuntimeError('source/protocol mismatch')
    else:atomic(path,contract)
    if sha(MODEL)!=EXPECTED:raise RuntimeError('SAC checkpoint changed')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    atomic(root/'process.json',dict(pid=os.getpid(),workers=args.workers))
    try:
        atomic(root/'status.json',dict(status='RUNNING',rho_star=None,pid=os.getpid()))
        init_worker();tree=prepare(root);queue=list(tree['descriptors']);seen={};new=0
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker) as pool:
            while queue:
                if STOP or (root/'STOP').exists() or (args.max_new_nodes is not None and new>=args.max_new_nodes):break
                batch=[]
                while queue and len(batch)<args.workers:
                    desc=queue.pop(0)
                    if desc['id'] in seen:raise RuntimeError('duplicate prefix descriptor')
                    cache=root/'nodes'/f"{desc['id']}.json"
                    if cache.exists():
                        row=expand((str(root),desc,tree['canonical']));seen[desc['id']]=row;queue.extend(row['children'])
                    else:
                        batch.append(desc)
                        if args.max_new_nodes is not None and new+len(batch)>=args.max_new_nodes:break
                if len(seen)+len(batch)>contract['node_work_budget']:
                    queue=batch+queue;break
                for row in pool.map(expand,[(str(root),d,tree['canonical']) for d in batch]):
                    seen[row['descriptor']['id']]=row;queue.extend(row['children']);new+=1
                atomic(root/'progress.json',dict(status='RUNNING',completed_nodes=len(seen),new_nodes_this_call=new,
                    frontier=len(queue),max_task_depth=max((len(r['descriptor']['sequence']) for r in seen.values()),default=0),
                    checkpoint_unchanged=sha(MODEL)==EXPECTED,task_rng_read=False,neural_updates=0))
                print(f'nodes={len(seen)} frontier={len(queue)}',flush=True)
        if queue:
            atomic(root/'status.json',dict(status='INCOMPLETE',completed_nodes=len(seen),frontier=len(queue),
                rho_star=None,reason='stop request / work budget / explicit partial invocation'))
            return
        models={k:r['model'] for k,r in seen.items()}
        result=solve(models,tree['roots'],tree['candidate_id'])
        result['node_metadata']={k:dict(origin=r['descriptor']['origin'],sequence=r['descriptor']['sequence'],
            state=r['descriptor']['physical']) for k,r in seen.items()}
        result['counts']=dict(total_nodes=len(seen),cycle_nodes=sum(r['descriptor']['origin']=='cycle' for r in seen.values()),
            candidate_nodes=sum(r['descriptor']['origin']=='candidate' for r in seen.values()),
            safe_C=sum(r['safe_C'] for r in seen.values()),
            causes={cause:sum(c['infeasibility'] is not None and c['infeasibility']['cause']==cause
                    for r in seen.values() for c in r['cases']) for cause in ['energy','navigation','both']})
        result['complete_tree']=True;result['neural_updates']=0
        atomic(root/'result.json',result)
        atomic(root/'status.json',dict(status='COMPLETE',completed_nodes=len(seen),frontier=0,
            decision=result['decision'],rho_star=result['rho_star'],VB_over_oracle=result['VB_over_oracle']))
    except BaseException as exc:
        atomic(root/'status.json',dict(status='ERROR',error=repr(exc),rho_star=None))
        raise

if __name__=='__main__':main()
