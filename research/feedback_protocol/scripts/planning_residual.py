"""Exact decision residuals and full-policy-state occupancy; no learner training."""
import argparse
from collections import defaultdict, Counter
from copy import deepcopy
from fractions import Fraction as F
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic, process_time
import zipfile
import fcntl
from fpl.belief import PlannerState, outcomes
from fpl.teacher_data import state_key, exact_label, advance
from fpl.protocols import validate_protocol, PlanningLimit
from fpl.utility import protocol_value
from fpl.work import WorkPolicy, WatchdogExpired
from fpl.policies.beam_bayes import BeamBayes
from fpl.policies.one_step_voi import OneStepVOI
from fpl.teachers.exact_policy import ExactPolicyEvaluator
from fpl.evaluation import digest, save_new, source_hashes
from fpl_v2.generator import from_json

ROOT=Path(__file__).resolve().parents[1]


def policy(p,method,tier,plan):
    agent=OneStepVOI(p,width=plan['beam_width']) if method=='one_step_voi' else BeamBayes(p,width=plan['beam_width'],depth=plan['beam_depth'])
    limits=plan['tiers'][tier]
    return WorkPolicy(p,agent,limits,{k:v*plan['episode_multiplier'] for k,v in limits.items()},
                      watchdog_seconds=plan['policy_watchdog_seconds'])


def protocol_type(p,operations):
    if operations is None:return 'stop'
    by={o.name:o for o in p.operations}
    n=sum(len(by[name].channels) for name in operations)
    if n:return 'joint_sensing' if n>1 else 'single_sensing'
    return 'task' if any(by[name].utility and any(by[name].utility.by_hypothesis) for name in operations) else 'empty_or_recovery'


def decision(p,state,label,route):
    if label['status']!='exact':raise PlanningLimit('exact label unresolved')
    names=None if route is None else list(route.operations)
    if route is not None:validate_protocol(p,route.operations,state.remaining)
    chosen=next((c for c in label['candidates'] if c['operations']==names),None)
    if chosen is None:raise AssertionError('planner selected action outside teacher catalogue')
    residual=F(label['value'])-F(chosen['q'])
    if residual<0:raise AssertionError('negative Bellman residual')
    types=sorted({protocol_type(p,n) for n in label['optimal_set']})
    flags=[t.endswith('sensing') for t in types]
    required=all(flags)
    return dict(state_id=state_key(state),posterior=list(map(str,state.posterior)),H=state.remaining,B=p.capacity,
                value=label['value'],action=names,q=chosen['q'],residual=str(residual),
                optimal_set=label['optimal_set'],optimal_types=types,chosen_type=protocol_type(p,names),
                prior_state=state.posterior==p.prior,measurement_required=required,
                decision_class='measurement_required' if required else 'measurement_optional' if any(flags) else 'nonmeasuring_optimal')


class Labels:
    def __init__(self,p,records,plan):
        self.p,self.plan=p,plan
        self.cache={r['id']:r['label'] for r in records}
        self.original=set(self.cache);self.supplemental=[]

    def get(self,state):
        sid=state_key(state)
        if sid not in self.cache:
            label=exact_label(self.p,state,self.plan['supplemental_label_limits'])
            self.cache[sid]=label
            self.supplemental.append(dict(id=sid,state=dict(posterior=list(map(str,state.posterior)),
                remaining=state.remaining,resource=state.resource),label=label))
        if self.cache[sid]['status']!='exact':raise PlanningLimit('supplemental teacher unresolved')
        return self.cache[sid]


def decompose(p,agent,labels,plan):
    rows=[];start=monotonic();limit=plan['evaluation']
    def visit(owned,state,mass,parent=None):
        if len(rows)>=limit['max_nodes']:raise PlanningLimit('occupancy node cap')
        if monotonic()-start>limit['watchdog_seconds']:raise WatchdogExpired('occupancy watchdog')
        if state.remaining==0:return F(0)
        before=dict(expansions=owned.expansions,model_calls=owned.model_calls)
        tick=monotonic();route=owned.select(state);latency=monotonic()-tick
        d=decision(p,state,labels.get(state),route)
        node=len(rows)
        rows.append(dict(d,node=node,parent=parent,occupancy=str(mass),weighted_residual=str(mass*F(d['residual'])),
                         history=list(state.history.released),work_before=before,
                         work_after=dict(expansions=owned.expansions,model_calls=owned.model_calls),
                         work=dict(owned.stats),latency_seconds=latency,in_census=d['state_id'] in labels.original))
        if route is None:return F(0)
        value=protocol_value(p,state.posterior,route)
        for feedback,prob,_ in outcomes(p,state.posterior,route.channels,limit['max_feedback']):
            value+=prob*visit(deepcopy(owned),advance(p,state,route,feedback),mass*prob,node)
        return value
    try:
        root=PlannerState(p.prior,p.budget,p.capacity)
        optimal=F(labels.get(root)['value'])
        value=visit(deepcopy(agent),root,F(1))
        weighted=sum(F(r['weighted_residual']) for r in rows)
        if weighted!=optimal-value:raise AssertionError('Bellman telescoping identity failed')
        # Separate preexisting evaluator uses hypothesis-wise integration.
        check=ExactPolicyEvaluator(p,**limit).evaluate(agent)
        if check['status']!='exact_conditional_policy':
            return dict(status='unresolved_crosscheck',rows=rows,check=check)
        checkvalue=sum(b*F(v) for b,v in zip(p.prior,check['utility_by_hypothesis']))
        if checkvalue!=value:raise AssertionError('independent policy integration disagreement')
        expected_exp=sum(F(r['occupancy'])*r['work'].get('expansions',0) for r in rows)
        expected_calls=sum(F(r['occupancy'])*r['work'].get('model_calls',0) for r in rows)
        for field,want in [('expansions_by_hypothesis',expected_exp),('model_calls_by_hypothesis',expected_calls)]:
            if sum(b*F(v) for b,v in zip(p.prior,check[field]))!=want:raise AssertionError('work integration mismatch')
        return dict(status='exact',value_star=str(optimal),value_policy=str(value),gap=str(optimal-value),
                    weighted_residual_sum=str(weighted),identity_exact=True,policy_crosscheck=check,
                    expected_expansions=str(expected_exp),expected_model_calls=str(expected_calls),rows=rows)
    except (PlanningLimit,WatchdogExpired) as exc:
        return dict(status='unresolved',reason=str(exc),rows=rows)


def load(archive,plan):
    if sha256(Path(archive).read_bytes()).hexdigest()!=plan['census_archive_sha256']:raise ValueError('wrong census')
    shards=[]
    with zipfile.ZipFile(archive) as z:
        hashes=json.loads(z.read('FILES_SHA256.json'))
        for name in sorted(z.namelist()):
            if name.startswith('data/') and name.endswith('.census.json'):
                raw=z.read(name)
                if sha256(raw).hexdigest()!=hashes[name]:raise ValueError('archive content mismatch')
                packed=json.loads(raw)
                if packed['hash']!=digest(packed['payload']):raise ValueError('census receipt mismatch')
                shards.append(packed['payload'])
    if len(shards)!=32 or sum(len(s['records']) for s in shards)!=1171:raise ValueError('wrong study scope')
    return shards


def variant(shard,plan):
    p=from_json(shard['problem']);labels=Labels(p,shard['records'],plan)
    rows=[];episodes=[];start=monotonic();cpu=process_time()
    for method in plan['methods']:
        for tier in plan['tiers']:
            for r in shard['records']:
                s=r['state'];state=PlannerState(tuple(map(F,s['posterior'])),s['remaining'],s['resource'])
                agent=policy(p,method,tier,plan)
                tick=monotonic()
                try:
                    route=None if state.remaining==0 else agent.select(state)
                    d=decision(p,state,r['label'],route)
                    rows.append(dict(d,status='exact',method=method,tier=tier,work=agent.stats,
                                     latency_seconds=monotonic()-tick,probe='fresh_work_pool'))
                except (PlanningLimit,WatchdogExpired) as exc:
                    rows.append(dict(state_id=r['id'],method=method,tier=tier,status='unresolved',reason=str(exc)))
            episode=decompose(p,policy(p,method,tier,plan),labels,plan)
            episodes.append(dict(episode,method=method,tier=tier))
    return dict(metadata=shard['metadata'],problem=shard['problem'],state_residuals=rows,
                occupancy=episodes,supplemental_labels=labels.supplemental,
                cpu_seconds=process_time()-cpu,wall_seconds=monotonic()-start)


def run(archive,plan,out,max_new=None,resume=False):
    shards=load(archive,plan);out=Path(out)
    seal=dict(plan=plan,source_hashes=source_hashes(),script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
              v2_generator_sha256=sha256((ROOT/'src/fpl_v2/generator.py').read_bytes()).hexdigest(),
              schedule=[s['metadata'] for s in shards])
    if out.exists():
        if not resume or json.loads((out/'seal.json').read_text())!=seal:raise ValueError('resume seal mismatch')
    else:out.mkdir(parents=True);save_new(out/'seal.json',seal)
    with (out/'writer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);count=0
        for s in shards:
            stem=s['problem']['instance_id'];target=out/(stem+'.residual.json')
            if target.exists():
                packed=json.loads(target.read_text())
                if digest(packed['payload'])!=packed['hash'] or packed['payload']['seal_hash']!=digest(seal):raise ValueError('bad receipt')
                continue
            if max_new is not None and count>=max_new:break
            marker=out/(stem+'.started.json')
            if marker.exists():raise RuntimeError('interrupted; explicit recovery required')
            save_new(marker,dict(status='started',metadata=s['metadata']))
            result=variant(s,plan);result['seal_hash']=digest(seal)
            save_new(target,dict(hash=digest(result),payload=result));count+=1
            print(json.dumps(dict(variant=stem,rows=len(result['state_residuals']),
                episodes=[dict(method=e['method'],tier=e['tier'],status=e['status'],gap=e.get('gap')) for e in result['occupancy']],
                seconds=result['wall_seconds'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--plan',required=True)
    p.add_argument('--output',required=True);p.add_argument('--max-new',type=int);p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.archive,json.loads(Path(a.plan).read_text()),a.output,a.max_new,a.resume)
