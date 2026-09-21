"""Audit saved counterpart branches; no simulator, fitting, or rollout imports."""
from pathlib import Path
from collections import defaultdict, Counter
import hashlib
import json
import math
import statistics
import sys

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / 'atlas_matched_pairs_v1'
TOL = 1e-7

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p): return json.loads(p.read_text())
def rows(p): return [json.loads(s) for s in p.read_text().splitlines()]
def dump(n,x): (OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def jsonl(n,xs): (OUT/n).write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in xs))
def avg(xs): return statistics.mean(xs) if xs else None
def sign(x): return 1 if x>TOL else (-1 if x< -TOL else 0)
def near(x,y): assert abs(x-y)<TOL,(x,y)

# Direction +1 means a larger value is preferred, -1 means smaller.
RULES = {'safe':1,'task_time':-1,'task_return_time':-1,'task_energy':-1,
         'return_reserve':1,'predicted_sjf_rank_all':-1,'predicted_sjf_rank_safe':-1,
         'actual_task_rank_all':-1,'chosen_next':1,
         'counterpart_distance':-1,'two_task_time':-1,'two_task_return_time':-1}

def extract(e, counterpart, f):
    post=e['post_observation']; cands=[b for b in e['post_task_branches'] if b['spec']['kind']=='task_return']
    assert {b['spec']['task_id'] for b in cands}=={q['id'] for q in post['queue']}
    c=next(b for b in cands if b['spec']['task_id']==counterpart)
    near(c['start_time'],post['time'])
    ordered=sorted(cands,key=lambda b:(b['prediction']['predicted_task_time'],b['spec']['task_id']))
    safe_ordered=[b for b in ordered if b['safe'] is True]
    def rank(seq): return next((i+1 for i,b in enumerate(seq) if b['spec']['task_id']==counterpart),None)
    actual_rank=rank(sorted(cands,key=lambda b:(b['task_time'],b['spec']['task_id']))) if all(b['task_cost_complete'] for b in cands) else None
    expected=safe_ordered[0]['spec']['task_id'] if safe_ordered else None
    action=e['first_downstream_action']
    if expected is not None: assert action['kind']=='serve' and action['task_id']==expected
    complete=c['task_cost_complete']; successful=c['safe'] is True
    assert bool(c['task_completed'])==bool(complete)
    if successful: assert c['reached_charger'] and complete and c['actual_reserve'] is not None
    task_t=c['task_time'] if complete else None
    full_t=c['duration'] if successful else None
    target=next(q for q in post['queue'] if q['id']==counterpart)
    arrivals=[x for x in e['first_task_events'] if x['event']=='arrival']
    return dict(first_task=e['task_id'],counterpart_task=counterpart,safe=c['safe'],outcome=c['outcome'],failure_phase=c['failure_phase'],task_completed=complete,task_time=task_t,task_energy=c['task_energy'] if complete else None,task_return_time=full_t,return_reserve=c['actual_reserve'] if successful else None,return_time=c['return_time'] if successful else None,return_energy=c['return_energy'] if successful else None,observed_duration=c['duration'],observed_energy=c['consumed_energy'],mission_cost_censored=not successful,predicted_task_time=c['prediction']['predicted_task_time'],predicted_sjf_rank_all=rank(ordered),predicted_sjf_rank_safe=rank(safe_ordered) if successful else None,actual_task_rank_all=actual_rank,chosen_next=action['kind']=='serve' and action['task_id']==counterpart,next_action=action,post_time=post['time'],post_battery=post['battery'],post_position=post['position'],counterpart_position=target['position'],counterpart_distance=math.dist(post['position'],target['position']),post_queue_ids=sorted(q['id'] for q in post['queue']),residual_other_tasks={str(q['id']):q['position'] for q in post['queue'] if q['id']!=counterpart},accepted_arrival_ids=[x['task_id'] for x in arrivals if x['accepted']],arrival_ids=[x['task_id'] for x in arrivals],N_W=f['N_W'],first_time=f['task_time'],first_energy=f['task_energy'],two_task_time=f['task_time']+task_t if task_t is not None else None,two_task_return_time=f['task_time']+full_t if full_t is not None else None,source_file=e['source_file'],source_sha256=e['source_sha256'],candidate=c)

def describe(ps):
    return dict(pairs=len(ps),roots=len({p['root_id'] for p in ps}),divergent_pairs=sum(p['delta_N']!=0 for p in ps),divergent_roots=len({p['root_id'] for p in ps if p['delta_N']!=0}),mean_abs_delta_N=avg([abs(p['delta_N']) for p in ps]))

def assess(ps,field,direction):
    counts=Counter();byroot=defaultdict(list);goodroots=set();badroots=set();diffs=[]
    for p in ps:
        x,y=p['i'][field],p['j'][field]; ds=sign(p['delta_N'])
        if x is None or y is None:counts['missing_divergent' if ds else 'missing_N_tie']+=1;continue
        fs=sign((x-y)*direction)
        if ds==0:
            counts['N_tie_feature_tie' if fs==0 else 'N_tie_feature_orders']+=1;continue
        diffs.append((x-y)*ds)
        if fs==0:counts['divergent_feature_tie']+=1;continue
        correct=fs==ds;counts['concordant' if correct else 'opposed']+=1
        byroot[p['root_id']].append(int(correct));(goodroots if correct else badroots).add(p['root_id'])
    for k in ['missing_divergent','missing_N_tie','N_tie_feature_tie','N_tie_feature_orders','divergent_feature_tie','concordant','opposed']:counts[k]+=0
    n=counts['concordant']+counts['opposed']; assert sum(counts.values())==len(ps)
    counts['missing']=counts['missing_divergent']+counts['missing_N_tie']
    return dict(counts,ordered_divergent_pairs=n,concordance=counts['concordant']/n if n else None,root_balanced_concordance=avg([avg(xs) for xs in byroot.values()]),roots_with_ordered_divergent_pairs=len(byroot),roots_with_concordance=len(goodroots),roots_with_opposition=len(badroots),higher_N_minus_lower_feature_mean=avg(diffs))

def main():
    integ=load(SRC/'integrity.json')
    for n in ['pairs.jsonl','branch_features.jsonl','post_task_evidence.jsonl']:
        assert sha(SRC/n)==integ['output_hashes'][n]
    selected=[p for p in rows(SRC/'pairs.jsonl') if p['joint_matched'] and p['same_arrival_ids']]
    assert len(selected)==77
    evidence={(e['root_id'],e['task_id']):e for e in rows(SRC/'post_task_evidence.jsonl')}
    features={(f['root_id'],f['task_id']):f for f in rows(SRC/'branch_features.jsonl')}
    result=[]
    for p in selected:
        rid=p['root_id']; ti,tj=p['task_i'],p['task_j']
        x=extract(evidence[rid,ti],tj,features[rid,ti]); y=extract(evidence[rid,tj],ti,features[rid,tj])
        assert x['N_W']==p['N_i'] and y['N_W']==p['N_j']
        result.append(dict(root_id=rid,policy=p['policy'],regime=p['regime'],seed=p['seed'],battery_multiple=p['battery_multiple'],task_i=ti,task_j=tj,delta_N=p['N_i']-p['N_j'],time_match_difference=p['relative_time_difference'],battery_match_difference=p['battery_fraction_difference'],same_arrival_ids=x['arrival_ids']==y['arrival_ids'],same_accepted_arrival_ids=x['accepted_arrival_ids']==y['accepted_arrival_ids'],same_other_tasks=x['residual_other_tasks']==y['residual_other_tasks'],both_safe=x['safe'] is True and y['safe'] is True,both_next=x['chosen_next'] and y['chosen_next'],i=x,j=y))
    groups={'all':result,'both_safe':[p for p in result if p['both_safe']],'both_next':[p for p in result if p['both_next']]}
    for s in ['B4_0.75','B5']:groups[s]=[p for p in result if p['policy']==s]
    for bm in [2,4,6]:groups['B='+str(bm)]=[p for p in result if p['battery_multiple']==bm]
    summary={name:dict(population=describe(ps),rules={k:assess(ps,k,d) for k,d in RULES.items()}) for name,ps in groups.items()}
    summary['queue_equivalence']={k:sum(p[k] for p in result) for k in ['same_arrival_ids','same_accepted_arrival_ids','same_other_tasks']}
    summary['safety_pairs']=dict(Counter(f"{p['i']['safe']}/{p['j']['safe']}" for p in result))
    summary['next_action_pairs']=dict(Counter(f"{p['i']['chosen_next']}/{p['j']['chosen_next']}" for p in result))
    summary['sampling']=dict(source_runs=len({(p['policy'],p['regime'],p['seed']) for p in result}),seeds=sorted({p['seed'] for p in result}),unique_counterpart_queries=len({(p['root_id'],p[side]['first_task'],p[side]['counterpart_task']) for p in result for side in ['i','j']}))
    summary['candidate_outcomes']=dict(Counter(p[side]['outcome'] for p in result for side in ['i','j']))
    summary['safety_subsets']={k:describe(ps) for k,ps in {'both_safe':[p for p in result if p['both_safe']],'one_safe':[p for p in result if p['i']['safe']!=p['j']['safe']],'neither_safe':[p for p in result if p['i']['safe'] is False and p['j']['safe'] is False]}.items()}
    summary['cross_leg_asymmetry']={}
    for key in ['task_time','task_energy','task_return_time','return_time','counterpart_distance','two_task_time']:
        ds=[abs(p['i'][key]-p['j'][key]) for p in result if p['i'][key] is not None and p['j'][key] is not None]
        summary['cross_leg_asymmetry'][key]=dict(available_pairs=len(ds),mean_abs_difference=avg(ds),median_abs_difference=statistics.median(ds) if ds else None,max_abs_difference=max(ds,default=None))
    jsonl('counterpart_pairs.jsonl',result);dump('summary.json',summary)
    dump('integrity.json',dict(passed=True,simulation_calls=0,input_hashes={n:sha(SRC/n) for n in ['integrity.json','pairs.jsonl','branch_features.jsonl','post_task_evidence.jsonl']},analysis_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'PLAN.md'),output_hashes={n:sha(OUT/n) for n in ['counterpart_pairs.jsonl','summary.json']},python=sys.version,checks=['input hash matches','retained counterpart present in post queue and candidate library','candidate start time equals first post observation time','candidate coverage equals post queue','archived next serve agrees with frozen predicted SJF over safe tasks','completed versus censored cost labels','N values match prior pair table'],queue_equivalence=summary['queue_equivalence']))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['B4_0.75','B5','B=2','B=4','B=6']},indent=2))

if __name__=='__main__': main()
