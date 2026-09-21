"""Read-only archived trajectory analysis; standard library only, no simulator."""
from pathlib import Path
import collections
import hashlib
import itertools
import json
import math
import statistics
import sys

OUT = Path(__file__).resolve().parent
SRC = OUT.parent / 'real_state_branching_atlas_v1'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(xs): return statistics.mean(xs) if xs else None
def lines(p): return [json.loads(s) for s in p.read_text().splitlines()]
def write(name, x): (OUT/name).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def jsonl(name, xs): (OUT/name).write_text(''.join(json.dumps(x, ensure_ascii=False, allow_nan=False)+'\n' for x in xs))
def near(a,b): assert abs(a-b)<1e-6, (a,b)

def main():
    integ=read(SRC/'integrity.json')
    for name,key in [('one_step_branches.jsonl','one_step_sha256'),('continuation_branches.jsonl','continuation_sha256'),('summary.json','summary_sha256')]:
        assert sha(SRC/name)==integ[key]
    roots={r['id']:r for r in read(SRC/'roots.json')}
    aa={(r['root_id'],r['spec']['task_id']):r for r in lines(SRC/'one_step_branches.jsonl') if r['spec']['kind']=='task_return'}
    bb=collections.defaultdict(list)
    for b in lines(SRC/'continuation_branches.jsonl'): bb[b['root_id']].append(b)
    eligible={rid:bs for rid,bs in bb.items() if len(bs)>=2 and all(b['N_W'] is not None for b in bs)}
    assert len(eligible)==128 and len(bb)-len(eligible)==17
    raw_paths={str(p.relative_to(p.parent.parent)):p for p in SRC.glob('worker_*/*/B_*.json')}
    features=[]; input_hashes={}; pairs=[]; excerpts=[]
    for rid,bs in sorted(eligible.items()):
        root=roots[rid]; ro=root['observation']; fs=[]
        safe_ids={tid for (rr,tid),a in aa.items() if rr==rid and a['safe'] is True}
        assert safe_ids=={b['task_id'] for b in bs}
        for b in sorted(bs,key=lambda b:b['task_id']):
            a=aa[rid,b['task_id']]; path=raw_paths[b['file']]
            assert sha(path)==b['sha256'];input_hashes[str(path.relative_to(SRC))]=b['sha256']
            raw=read(path); c=raw['first_task_completion']; near(c['time'],a['task_completion']['time']);near(c['battery'],a['post_task_battery'])
            assert c['position']==a['post_task_position']
            events=raw['events']; ci=next(i for i,e in enumerate(events) if e['event']=='task_completed' and e['task_id']==b['task_id'])
            assert events[ci]==c
            arrivals=[e for e in events[:ci+1] if e['event']=='arrival']
            decisions=raw['oracle_decisions']; d=decisions[0] if decisions and abs(decisions[0]['time']-c['time'])<1e-6 else None
            assert d is not None, (rid,b['task_id'],'missing immediate decision')
            post=d['observation'];near(post['battery'],c['battery']);assert post['position']==c['position']
            q=post['queue']; cand=[x for x in d['branches'] if x['spec']['kind']=='task_return']
            assert {x['spec']['task_id'] for x in cand}=={x['id'] for x in q}
            safe_count=sum(x['safe'] is True for x in cand) if all(x['safe'] is not None for x in cand) else None
            arrival_ids=[e['task_id'] for e in arrivals]
            rejected=[e for e in arrivals if not e['accepted']]
            expected_q=({t['id'] for t in ro['queue']}-{b['task_id']}) | {e['task_id'] for e in arrivals if e['accepted']}
            assert expected_q=={t['id'] for t in q}
            root_geom=[math.dist(x['position'],y['position']) for x,y in itertools.combinations(ro['queue'],2)]
            f=dict(root_id=rid,task_id=b['task_id'],policy=root['policy'],regime=root['regime'],seed=root['seed'],battery_multiple=2*(root['regime']//9+1),capacity=ro['capacity'],N_W=b['N_W'],task_time=a['task_time'],task_energy=a['task_energy'],post_battery=c['battery'],post_position=c['position'],charger_distance=math.dist(c['position'],ro['charger_position']),root_queue=ro['queue'],root_queue_mean_pair_distance=mean(root_geom),post_queue=q,post_queue_ids=sorted(expected_q),post_queue_size=len(q),post_queue_mean_distance=mean([math.dist(c['position'],t['position']) for t in q]),arrivals=len(arrivals),arrival_ids=arrival_ids,accepted_arrivals=sum(e['accepted'] for e in arrivals),overflow_first=len(rejected),safe_count=safe_count,first_downstream_action=d['action'],forced_returns=b['forced_return_count'],overflow_window=b['overflow'],flight_time=b['time_by_mode']['flight'],charging_time=b['time_by_mode']['charging'],waiting_time=b['time_by_mode']['waiting'],source_file=b['file'],source_sha256=b['sha256'])
            near(f['task_energy'],ro['battery']-c['battery']);near(f['task_time'],c['time']-root['decision_time'])
            fs.append(f);features.append(f)
            excerpts.append(dict(root_id=rid,task_id=b['task_id'],source_file=b['file'],source_sha256=b['sha256'],first_task_events=events[:ci+1],post_observation=post,first_downstream_action=d['action'],post_task_branches=[{k:v for k,v in x.items() if k!='events'} for x in d['branches']]))
        for i,j in itertools.combinations(fs,2):
            dt=abs(i['task_time']-j['task_time']);db=abs(i['post_battery']-j['post_battery'])/ro['capacity']
            tr=dt/((i['task_time']+j['task_time'])/2)
            p=dict(root_id=rid,policy=root['policy'],regime=root['regime'],seed=root['seed'],battery_multiple=i['battery_multiple'],task_i=i['task_id'],task_j=j['task_id'],N_i=i['N_W'],N_j=j['N_W'],abs_delta_N=abs(i['N_W']-j['N_W']),relative_time_difference=tr,window_time_difference=dt/1308.6,battery_fraction_difference=db,time_matched=tr<.10,battery_matched=db<.05,joint_matched=tr<.10 and db<.05,same_arrival_ids=i['arrival_ids']==j['arrival_ids'],same_post_queue_ids=i['post_queue_ids']==j['post_queue_ids'],post_position_separation=math.dist(i['post_position'],j['post_position']))
            for k in ['task_time','task_energy','post_battery','charger_distance','post_queue_size','post_queue_mean_distance','arrivals','accepted_arrivals','overflow_first','safe_count','forced_returns','overflow_window','flight_time','charging_time','waiting_time']:
                p[k+'_i']=i[k];p[k+'_j']=j[k]
            pairs.append(p)
    def stats(ps):
        by=collections.defaultdict(list)
        for p in ps:by[p['root_id']].append(p['abs_delta_N'])
        return dict(pairs=len(ps),divergent_pairs=sum(p['abs_delta_N']>0 for p in ps),roots=len(by),divergent_roots=sum(any(x>0 for x in xs) for xs in by.values()),mean_abs_delta_N=mean([p['abs_delta_N'] for p in ps]),root_balanced_mean_abs_delta_N=mean([mean(xs) for xs in by.values()]),max_abs_delta_N=max((p['abs_delta_N'] for p in ps),default=None))
    groups={'all':pairs}
    for source in ['B4_0.75','B5']:groups[source]=[p for p in pairs if p['policy']==source]
    for bm in [2,4,6]:groups['B='+str(bm)]=[p for p in pairs if p['battery_multiple']==bm]
    summary={g:{kind:stats([p for p in ps if kind=='all' or p[kind]]) for kind in ['all','time_matched','battery_matched','joint_matched']} for g,ps in groups.items()}
    joint=[p for p in pairs if p['joint_matched']]; divergent=[p for p in joint if p['abs_delta_N']>0]
    descriptors={}
    for k in ['charger_distance','post_queue_mean_distance','post_queue_size','arrivals','overflow_first','safe_count','forced_returns','overflow_window','charging_time','flight_time']:
        ds=[(p[k+'_i']-p[k+'_j'])*(1 if p['N_i']>p['N_j'] else -1) for p in divergent if p[k+'_i'] is not None and p[k+'_j'] is not None]
        descriptors[k]=dict(available_pairs=len(ds),higher_N_has_lower=sum(d< -1e-8 for d in ds),equal=sum(abs(d)<=1e-8 for d in ds),higher_N_has_higher=sum(d>1e-8 for d in ds),mean_higher_minus_lower=mean(ds))
    summary['joint_same_arrival_ids']=stats([p for p in joint if p['same_arrival_ids']])
    summary['joint_equal_safe_count']=stats([p for p in joint if p['safe_count_i'] is not None and p['safe_count_i']==p['safe_count_j']])
    summary['joint_equal_arrival_and_safe_count']=stats([p for p in joint if p['same_arrival_ids'] and p['safe_count_i'] is not None and p['safe_count_i']==p['safe_count_j']])
    summary['joint_divergent_descriptors']=descriptors
    summary['joint_divergent_root_ids']=sorted({p['root_id'] for p in divergent})
    summary['sampling'] = dict(complete_roots=len(eligible),complete_source_runs=len({(roots[r]['policy'],roots[r]['regime'],roots[r]['seed']) for r in eligible}),complete_seeds=sorted({roots[r]['seed'] for r in eligible}),joint_roots=len({p['root_id'] for p in joint}),joint_divergent_seeds=sorted({p['seed'] for p in divergent}),joint_same_post_queue_pairs=sum(p['same_post_queue_ids'] for p in joint))
    jsonl('branch_features.jsonl',features);jsonl('pairs.jsonl',pairs);jsonl('post_task_evidence.jsonl',excerpts);write('summary.json',summary)
    write('integrity.json',dict(passed=True,complete_roots=128,unresolved_roots_excluded=17,features=len(features),pairs=len(pairs),raw_files_verified=len(input_hashes),input_raw_hashes=input_hashes,input_summary_hashes={n:sha(SRC/n) for n in ['manifest.json','roots.json','one_step_branches.jsonl','continuation_branches.jsonl','summary.json']},output_hashes={n:sha(OUT/n) for n in ['branch_features.jsonl','pairs.jsonl','post_task_evidence.jsonl','summary.json']},analysis_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'PLAN.md'),python=sys.version,simulation_calls=0))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
