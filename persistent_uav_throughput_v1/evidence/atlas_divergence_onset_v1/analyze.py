"""Actual archived event alignment. No simulator dependencies."""
from pathlib import Path
from collections import defaultdict,Counter
import json,hashlib,math,statistics,sys
OUT=Path(__file__).resolve().parent
PRE=OUT.parent/'atlas_retained_counterpart_v1'
ATLAS=OUT.parent/'real_state_branching_atlas_v1'
def load(p):return json.loads(p.read_text())
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(n,x):(OUT/n).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def tick(t):
    k=round(t*20);assert abs(t-k/20)<1e-6,t
    return k

def identity(e):return (e['action']['kind'],e['action'].get('task_id'))
def newstate(root):return {'queue':set(q['id'] for q in root['queue']),'completed':set(),'overflow':0,'downstream_decisions':0,'last_decision':{'time':root['time'],'observation':root}}
def context(s,k):
    d=s['last_decision']
    return {'queue_ids':sorted(s['queue']),'completed_since_root':sorted(s['completed']),'overflow_since_root':s['overflow'],'downstream_decisions_selected':s['downstream_decisions'],'last_predecision_observation':d['observation'] if d else None,'observation_age_seconds':(k-tick(d['time']))/20 if d else None}
def advance(s,e,is_root):
    typ=e['event']
    if typ=='decision':
        assert s['queue']=={q['id'] for q in e['observation']['queue']},(s['queue'],e)
        s['last_decision']=e
        if not is_root:s['downstream_decisions']+=1
        if e['action']['kind']=='serve':s['queue'].remove(e['action']['task_id'])
    elif typ=='arrival':
        if e['accepted']:s['queue'].add(e['task_id'])
        else:s['overflow']+=1
    elif typ=='task_completed':
        assert e['task_id'] not in s['completed'];s['completed'].add(e['task_id'])

def timings(events,kind):
    if kind=='recharge_request':return [tick(e['time']) for e in events if e['event']=='decision' and e['action']['kind']=='recharge']
    return [tick(e['time']) for e in events if e['event']==kind]
def first_timing_difference(xs,ys,origin):
    for n in range(max(len(xs),len(ys))):
        x=xs[n] if n<len(xs) else None;y=ys[n] if n<len(ys) else None
        if x!=y:return dict(ordinal=n+1,i_elapsed=(x-origin)/20 if x is not None else None,j_elapsed=(y-origin)/20 if y is not None else None,unmatched=x is None or y is None)
    return None

def summarize(ps):
    def distribution(key):return dict(sorted(Counter(str(p[key]) for p in ps).items()))
    return dict(pairs=len(ps),roots=len({p['root_id'] for p in ps}),first_action_difference_ordinal=distribution('first_action_difference_ordinal'),persistent_lead_decisions_max=distribution('persistent_lead_decisions_max'),final_gap_settled_decisions_max=distribution('final_gap_settled_decisions_max'),persistent_lead_elapsed_median=statistics.median([p['persistent_lead']['elapsed'] for p in ps]) if ps else None,final_gap_settled_elapsed_median=statistics.median([p['final_gap_settled']['elapsed'] for p in ps]) if ps else None,ever_same_completed_set_after_both_first=sum(p['same_completed_set_after_both_first'] is not None for p in ps),persistent_after_any_recharge=sum(p['persistent_after_any_recharge'] for p in ps),persistent_after_any_charge_complete=sum(p['persistent_after_any_charge_complete'] for p in ps),persistent_after_overflow_acceptance_difference=sum(p['persistent_after_acceptance_difference'] for p in ps))

def main():
    pairs=[p for p in rows(PRE/'counterpart_pairs.jsonl') if p['delta_N']]
    assert len(pairs)==41
    assert sha(PRE/'counterpart_pairs.jsonl')==load(PRE/'integrity.json')['output_hashes']['counterpart_pairs.jsonl']
    rawpaths={str(p.relative_to(p.parent.parent)):p for p in ATLAS.glob('worker_*/*/B_*.json')}
    summaries=[];timelines=[];trace_store={};inputs={}
    for p in pairs:
        branches=[]
        for side in ['i','j']:
            meta=p[side];path=rawpaths[meta['source_file']];assert sha(path)==meta['source_sha256']
            inputs[str(path.relative_to(ATLAS))]=meta['source_sha256'];r=load(path)
            assert r['N_W']==meta['N_W'] and r['outcome']=='window_end'
            events=r['events'];assert events[0]['event']=='decision'
            trace_store[meta['source_file']]=events
            branches.append(r)
        a,b=branches;origin=tick(a['window_origin']);end=tick(a['window_end']);assert (origin,end)==(tick(b['window_origin']),tick(b['window_end']))
        events_a,events_b=a['events'],b['events']; buckets=[defaultdict(list),defaultdict(list)]
        for ix,es in enumerate([events_a,events_b]):
            for n,e in enumerate(es):
                k=tick(e['time']);assert origin<=k<=end
                buckets[ix][k].append((n,e))
        states=[newstate(events_a[0]['observation']),newstate(events_b[0]['observation'])]
        timeline=[];first_done=max(tick(a['first_task_completion']['time']),tick(b['first_task_completion']['time']))
        for k in sorted({origin,end}|set(buckets[0])|set(buckets[1])):
            before=[context(s,k) for s in states]
            for ix in [0,1]:
                for n,e in buckets[ix][k]:advance(states[ix],e,n==0)
            ca,cb=map(lambda s:len(s['completed']),states)
            other=[s['queue']-{p['task_i'],p['task_j']} for s in states]
            timeline.append(dict(tick=k,elapsed=(k-origin)/20,N_i=ca,N_j=cb,delta=ca-cb,queue_i=sorted(states[0]['queue']),queue_j=sorted(states[1]['queue']),same_queue=states[0]['queue']==states[1]['queue'],same_other_queue=other[0]==other[1],same_completed=states[0]['completed']==states[1]['completed'],completed_i=sorted(states[0]['completed']),completed_j=sorted(states[1]['completed']),overflow_i=states[0]['overflow'],overflow_j=states[1]['overflow'],decisions_i=states[0]['downstream_decisions'],decisions_j=states[1]['downstream_decisions'],events_i=[e for n,e in buckets[0][k]],events_j=[e for n,e in buckets[1][k]],before=before))
        assert timeline[-1]['delta']==p['delta_N'] and (timeline[-1]['N_i'],timeline[-1]['N_j'])==(a['N_W'],b['N_W'])
        for ix,r in enumerate(branches): assert states[ix]['overflow']==r['overflow']
        def first(pred):return next((t for t in timeline if pred(t)),None)
        # Earliest suffix with persistent final sign, then earliest suffix with exact final gap.
        final=p['delta_N'];last_nonlead=max((n for n,t in enumerate(timeline) if t['delta']*final<=0),default=-1)
        last_different=max((n for n,t in enumerate(timeline) if t['delta']!=final),default=-1)
        persistent=timeline[last_nonlead+1];settled=timeline[last_different+1]
        def pack(t):
            if t is None:return None
            return {k:t[k] for k in ['tick','elapsed','N_i','N_j','delta','completed_i','completed_j','decisions_i','decisions_j','events_i','events_j','before','queue_i','queue_j','overflow_i','overflow_j']} | {'same_completed_before':t['before'][0]['completed_since_root']==t['before'][1]['completed_since_root'],'same_completed_before_nonempty':bool(t['before'][0]['completed_since_root']) and t['before'][0]['completed_since_root']==t['before'][1]['completed_since_root']}
        decisions=[[e for e in es if e['event']=='decision'][1:] for es in [events_a,events_b]]
        mismatch=None
        for n in range(max(map(len,decisions))):
            x=decisions[0][n] if n<len(decisions[0]) else None;y=decisions[1][n] if n<len(decisions[1]) else None
            if x is None or y is None or identity(x)!=identity(y):
                mismatch=dict(ordinal=n+1,i=x,j=y,unmatched=x is None or y is None);break
        timing={key:first_timing_difference(timings(events_a,key),timings(events_b,key),origin) for key in ['recharge_request','charger_arrival','charge_complete']}
        charge_cycles=[];unfinished_charges=[]
        for es in [events_a,events_b]:
            start=None;cycles=[]
            for e in es:
                if e['event']=='charger_arrival':start=tick(e['time'])
                if e['event']=='charge_complete':
                    assert start is not None
                    cycles.append(dict(start=start,end=tick(e['time']),seconds=(tick(e['time'])-start)/20));start=None
            charge_cycles.append(cycles);unfinished_charges.append((start-origin)/20 if start is not None else None)
        duration_diff=None
        for n in range(min(map(len,charge_cycles))):
            if charge_cycles[0][n]['seconds']!=charge_cycles[1][n]['seconds']:
                duration_diff=dict(ordinal=n+1,i=charge_cycles[0][n],j=charge_cycles[1][n]);break
        arrivals=[]
        for es in [events_a,events_b]:arrivals.append({e['task_id']:e for e in es if e['event']=='arrival'})
        assert set(arrivals[0])==set(arrivals[1])
        for tid in arrivals[0]:assert tick(arrivals[0][tid]['time'])==tick(arrivals[1][tid]['time'])
        acceptance_diff=next((dict(task_id=tid,tick=tick(e['time']),elapsed=(tick(e['time'])-origin)/20,i=e,j=arrivals[1][tid]) for tid,e in sorted(arrivals[0].items(),key=lambda x:tick(x[1]['time'])) if e['accepted']!=arrivals[1][tid]['accepted']),None)
        recharge_times=timings(events_a,'recharge_request')+timings(events_b,'recharge_request');charge_done=timings(events_a,'charge_complete')+timings(events_b,'charge_complete')
        q=dict(root_id=p['root_id'],policy=p['policy'],regime=p['regime'],seed=p['seed'],battery_multiple=p['battery_multiple'],task_i=p['task_i'],task_j=p['task_j'],source_i=p['i']['source_file'],source_j=p['j']['source_file'],origin_tick=origin,end_tick=end,final_delta=final,both_first_complete_elapsed=(first_done-origin)/20,first_action_difference=mismatch,first_action_difference_ordinal=mismatch['ordinal'] if mismatch else None,first_downstream_decision_time_difference=first_timing_difference([tick(e['time']) for e in decisions[0]],[tick(e['time']) for e in decisions[1]],origin),recharge_timing_differences=timing,first_charge_duration_difference=duration_diff,charge_cycles=charge_cycles,unfinished_charge_start_elapsed=unfinished_charges,charge_cycle_completed_counts=[len(x) for x in charge_cycles],first_queue_difference=pack(first(lambda t:not t['same_queue'])),first_other_queue_difference=pack(first(lambda t:not t['same_other_queue'])),first_other_queue_difference_after_both_first=pack(first(lambda t:t['tick']>=first_done and not t['same_other_queue'])),first_count_difference=pack(first(lambda t:t['delta']!=0)),first_count_difference_after_both_first=pack(first(lambda t:t['tick']>=first_done and t['delta']!=0)),same_completed_set_after_both_first=pack(first(lambda t:t['tick']>=first_done and t['same_completed'] and t['N_i']>0)),first_acceptance_difference=acceptance_diff,first_overflow_count_difference=pack(first(lambda t:t['overflow_i']!=t['overflow_j'])),persistent_lead=pack(persistent),final_gap_settled=pack(settled),persistent_lead_decisions_max=max(x['downstream_decisions_selected'] for x in persistent['before']),final_gap_settled_decisions_max=max(x['downstream_decisions_selected'] for x in settled['before']),persistent_lead_decisions_through_batch_max=max(persistent['decisions_i'],persistent['decisions_j']),persistent_after_any_recharge=any(k<persistent['tick'] for k in recharge_times),persistent_after_any_charge_complete=any(k<persistent['tick'] for k in charge_done),persistent_after_acceptance_difference=acceptance_diff is not None and acceptance_diff['tick']<persistent['tick'])
        q['count_tie_reentries']=sum(x['delta']!=0 and y['delta']==0 for x,y in zip(timeline,timeline[1:]))
        signs=[1 if t['delta']>0 else -1 for t in timeline if t['delta']]
        q['nonzero_leader_reversals']=sum(x!=y for x,y in zip(signs,signs[1:]))
        q['first_count_leader_is_final_winner']=q['first_count_difference']['delta']*final>0
        summaries.append(q)
        # Keep state contexts in summaries; compact aligned table for viewer.
        timelines.append(dict(root_id=p['root_id'],task_i=p['task_i'],task_j=p['task_j'],origin_tick=origin,end_tick=end,rows=[{k:v for k,v in t.items() if k not in ['before','events_i','events_j']} for t in timeline]))
    dump('pair_audits.json',summaries);dump('timelines.json',timelines);dump('actual_event_traces.json',trace_store)
    grouped={'all':summaries}
    for src in ['B4_0.75','B5']:grouped[src]=[p for p in summaries if p['policy']==src]
    for bm in [2,4,6]:grouped['B='+str(bm)]=[p for p in summaries if p['battery_multiple']==bm]
    summary={k:summarize(ps) for k,ps in grouped.items()}
    summary['all']['pairs_with_count_tie_reentry']=sum(p['count_tie_reentries']>0 for p in summaries)
    summary['all']['median_count_tie_reentries']=statistics.median(p['count_tie_reentries'] for p in summaries)
    summary['all']['pairs_with_leader_reversal']=sum(p['nonzero_leader_reversals']>0 for p in summaries)
    summary['all']['first_count_leader_is_final_winner']=sum(p['first_count_leader_is_final_winner'] for p in summaries)
    summary['all']['raw_first_count_is_first_task_phase']=sum(x['first_count_difference']['tick']<origin2 for x,origin2 in [(x,x['origin_tick']+round(x['both_first_complete_elapsed']*20)) for x in summaries])
    summary['all']['first_recharge_request_difference_ordinals']=dict(Counter(str(p['recharge_timing_differences']['recharge_request']['ordinal']) if p['recharge_timing_differences']['recharge_request'] else 'none' for p in summaries))
    summary['all']['charge_duration_difference_ordinals']=dict(Counter(str(p['first_charge_duration_difference']['ordinal']) if p['first_charge_duration_difference'] else 'none_among_completed' for p in summaries))
    summary['all']['persistent_same_completed_before_nonempty']=sum(p['persistent_lead']['same_completed_before_nonempty'] for p in summaries)
    summary['all']['first_other_queue_difference_elapsed_median']=statistics.median(p['first_other_queue_difference']['elapsed'] for p in summaries if p['first_other_queue_difference'])
    summary['all']['first_acceptance_difference_pairs']=sum(p['first_acceptance_difference'] is not None for p in summaries)
    summary['all']['first_action_difference_is_unmatched_tail']=sum(p['first_action_difference']['unmatched'] for p in summaries if p['first_action_difference'])
    dump('summary.json',summary)
    dump('integrity.json',dict(passed=True,pairs=len(summaries),raw_files_verified=len(inputs),raw_hashes=inputs,input_pair_sha256=sha(PRE/'counterpart_pairs.jsonl'),analysis_sha256=sha(Path(__file__)),plan_sha256=sha(OUT/'PLAN.md'),output_hashes={n:sha(OUT/n) for n in ['pair_audits.json','timelines.json','actual_event_traces.json','summary.json']},simulation_calls=0,python=sys.version))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
