"""Independent extraction checks and orientation/censoring audit."""
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import analyze

p=Path(__file__).resolve().parent; src=p.parent/'atlas_matched_pairs_v1'
load=lambda f:json.loads(f.read_text())
lines=lambda f:[json.loads(s) for s in f.read_text().splitlines()]
es={(e['root_id'],e['task_id']):e for e in lines(src/'post_task_evidence.jsonl')}
ps=lines(p/'counterpart_pairs.jsonl'); count=Counter(); hashes=load(p/'integrity.json')['output_hashes']
for n,h in hashes.items(): assert hashlib.sha256((p/n).read_bytes()).hexdigest()==h
for pair in ps:
    assert pair['same_other_tasks'] and pair['same_arrival_ids'] and pair['same_accepted_arrival_ids']
    for side,first,other in [('i',pair['task_i'],pair['task_j']),('j',pair['task_j'],pair['task_i'])]:
        e=es[pair['root_id'],first]; row=pair[side]
        raw=[b for b in e['post_task_branches'] if b['spec']['kind']=='task_return' and b['spec']['task_id']==other]
        assert len(raw)==1 and raw[0]==row['candidate']
        c=raw[0]
        assert row['safe']==c['safe']
        if not c['safe']:assert row['task_return_time'] is None and row['return_reserve'] is None
        if not c['task_completed']:assert row['task_time'] is None and row['task_energy'] is None
        if c['task_completed']:assert abs(c['task_completion']['time']-c['start_time']-row['task_time'])<1e-7
        count['queries']+=1
    if pair['delta_N']:
        x,y=pair['i']['task_time'],pair['j']['task_time']
        if x is None or y is None: count['task_time_missing_divergent']+=1
        elif abs(x-y)<=1e-7:count['task_time_tie_divergent']+=1
        elif (x<y)==(pair['delta_N']>0):count['task_time_concordant']+=1
        else:count['task_time_opposed']+=1
swapped=copy.deepcopy(ps)
for pair in swapped:pair['i'],pair['j']=pair['j'],pair['i'];pair['delta_N']=-pair['delta_N']
for field,direction in analyze.RULES.items():assert analyze.assess(ps,field,direction)==analyze.assess(swapped,field,direction)
assert count['queries']==154 and count['task_time_concordant']==25 and count['task_time_opposed']==15 and count['task_time_missing_divergent']==1
report={'passed':True,'counts':dict(count),'checks':['every counterpart candidate independently located in prior evidence by reversed task ID','all 77 pairs have identical other tasks and accepted arrival IDs','failed full-mission costs and incomplete task costs stay null','completed task duration equals completion minus start','all 12 direction summaries invariant under swapping pair sides','independent task-time direction recount','output SHA256 verification'],'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'new_simulations':0}
(p/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
