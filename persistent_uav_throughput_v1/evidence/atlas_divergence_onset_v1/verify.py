from pathlib import Path
import json,hashlib
P=Path(__file__).resolve().parent
load=lambda n:json.loads((P/n).read_text())
audits=load('pair_audits.json');tls=load('timelines.json');traces=load('actual_event_traces.json');integ=load('integrity.json')
checks=0
for n,h in integ['output_hashes'].items():assert hashlib.sha256((P/n).read_bytes()).hexdigest()==h
for a,tl in zip(audits,tls):
 assert (a['root_id'],a['task_i'],a['task_j'])==(tl['root_id'],tl['task_i'],tl['task_j'])
 rows=tl['rows'];final=a['final_delta']
 for side in ['i','j']:
  es=traces[a['source_'+side]];done=[e for e in es if e['event']=='task_completed']
  for r in rows:
   assert r['N_'+side]==sum(round(e['time']*20)<=r['tick'] for e in done);checks+=1
  for key in ['persistent_lead','final_gap_settled']:
   n=a[key]['tick'];before=a[key]['before'][0 if side=='i' else 1]
   assert before['downstream_decisions_selected']==sum(e['event']=='decision' and round(e['time']*20)<n for e in es)-int(a['origin_tick']<n)
 k=next(i for i,r in enumerate(rows) if r['tick']==a['persistent_lead']['tick'])
 assert all(r['delta']*final>0 for r in rows[k:]) and (k==0 or rows[k-1]['delta']*final<=0)
 k=next(i for i,r in enumerate(rows) if r['tick']==a['final_gap_settled']['tick'])
 assert all(r['delta']==final for r in rows[k:]) and (k==0 or rows[k-1]['delta']!=final)
 first=next(r for r in rows if r['delta']);assert first['tick']==a['first_count_difference']['tick']
assert len(audits)==41 and len({a['root_id'] for a in audits})==29
report={'passed':True,'pairs':41,'roots':29,'independent_count_checks':checks,'checks':['completion counts independently recomputed at every merged event batch','persistent winner lead suffix and earliest boundary','final exact gap suffix and earliest boundary','first nonzero count timestamp','strictly pre-onset downstream decision counts independently recomputed','all generated data SHA256'],'simulation_calls':0,'verification_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'viewer_sha256':hashlib.sha256((P/'timeline_viewer.html').read_bytes()).hexdigest()}
(P/'validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
