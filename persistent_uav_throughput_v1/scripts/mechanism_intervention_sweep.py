"""Frozen complete-only mechanism campaign; never reads outcomes to select jobs."""
import argparse,gzip,itertools,json,os,pickle,signal,subprocess,sys,time,traceback
from pathlib import Path
from collections import Counter,defaultdict
import mechanism_sweep_engine as engine
import infinite_queue_intervention as iq
atlas=engine.atlas;census=atlas.census
PROJECT=atlas.PROJECT;SCRIPT=Path(__file__).resolve();PLAN=PROJECT/'MECHANISM_INTERVENTION_SWEEP_V1.md'
DEFAULT=PROJECT/'evidence/mechanism_intervention_sweep_v1';ORIGINAL=iq.ORIGINAL
FEATURES=PROJECT/'evidence/atlas_matched_pairs_v1/branch_features.jsonl'
read=iq.read;write=iq.write;sha=iq.sha
rows=lambda p:list(map(json.loads,Path(p).read_text().splitlines()))

def prepare(output):
    assert not (output/'manifest.json').exists()
    old,roots=atlas.verify(ORIGINAL);roots={r['id']:r for r in roots}
    iq.verify(iq.DEFAULT)
    assert sha(FEATURES)==read(FEATURES.parent/'integrity.json')['output_hashes'][FEATURES.name]
    assert sha(iq.PAIRS)==read(iq.PAIRS.parent/'integrity.json')['output_hashes'][iq.PAIRS.name]
    features=rows(FEATURES);pairs=rows(iq.PAIRS)
    assert len(features)==420 and len(pairs)==77
    primary={(p['root_id'],p['task_'+s]) for p in pairs for s in ('i','j')}
    assert len(primary)==130 and len({x[0] for x in primary})==50
    arows={(r['root_id'],r['spec']['task_id']):r for r in rows(ORIGINAL/'one_step_branches.jsonl') if r['spec']['kind']=='task_return'}
    snaps={p.parent.parent.name:p.parent for p in ORIGINAL.glob('worker_*/*/root/resume.json')}
    bases={}
    for f in features:
        rid,tid=f['root_id'],f['task_id'];key=f'{rid}_task{tid}';folder=snaps[rid];ref=read(folder/'resume.json')
        assert sha(folder/ref['file'])==ref['sha256']==read(folder.parent/'root_verified.json')['root_snapshot']['sha256']
        bases[key]=dict(id=key,root_id=rid,task_id=tid,policy=f['policy'],regime=f['regime'],root=roots[rid],root_folder=str(folder),root_ref=ref,first=arows[rid,tid],baseline_N=f['N_W'],baseline_file=f['source_file'],baseline_sha256=f['source_sha256'],primary=(rid,tid) in primary)
    jobs=[];previous={j['id']:j for j in read(iq.DEFAULT/'jobs.json')}
    for key,b in sorted(bases.items()):
        conditions=engine.CONDITIONS if b['primary'] else ('C1','C4')
        for cond in conditions:
            if cond=='E':continue
            jobs.append(dict(id=cond+'__'+key,base=key,condition=cond,import_id=key if cond=='C1' and key in previous else None))
    pairmap=[]
    for n,p in enumerate(pairs):
        pid=f'p{n:03d}';item=dict(id=pid,root_id=p['root_id'],policy=p['policy'],regime=p['regime'],seed=p['seed'],old_delta=p['delta_N'])
        for s,other in [('i','j'),('j','i')]:
            key=f"{p['root_id']}_task{p['task_'+s]}";item[s]=key
            jobs.append(dict(id='E__'+pid+'_'+s,base=key,condition='E',erase=p['task_'+other],pair_id=pid,side=s,import_id=None))
        pairmap.append(item)
    jobs.sort(key=lambda j:j['id'])
    assert len(jobs)==2164 and sum(j['import_id'] is not None for j in jobs)==70
    for key,j in previous.items():
        b=bases[key]
        for field in ('root_ref','first','baseline_sha256','task_id','root_id'):assert b[field]==j[field]
    output.mkdir(parents=True,exist_ok=True)
    for name,data in [('bases.json',bases),('jobs.json',jobs),('pairs.json',pairmap)]:write(output/name,data)
    deps=[SCRIPT,Path(engine.__file__),PLAN,FEATURES,iq.PAIRS,iq.SCRIPT,iq.PLAN,iq.DEFAULT/'manifest.json',ORIGINAL/'manifest.json',ORIGINAL/'one_step_branches.jsonl',ORIGINAL/'continuation_branches.jsonl']
    write(output/'manifest.json',dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=atlas.ROOT,text=True).strip(),dependencies={str(p):sha(p) for p in deps},files={n:sha(output/n) for n in ['bases.json','jobs.json','pairs.json']},workers=16,total_jobs=len(jobs),imported_jobs=70,new_jobs=2094,primary_pairs=77,primary_branches=130,broad_roots=128,broad_branches=420,windows=[436.2,872.4,1308.6],conditions=list(engine.CONDITIONS),created_unix=time.time(),outcome_adaptive_selection=False,training_updates=0))
    print('Frozen 2164 jobs: 70 prior C1 imports, 2094 new',flush=True)

def verify(output):
    m=read(output/'manifest.json')
    for p,h in m['dependencies'].items():assert sha(Path(p))==h,p
    for p,h in m['files'].items():assert sha(output/p)==h,p
    old,_=atlas.verify(ORIGINAL)
    return m,read(output/'jobs.json'),read(output/'bases.json'),atlas.EstimateModel(**read(old['frozen'])['model'])

def rawread(p):
    with (gzip.open(p,'rt') if str(p).endswith('.gz') else Path(p).open()) as f:return json.load(f)

def rawwrite(path,r):
    temp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp,'wt',compresslevel=3) as f:json.dump(r,f,allow_nan=False)
    os.replace(temp,path)

def work(c):return c['env'].nav.policy_steps-c['base_steps']+c['oracle_physics'] if c else 0

def begin(job,bases):
    root=iq.root_for(bases[job['base']]);before=pickle.dumps(root)
    c=engine.start(root,bases[job['base']]['first'],job['condition'],job.get('erase'))
    assert pickle.dumps(root)==before;root['env'].nav.close();return c

def validate(r,job,base):
    assert r['task_id']==base['task_id'] and r['window_seconds']==atlas.WINDOW
    assert r.get('original_stream_sha256',r['stream_sha256'])==base['root']['stream_sha256']
    if job['condition'] in ('C1','C6'):iq.assert_admission(r)
    if job['condition'] in ('C3','C7'):assert not any(e['event']=='arrival' for e in r['events'])
    if job['condition'] in ('C4','C7'):
        assert not r['depletion'] and r['recharge_requests']==0
        assert not any(e['event'] in ('charger_arrival','charge_complete') for e in r['events'])
        for d in r['oracle_decisions']:
            assert d['action'] is None or d['action']['kind']=='serve'
            for b in d['branches']:
                assert b['spec']['kind']=='task_return'
                assert not any(e['event'] in ('charger_arrival','charge_complete') for e in b['events'])
    if job['condition']=='E' and r['first_task_completion'] is not None:assert r['erased_tasks']==1

def worker(output,w,resume=False):
    m,jobs,bases,model=verify(output);assigned=[j for j in jobs if j['import_id'] is None][w::m['workers']];folder=output/f'worker_{w:02d}'
    contract=dict(manifest=sha(output/'manifest.json'),worker=w,jobs=[j['id'] for j in assigned])
    with atlas.exclusive_run(folder):
        if resume:
            state=atlas.restore(folder);assert state['contract']==contract;census.set_rng(state['rng'])
        else:
            folder.mkdir(exist_ok=False);write(folder/'manifest.json',contract)
            state=dict(contract=contract,index=0,continuation=None,results=[],finished_steps=0,rng=census.rng_state())
        stop=[False]
        def halt(*args):stop[0]=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        def save(status):
            state['rng']=census.rng_state();ref=atlas.snapshot(folder,state);c=state['continuation']
            write(folder/'status.json',dict(status=status,index=state['index'],planned=len(assigned),job=assigned[state['index']]['id'] if state['index']<len(assigned) else None,stage=c['stage'] if c else None,policy_steps=state['finished_steps']+work(c),snapshot=ref,checkpoint_unix=time.time(),training_updates=0))
        saved=time.monotonic();steps=0
        try:
            while state['index']<len(assigned):
                job=assigned[state['index']]
                if state['continuation'] is None:state['continuation']=begin(job,bases);save('running')
                c=state['continuation'];r=engine.advance(c,model)
                if r is not None:
                    validate(r,job,bases[job['base']]);r.update(job_id=job['id'],root_id=bases[job['base']]['root_id'],policy=bases[job['base']]['policy'],regime=bases[job['base']]['regime'])
                    path=folder/(job['id']+'.json.gz');rawwrite(path,r)
                    state['results'].append(dict(job_id=job['id'],file=str(path.relative_to(output)),sha256=sha(path)))
                    state['finished_steps']+=work(c);c['env'].nav.close();state['continuation']=None;state['index']+=1;save('running')
                total=state['finished_steps']+work(state['continuation'])
                if stop[0] or total-steps>=100 or time.monotonic()-saved>=60:
                    save('paused' if stop[0] else 'running');saved=time.monotonic();steps=total
                    if stop[0]:return
            write(folder/'results.json',state['results']);save('complete')
        except Exception as exc:
            write(folder/'error.json',dict(error=repr(exc),traceback=traceback.format_exc(),job=assigned[state['index']]['id'] if state['index']<len(assigned) else None,time=time.time()));raise

def import_prior(output):
    m,jobs,bases,_=verify(output);proof=read(iq.DEFAULT/'integrity.json');assert proof['passed'] and proof['manifest_sha256']==sha(iq.DEFAULT/'manifest.json')
    for name,h in proof['output_hashes'].items():assert sha(iq.DEFAULT/name)==h
    prior={r['job_id']:r for r in read(iq.DEFAULT/'continuations.json')};assert len(prior)==70
    imported=[]
    for j in jobs:
        if not j['import_id']:continue
        r=prior[j['import_id']];path=iq.DEFAULT/r['file'];assert sha(path)==r['sha256'];raw=read(path);iq.assert_admission(raw)
        assert raw['task_id']==bases[j['base']]['task_id'] and raw['stream_sha256']==bases[j['base']]['root']['stream_sha256']
        imported.append(dict(job_id=j['id'],file=str(path),sha256=r['sha256'],imported=True))
    assert len(imported)==70
    write(output/'imports.json',dict(source_integrity_sha256=sha(iq.DEFAULT/'integrity.json'),results=imported,imported_unix=time.time()))

def launch(output):
    m,_,_,_=verify(output);assert read(output/'engineering_smoke.json')['passed'];assert len(read(output/'imports.json')['results'])==70
    assert not (output/'launch.json').exists();records=[]
    for w in range(16):
        cmd=[str(atlas.WORK/'.venv/bin/python'),'-u',str(SCRIPT),'worker','--output',str(output),'--worker',str(w)]
        with (output/f'worker_{w:02d}.log').open('xb') as log:
            proc=subprocess.Popen(cmd,cwd=PROJECT,env=dict(os.environ,**atlas.THREADS),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(worker=w,pid=proc.pid,command=cmd));write(output/'launch.json',records)

def check(output):
    records=[]
    for rec in read(output/'launch.json'):
        folder=output/f"worker_{rec['worker']:02d}";assert not (folder/'error.json').exists()
        s=read(folder/'status.json');assert s['policy_steps']>0 or s['status']=='complete'
        state=atlas.restore(folder);assert state['contract']['manifest']==sha(output/'manifest.json')
        if s['status']!='complete':
            assert Path(f"/proc/{rec['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')==rec['command']
            assert int(next(x for x in Path(f"/proc/{rec['pid']}/status").read_text().splitlines() if x.startswith('Threads:')).split()[1])==1
        records.append(dict(rec,status=s,checkpoint_deserialized=True))
    write(output/'startup_health.json',dict(passed=True,workers=records,training_updates=0))

def smoke(output):
    _,jobs,bases,model=verify(output);job=next(j for j in jobs if j['condition']=='C4');c=begin(job,bases)
    # One historical first task and one real nested step. No outcomes used in design.
    while c['oracle'] is None:
        assert engine.advance(c,model) is None
    folder=output/'engineering_smoke';folder.mkdir(exist_ok=False)
    atlas.snapshot(folder,dict(c=c,rng=census.rng_state()));saved=atlas.restore(folder)
    before=pickle.dumps(c['env']);other_before=pickle.dumps(saved['c']['env'])
    census.set_rng(saved['rng']);a=engine.advance(c,model)
    census.set_rng(saved['rng']);b=engine.advance(saved['c'],model)
    census.assert_equal(census.normalized(a),census.normalized(b),'noenergy nested resume result')
    census.assert_equal(c['oracle']['env'].observe(),census.normalized(saved['c']['oracle']['env'].observe()),'noenergy nested resume observation')
    census.assert_equal(c['oracle']['env'].events,census.normalized(saved['c']['oracle']['env'].events),'noenergy nested resume events')
    assert pickle.dumps(c['env'])==before and pickle.dumps(saved['c']['env'])==other_before
    write(output/'engineering_smoke.json',dict(passed=True,job=job['id'],first_task_physics_exact_except_battery=True,nested_task_only_physical_step=True,midflight_disk_resume_exact=True,parent_unchanged=True,training_updates=0))
    print('Historical task-only and nested disk-resume smoke PASS',flush=True)

# Collection is deliberately complete-only; no intermediate response tables.
def prefix(r,w,root_queue):
    start=r['start_time'];end=start+w;failure=r.get('failure_time');stop=r['physical_end_time']
    known=(stop>=end-1e-7) or (failure is not None and failure<=end+1e-7)
    events=[e for e in r['events'] if e['time']<=end+1e-7]
    ids=[e['task_id'] for e in events if e['event']=='task_completed']
    failed=failure is not None and failure<=end+1e-7
    # Event-exact mode and queue integration, including clipped final segments.
    mode='IDLE';clock=start;times=Counter();q=len(root_queue);qclock=start;integral=0.
    live_end=min(end,failure if failed else stop)
    observed_end=end if failed else min(end,stop)
    for e in events:
        t=min(e['time'],observed_end)
        if t<clock-1e-7:continue
        times[mode]+=max(0.,t-clock);clock=t
        integral+=q*max(0.,t-qclock);qclock=t
        kind=e['event']
        if kind=='decision':
            a=e['action'];mode={'serve':'SERVING','recharge':'RETURNING','idle':'WAITING'}[a['kind']]
            if a['kind']=='serve':q-=1
        elif kind=='task_completed':mode='IDLE'
        elif kind=='charger_arrival':mode='CHARGING'
        elif kind=='charge_complete':mode='IDLE'
        elif kind=='arrival':q+=int(e['accepted'])
        elif kind=='counterpart_erased':q-=1
        elif kind=='failure':mode='FAILED'
    times[mode]+=max(0.,observed_end-clock);integral+=q*max(0.,observed_end-qclock)
    arrivals=[e for e in events if e['event']=='arrival']
    available=len(root_queue)+sum(e['accepted'] for e in arrivals)-sum(e['event']=='counterpart_erased' for e in events)
    return dict(window=w,N=len(ids) if known else None,N_observed=len(ids),survived=(not failed) if known else None,failure=r['failure'] if failed else None,time_alive=live_end-start,completed_ids=ids,available_task_count=available,observed_task_exhaustion=len(ids)==available,flight_time=times['SERVING']+times['RETURNING'],charge_time=times['CHARGING'],wait_time=times['WAITING']+times['IDLE'],failed_time=times['FAILED'],recharge_count=sum(e['event']=='decision' and e['action']['kind']=='recharge' for e in events),forced_returns=sum(e['event']=='decision' and e['action']['kind']=='recharge' and e['action'].get('reason') in ('empty_queue_return','no_safe_task_return','sweep_post_first_forced_regeneration') for e in events),accepted_arrival_ids=[e['task_id'] for e in arrivals if e['accepted']],overflow=sum(not e['accepted'] for e in arrivals),queue_occupancy_integral=integral,known=known)

def response(pair,cond,w,a,b,oa,ob):
    old=oa['N']-ob['N'] if oa['known'] and ob['known'] else None
    new=a['N']-b['N'] if a['known'] and b['known'] else None
    known=new is not None and old is not None
    return dict(pair_id=pair['id'],root_id=pair['root_id'],policy=pair['policy'],regime=pair['regime'],prior_divergent=pair['old_delta']!=0,condition=cond,window=w,baseline_gap=old,gap=new,abs_gap=abs(new) if new is not None else None,mean_throughput=(a['N']+b['N'])/2 if new is not None else None,disappeared=known and old!=0 and new==0,shrank=known and abs(new)<abs(old),unchanged=known and abs(new)==abs(old),enlarged=known and abs(new)>abs(old),reversed=known and old*new<0,induced=known and old==0 and new!=0,resolved=known,i=a,j=b)

def stats(ps):
    rs=[p for p in ps if p['resolved']];roots=defaultdict(list)
    for p in rs:roots[p['root_id']].append(p)
    return dict(pairs=len(ps),resolved=len(rs),unknown=len(ps)-len(rs),roots=len(roots),mean_abs_gap=atlas.mean([p['abs_gap'] for p in rs]),baseline_abs_gap=atlas.mean([abs(p['baseline_gap']) for p in rs]),mean_throughput=atlas.mean([p['mean_throughput'] for p in rs]),root_balanced_abs_gap=atlas.mean([atlas.mean([p['abs_gap'] for p in xs]) for xs in roots.values()]),**{key:sum(p[key] for p in rs) for key in ['disappeared','shrank','unchanged','enlarged','reversed','induced']})

def collect(output):
    m,jobs,bases,_=verify(output);refs=read(output/'imports.json')['results'][:]
    for w in range(16):
        folder=output/f'worker_{w:02d}';assert read(folder/'status.json')['status']=='complete'
        part=read(folder/'results.json');assert atlas.restore(folder)['results']==part;refs+=part
    assert len(refs)==len({r['job_id'] for r in refs})==2164
    assert {r['job_id'] for r in refs}=={j['id'] for j in jobs}
    jobmap={j['id']:j for j in jobs};measurements={};outcomes=Counter()
    for ref in refs:
        path=output/ref['file'];assert sha(path)==ref['sha256'];r=rawread(path);j=jobmap[ref['job_id']];base=bases[j['base']]
        validate(r,j,base);outcomes[j['condition']+':'+r['outcome']]+=1
        measurements[j['id']]={str(w):prefix(r,w,base['root']['observation']['queue']) for w in m['windows']}
        if r['N_W'] is not None:assert measurements[j['id']][str(m['windows'][-1])]['N']==r['N_W']
    baseline={}
    for key,b in bases.items():
        path=Path(b['root_folder']).parent/Path(b['baseline_file']).name;assert sha(path)==b['baseline_sha256'];r=read(path)
        baseline[key]={str(w):prefix(r,w,b['root']['observation']['queue']) for w in m['windows']}
    pairs=read(output/'pairs.json');responses=[]
    for p in pairs:
        for cond in engine.CONDITIONS:
            keys=[('E__'+p['id']+'_'+s) if cond=='E' else cond+'__'+p[s] for s in ('i','j')]
            for w in m['windows']:
                z=str(w);responses.append(response(p,cond,w,measurements[keys[0]][z],measurements[keys[1]][z],baseline[p['i']][z],baseline[p['j']][z]))
    groups={}
    for cond in engine.CONDITIONS:
        for w in m['windows']:
            subset=[p for p in responses if p['condition']==cond and p['window']==w]
            groups[cond+'/'+str(w)]={name:stats(xs) for name,xs in [('all',subset),('prior_divergent',[p for p in subset if p['prior_divergent']]),('prior_tied',[p for p in subset if not p['prior_divergent']])]+[(src,[p for p in subset if p['policy']==src]) for src in ('B4_0.75','B5')]+[(f'B={bm}',[p for p in subset if 2*(p['regime']//9+1)==bm]) for bm in (2,4,6)]}
    broad=[];byroot=defaultdict(list)
    for key,b in bases.items():byroot[b['root_id']].append(key)
    broad_roots=[]
    for rid,keys in byroot.items():
        for cond in ('C1','C4'):
            for w in m['windows']:
                z=str(w);ns=[measurements[cond+'__'+k][z]['N'] for k in keys];old=[baseline[k][z]['N'] for k in keys]
                broad_roots.append(dict(root_id=rid,condition=cond,window=w,resolved=all(n is not None for n in ns),gap=max(ns)-min(ns) if all(n is not None for n in ns) else None,baseline_gap=max(old)-min(old)))
                for i,j in itertools.combinations(keys,2):
                    p=dict(id=i+'__'+j,root_id=rid,policy=bases[i]['policy'],regime=bases[i]['regime'],old_delta=bases[i]['baseline_N']-bases[j]['baseline_N'])
                    broad.append(response(p,cond,w,measurements[cond+'__'+i][z],measurements[cond+'__'+j][z],baseline[i][z],baseline[j][z]))
    assert len(broad)==551*2*3
    payloads={'continuations.json':refs,'measurements.json':measurements,'pair_responses.json':responses,'broad_pair_responses.json':broad,'broad_roots.json':broad_roots,'summary.json':dict(primary=groups,broad={c+'/'+str(w):stats([p for p in broad if p['condition']==c and p['window']==w]) for c in ('C1','C4') for w in m['windows']},outcomes=dict(outcomes),unique_jobs=2164,imported=70,new=2094)}
    for name,data in payloads.items():write(output/name,data)
    write(output/'integrity.json',dict(passed=True,manifest_sha256=sha(output/'manifest.json'),raw_files=len(refs),output_hashes={n:sha(output/n) for n in payloads},training_updates=0))

def supervise(output):
    verify(output)
    while not (iq.DEFAULT/'integrity.json').exists():
        if list(iq.DEFAULT.glob('worker_*/error.json')):raise RuntimeError('prior IQ worker error; no new jobs launched')
        write(output/'supervisor_status.json',dict(status='waiting_for_prior_complete_verified',time=time.time(),manifest_sha256=sha(output/'manifest.json')))
        time.sleep(30)
    if not (output/'imports.json').exists():import_prior(output)
    # All former worker processes must have exited before new 16-worker allocation.
    while any(Path(f"/proc/{r['pid']}/cmdline").exists() and Path(f"/proc/{r['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0')==r['command'] for r in read(iq.DEFAULT/'launch.json')):time.sleep(5)
    if not (output/'launch.json').exists():launch(output)
    while True:
        if list(output.glob('worker_*/error.json')):raise RuntimeError('sweep worker error; preserve checkpoints; do not collect partial science')
        statuses=list(output.glob('worker_*/status.json'))
        if not (output/'startup_health.json').exists() and len(statuses)==16 and all(read(p)['policy_steps']>0 or read(p)['status']=='complete' for p in statuses):check(output)
        if len(statuses)==16 and all(read(p)['status']=='complete' for p in statuses):collect(output);write(output/'supervisor_status.json',dict(status='complete',time=time.time()));return
        write(output/'supervisor_status.json',dict(status='running_frozen_campaign',time=time.time()))
        time.sleep(30)

def defer(output):
    verify(output);assert read(output/'engineering_smoke.json')['passed'];assert not (output/'supervisor.json').exists()
    cmd=[str(atlas.WORK/'.venv/bin/python'),'-u',str(SCRIPT),'supervise','--output',str(output)]
    with (output/'supervisor.log').open('xb') as log:
        p=subprocess.Popen(cmd,cwd=PROJECT,env=dict(os.environ,**atlas.THREADS),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(output/'supervisor.json',dict(pid=p.pid,command=cmd,manifest_sha256=sha(output/'manifest.json')))
    print('Deferred complete-only campaign supervisor started',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','worker','collect','check','supervise','defer']);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--worker',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.output.resolve()
    try:
        if a.command=='worker':worker(out,a.worker,a.resume)
        else:globals()[a.command](out)
    except Exception as exc:
        if a.command=='supervise':write(out/'supervisor_error.json',dict(error=repr(exc),traceback=traceback.format_exc(),time=time.time()))
        raise
if __name__=='__main__':main()
