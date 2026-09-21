"""Frozen root-local queue-capacity intervention; reuse unchanged Atlas physics."""
import argparse
from dataclasses import asdict,replace
import json,os,pickle,signal,subprocess,sys,time,shlex
from pathlib import Path
from collections import defaultdict,Counter
import real_state_branching_atlas as atlas
census=atlas.census
PROJECT=atlas.PROJECT;SCRIPT=Path(__file__).resolve()
ORIGINAL=PROJECT/'evidence/real_state_branching_atlas_v1'
PAIRS=PROJECT/'evidence/atlas_retained_counterpart_v1/counterpart_pairs.jsonl'
PLAN=PROJECT/'INFINITE_QUEUE_INTERVENTION_V1.md'
DEFAULT=PROJECT/'evidence/atlas_infinite_queue_v1'
read=lambda p:json.loads(Path(p).read_text())
write=atlas.write_json;sha=atlas.sha

def unbounded(env):
    cap=max(env.config.queue_capacity,len(env._tasks))
    assert len(env.queue)+len(env._tasks)-env.cursor<=cap
    env.config=replace(env.config,queue_capacity=cap)
    env.queue_time.extend([0.]*(cap+1-len(env.queue_time)))
    return cap

def prepare(output):
    assert not (output/'manifest.json').exists()
    old,roots=atlas.verify(ORIGINAL);roots={r['id']:r for r in roots}
    assert sha(PAIRS)==read(PAIRS.parent/'integrity.json')['output_hashes']['counterpart_pairs.jsonl']
    assert sha(ORIGINAL/'one_step_branches.jsonl')==read(ORIGINAL/'integrity.json')['one_step_sha256']
    pairs=[p for p in map(json.loads,PAIRS.read_text().splitlines()) if p['delta_N']]
    assert len(pairs)==41
    jobs={};snapshots={p.parent.parent.name:p.parent for p in ORIGINAL.glob('worker_*/*/root/resume.json')}
    arows={(r['root_id'],r['spec']['task_id']):r for r in map(json.loads,(ORIGINAL/'one_step_branches.jsonl').read_text().splitlines()) if r['spec']['kind']=='task_return'}
    for p in pairs:
        for side in ['i','j']:
            rid=p['root_id'];tid=p[side]['first_task'];key=f'{rid}_task{tid}'
            folder=snapshots[rid];ref=read(folder/'resume.json');assert sha(folder/ref['file'])==ref['sha256']
            verified=read(folder.parent/'root_verified.json');assert verified['root_snapshot']['sha256']==ref['sha256']
            jobs[key]=dict(id=key,root_id=rid,task_id=tid,policy=p['policy'],regime=p['regime'],root=roots[rid],root_folder=str(folder),root_ref=ref,first=arows[rid,tid],baseline_N=p[side]['N_W'],baseline_file=p[side]['source_file'],baseline_sha256=p[side]['source_sha256'])
    jobs=[jobs[k] for k in sorted(jobs)];assert len(jobs)==70 and len({j['root_id'] for j in jobs})==29
    output.mkdir(parents=True,exist_ok=True);write(output/'jobs.json',jobs)
    pairmap=[dict(root_id=p['root_id'],task_i=p['task_i'],task_j=p['task_j'],policy=p['policy'],regime=p['regime'],seed=p['seed'],old_delta=p['delta_N'],i=f"{p['root_id']}_task{p['task_i']}",j=f"{p['root_id']}_task{p['task_j']}") for p in pairs]
    write(output/'pairs.json',pairmap)
    write(output/'manifest.json',dict(source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=atlas.ROOT,text=True).strip(),script_sha256=sha(SCRIPT),atlas_sha256=sha(atlas.SCRIPT),plan_sha256=sha(PLAN),jobs_sha256=sha(output/'jobs.json'),pairs_sha256=sha(output/'pairs.json'),source_pairs_sha256=sha(PAIRS),original_manifest_sha256=sha(ORIGINAL/'manifest.json'),workers=16,jobs=70,roots=29,pairs=41,window=atlas.WINDOW,intervention='root onward queue_capacity=max(old capacity,len(full frozen stream)); extend queue_time with zeros',selection='all 41 prior same-arrival joint-matched divergent pairs; outcome-conditioned historical selection',created_unix=time.time()))
    print('Prepared 70 jobs / 41 pairs / 29 roots',flush=True)

def verify(output):
    m=read(output/'manifest.json');assert m['script_sha256']==sha(SCRIPT) and m['atlas_sha256']==sha(atlas.SCRIPT) and m['plan_sha256']==sha(PLAN)
    assert m['jobs_sha256']==sha(output/'jobs.json') and m['pairs_sha256']==sha(output/'pairs.json') and m['source_pairs_sha256']==sha(PAIRS)
    assert m['original_manifest_sha256']==sha(ORIGINAL/'manifest.json')
    old,_=atlas.verify(ORIGINAL)
    return m,read(output/'jobs.json'),atlas.EstimateModel(**read(old['frozen'])['model'])

def root_for(job):
    folder=Path(job['root_folder']);assert read(folder/'resume.json')==job['root_ref']
    root=atlas.restore(folder);env=root['env'];census.assert_equal(env.observe(),job['root']['observation'],'archived root observation')
    assert atlas.stream_hash(env._tasks)==job['root']['stream_sha256']
    return root

def start(job):
    root=root_for(job);before=pickle.dumps(root,protocol=pickle.HIGHEST_PROTOCOL)
    c=atlas.new_continuation(root,job['first']);old=asdict(c['env'].config);cap=unbounded(c['env'])
    new=asdict(c['env'].config);assert {k for k in old if old[k]!=new[k]}<= {'queue_capacity'}
    assert pickle.dumps(root,protocol=pickle.HIGHEST_PROTOCOL)==before
    c['intervention_capacity']=cap;c['base_policy_steps']=c['env'].nav.policy_steps
    root['env'].nav.close();return c

def work_steps(c):return c['env'].nav.policy_steps-c['base_policy_steps']+c['oracle_physics'] if c else 0

def assert_admission(result):
    arrivals=[e for e in result['events'] if e['event']=='arrival'];assert all(e['accepted'] for e in arrivals) and result['overflow']==0
    for d in result['oracle_decisions']:
        assert {b['spec']['task_id'] for b in d['branches'] if b['spec']['kind']=='task_return'}=={q['id'] for q in d['observation']['queue']}
        for b in d['branches']:assert all(e['accepted'] for e in b['events'] if e['event']=='arrival')

def worker(output,w,resume=False):
    m,jobs,model=verify(output);assigned=jobs[w::m['workers']];folder=output/f'worker_{w:02d}'
    contract=dict(manifest=sha(output/'manifest.json'),worker=w,jobs=[j['id'] for j in assigned])
    with atlas.exclusive_run(folder):
        if resume:
            state=atlas.restore(folder);assert state['contract']==contract;census.set_rng(state['rng'])
        else:
            folder.mkdir(exist_ok=False);write(folder/'manifest.json',contract)
            state=dict(contract=contract,index=0,continuation=None,results=[],finished_steps=0,rng=census.rng_state())
        stop=[False]
        def halt(*a):stop[0]=True
        signal.signal(signal.SIGTERM,halt);signal.signal(signal.SIGINT,halt)
        def save(status):
            state['rng']=census.rng_state();ref=atlas.snapshot(folder,state);c=state['continuation']
            write(folder/'status.json',dict(status=status,index=state['index'],planned=len(assigned),job=assigned[state['index']]['id'] if state['index']<len(assigned) else None,stage=c['stage'] if c else None,policy_steps=state['finished_steps']+work_steps(c),queue_capacity=c['intervention_capacity'] if c else None,queue_length=len(c['env'].queue) if c else None,snapshot=ref,checkpoint_unix=time.time(),training_updates=0))
        saved=time.monotonic();steps=0
        try:
            while state['index']<len(assigned):
                job=assigned[state['index']]
                if state['continuation'] is None:state['continuation']=start(job);save('running')
                c=state['continuation'];r=atlas.advance_continuation(c,model)
                if r is not None:
                    assert_admission(r);assert r['stream_sha256']==job['root']['stream_sha256']
                    r.update(job_id=job['id'],root_id=job['root_id'],policy=job['policy'],regime=job['regime'],intervention_capacity=c['intervention_capacity'])
                    path=folder/(job['id']+'.json');write(path,r)
                    compact=atlas.compact(r);compact.update(file=str(path.relative_to(output)),sha256=sha(path));state['results'].append(compact)
                    state['finished_steps']+=work_steps(c);c['env'].nav.close();state['continuation']=None;state['index']+=1;save('running')
                total=state['finished_steps']+work_steps(state['continuation'])
                if stop[0] or total-steps>=100 or time.monotonic()-saved>=60:
                    save('paused' if stop[0] else 'running');saved=time.monotonic();steps=total
                    if stop[0]:return
            write(folder/'results.json',state['results']);save('complete')
        except Exception as e:
            write(folder/'error.json',dict(error=repr(e),job=assigned[state['index']]['id'] if state['index']<len(assigned) else None,time=time.time()));raise

def smoke(output):
    _,jobs,model=verify(output);job=jobs[0];folder=output/'engineering_smoke';folder.mkdir(exist_ok=False)
    # Finite-queue control: reproduce archived first-task events exactly.
    root=root_for(job);c=atlas.new_continuation(root,job['first'])
    while c['initial_completion'] is None:assert atlas.advance_continuation(c,model) is None
    original=next(ORIGINAL.glob('worker_*/'+job['baseline_file']));assert sha(original)==job['baseline_sha256'];raw=read(original)
    ci=next(i for i,e in enumerate(raw['events']) if e['event']=='task_completed')
    census.assert_equal(c['env'].events[c['event_start']:],raw['events'][:ci+1],'finite-capacity first-task control')
    c['env'].nav.close();root['env'].nav.close()
    c=start(job)
    while c['initial_completion'] is None:assert atlas.advance_continuation(c,model) is None
    # Advance one real nested step, then disk resume equivalence at same state.
    while c['oracle'] is None:
        assert atlas.advance_continuation(c,model) is None
    assert c['oracle']['env'].nav.policy_steps>c['env'].nav.policy_steps
    atlas.snapshot(folder/'resume',dict(c=c,rng=census.rng_state()))
    clone=atlas.restore(folder/'resume');parent_before=pickle.dumps(c['env']);clone_parent_before=pickle.dumps(clone['c']['env']);census.set_rng(clone['rng']);r1=atlas.advance_continuation(c,model)
    census.set_rng(clone['rng']);r2=atlas.advance_continuation(clone['c'],model)
    census.assert_equal(census.normalized(r1),census.normalized(r2),'nested resume result')
    census.assert_equal(c['oracle']['env'].observe(),census.normalized(clone['c']['oracle']['env'].observe()),'nested resume observation')
    census.assert_equal(c['oracle']['env'].events,census.normalized(clone['c']['oracle']['env'].events),'nested resume events')
    assert pickle.dumps(c['env'])==parent_before and pickle.dumps(clone['c']['env'])==clone_parent_before
    census.assert_equal(c['env'].observe(),census.normalized(clone['c']['env'].observe()),'resume parent observation')
    write(output/'engineering_smoke.json',dict(passed=True,job=job['id'],finite_control_first_task_exact=True,infinite_first_task_exact=True,root_clone_unchanged=True,nested_physical_step=True,midflight_disk_resume_exact=True,queue_capacity=c['intervention_capacity'],training_updates=0))
    print('Real root and nested resume smoke PASS',flush=True)

def launch(output):
    m,_,_=verify(output);assert read(output/'engineering_smoke.json')['passed'];assert not (output/'launch.json').exists()
    records=[]
    for i in range(m['workers']):
        command=[str(atlas.WORK/'.venv/bin/python'),'-u',str(SCRIPT),'worker','--output',str(output),'--worker',str(i)]
        with (output/f'worker_{i:02d}.log').open('xb') as log:
            proc=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,**atlas.THREADS),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(worker=i,pid=proc.pid,command=command));write(output/'launch.json',records)
    (output/'resume_commands.txt').write_text('\n'.join(shlex.join(r['command']+['--resume']) for r in records)+'\n')
    cmd=[str(atlas.WORK/'.venv/bin/python'),'-u',str(SCRIPT),'supervise','--output',str(output)]
    with (output/'supervisor.log').open('xb') as log:p=subprocess.Popen(cmd,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write(output/'supervisor.json',dict(pid=p.pid,command=cmd));print('Launched 16 workers and complete-only collector',flush=True)

def check(output):
    m,jobs,_=verify(output);records=[]
    for rec in read(output/'launch.json'):
        folder=output/f"worker_{rec['worker']:02d}"
        assert not (folder/'error.json').exists()
        status=read(folder/'status.json');assert status['policy_steps']>0 or status['status']=='complete'
        state=atlas.restore(folder);assert state['contract']['manifest']==sha(output/'manifest.json')
        if status['status']!='complete':
            command=Path(f"/proc/{rec['pid']}/cmdline").read_bytes().decode().strip('\0').split('\0');assert command==rec['command']
            thread_line=next(x for x in Path(f"/proc/{rec['pid']}/status").read_text().splitlines() if x.startswith('Threads:'))
            assert int(thread_line.split()[1])==1
        c=state['continuation']
        if c is not None:
            env=c['env'];assert env.config.queue_capacity==max(5,len(env._tasks))
            assert len(env.queue_time)==env.config.queue_capacity+1 and all(atlas.np.isfinite(env.observe()['position']))
            assert env.overflow==c['start_overflow']
        records.append(dict(worker=rec['worker'],pid=rec['pid'],status=status,checkpoint_deserialized=True,contract_exact=True))
    assert len(records)==16
    write(output/'startup_health.json',dict(passed=True,workers=records,training_updates=0,scope='first nonzero resumable checkpoint only; not scientific completion'))
    print('16-worker first-checkpoint health PASS',flush=True)

def collect(output):
    m,jobs,_=verify(output);results=[]
    for i in range(m['workers']):
        folder=output/f'worker_{i:02d}';assert read(folder/'status.json')['status']=='complete'
        shard=read(folder/'results.json');assert atlas.restore(folder)['results']==shard
        for r in shard:
            assert sha(output/r['file'])==r['sha256'];raw=read(output/r['file']);assert_admission(raw)
        results.extend(shard)
    lookup={r['job_id']:r for r in results};assert len(lookup)==len(results)==70 and set(lookup)=={j['id'] for j in jobs}
    for job in jobs:
        r=lookup[job['id']]
        if r['outcome']=='window_end':
            baseline=Path(job['root_folder']).parent/Path(job['baseline_file']).name
            assert sha(baseline)==job['baseline_sha256']
            old_arr=[(e['task_id'],e['time']) for e in read(baseline)['events'] if e['event']=='arrival']
            new_arr=[(e['task_id'],e['time']) for e in read(output/r['file'])['events'] if e['event']=='arrival']
            assert new_arr==old_arr
    comparisons=[]
    for p in read(output/'pairs.json'):
        x,y=lookup[p['i']],lookup[p['j']];resolved=x['N_W'] is not None and y['N_W'] is not None
        new=x['N_W']-y['N_W'] if resolved else None
        comparisons.append(dict(p,new_delta=new,resolved=resolved,absolute_gap_change=abs(new)-abs(p['old_delta']) if resolved else None,sign_reversed=new*p['old_delta']<0 if resolved else None,i_outcome=x['outcome'],j_outcome=y['outcome']))
        if x['outcome']==y['outcome']=='window_end':
            ax=[(e['task_id'],e['time']) for e in read(output/x['file'])['events'] if e['event']=='arrival']
            ay=[(e['task_id'],e['time']) for e in read(output/y['file'])['events'] if e['event']=='arrival'];assert ax==ay
    def stats(ps):
        rs=[p for p in ps if p['resolved']];byroot=defaultdict(list)
        for p in rs:byroot[p['root_id']].append(p)
        return dict(pairs=len(ps),resolved=len(rs),unknown=len(ps)-len(rs),old_abs_gap_mean=atlas.mean([abs(p['old_delta']) for p in rs]),new_abs_gap_mean=atlas.mean([abs(p['new_delta']) for p in rs]),root_balanced_abs_change=atlas.mean([atlas.mean([p['absolute_gap_change'] for p in xs]) for xs in byroot.values()]),disappeared=sum(p['new_delta']==0 for p in rs),smaller=sum(p['absolute_gap_change']<0 for p in rs),equal=sum(p['absolute_gap_change']==0 for p in rs),larger=sum(p['absolute_gap_change']>0 for p in rs),reversed=sum(p['sign_reversed'] for p in rs))
    groups={'all':comparisons}
    for src in ['B4_0.75','B5']:groups[src]=[p for p in comparisons if p['policy']==src]
    for bm in [2,4,6]:groups['B='+str(bm)]=[p for p in comparisons if 2*(p['regime']//9+1)==bm]
    write(output/'continuations.json',results);write(output/'pair_comparisons.json',comparisons)
    write(output/'summary.json',dict(groups={k:stats(v) for k,v in groups.items()},outcomes=dict(Counter(r['outcome'] for r in results)),unique_jobs=70,pairs=41,all_observed_arrivals_accepted=True))
    write(output/'integrity.json',dict(passed=True,manifest_sha256=sha(output/'manifest.json'),output_hashes={n:sha(output/n) for n in ['continuations.json','pair_comparisons.json','summary.json']},raw_files=70))

def supervise(output):
    while True:
        if list(output.glob('worker_*/error.json')):write(output/'collection_blocked.json',dict(reason='worker error; preserve checkpoints'));return
        statuses=list(output.glob('worker_*/status.json'))
        if len(statuses)==16 and all(read(p)['status']=='complete' for p in statuses):collect(output);return
        time.sleep(30)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','smoke','worker','launch','collect','supervise','check']);p.add_argument('--output',type=Path,default=DEFAULT);p.add_argument('--worker',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args();out=a.output.resolve()
    if a.command=='worker':worker(out,a.worker,a.resume)
    else:globals()[a.command](out)
if __name__=='__main__':main()
