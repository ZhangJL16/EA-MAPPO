"""One entry, frozen parameters, restartable deterministic jobs, no outcome gates."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from single_life_rl.scripts.io_utils import write_json, digest
from single_life_rl.scripts.freeze_protocol import freeze, verify


def suite_worker(args):
    output,suite,worker,workers=args
    output=Path(output)
    f=verify(output); config=f['config']
    folder=output/suite/f'worker_{worker:02d}';folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'process.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    jobs=[j for j in json.loads((output/'jobs.json').read_text()) if j['suite']==suite][worker::workers]
    from single_life_rl.algorithms.safe_refine import run_toy
    from single_life_rl.envs.uav_mission_library import run_uav
    library=json.loads((output/'library/library.json').read_text()) if suite=='uav' else None
    done=0
    for job in jobs:
        file=folder/(job['job_id']+'.json')
        if file.exists():
            row=json.loads(file.read_text())
            if row['spec']!=job: raise RuntimeError('existing result contract mismatch')
        else:
            active=dict(config,delta=job['delta'])
            if suite=='uav' and library['battery'] is None:
                row=dict(job_id=job['job_id'],spec=job,suite=suite,method=job['method'],status='structurally_unavailable_no_successful_missions')
            else:
                row=run_uav(job,active,library) if suite=='uav' else run_toy(job,active)
            write_json(file,row)
        done+=1
        if done%20==0 or done==len(jobs):
            write_json(folder/'status.json',dict(status='complete' if done==len(jobs) else 'running',completed=done,planned=len(jobs),updated=time.time(),training_updates=0))
    return {'worker':worker,'suite':suite,'complete':True}


def run(config_path,output,workers):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    lock=(output/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    f=verify(output) if (output/'freeze.json').exists() else freeze(config_path,output)
    if Path(config_path).resolve()!=Path(f['config_path']):raise RuntimeError('different config argument')
    if workers!=f['config']['workers']:raise ValueError('workers differs from frozen protocol')
    from single_life_rl.scripts.build_uav_library import prepare_library, mission_worker, collect_library
    write_json(output/'status.json',dict(stage='physical_library',status='running',time=time.time()))
    manifest=prepare_library(f['config'],output/'library')
    ctx=mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers,mp_context=ctx) as pool:
        fs=[pool.submit(mission_worker,(str(output/'library'),str(manifest),i,workers)) for i in range(workers)]
        for future in as_completed(fs):
            result=future.result()
            if result.get('paused'):raise RuntimeError('mission worker paused; resume same entry')
        library=collect_library(f['config'],output/'library')
        write_json(output/'library_frozen.json',dict(sha256=digest(output/'library/library.json'),derived_before_learning=True))
        for suite in ('binary','recursive','nuisance','random','uav'):
            verify(output)
            write_json(output/'status.json',dict(stage=suite,status='running',time=time.time()))
            fs=[pool.submit(suite_worker,(str(output),suite,i,workers)) for i in range(workers)]
            for future in as_completed(fs): future.result()
    verify(output)
    from single_life_rl.scripts.collect import collect
    from single_life_rl.scripts.analyze import analyze
    collect(output);analyze(output)
    write_json(output/'status.json',dict(status='complete',stage='analysis_complete',time=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--workers',type=int,default=16)
    parser.add_argument('--output',default='single_life_rl/artifacts/protocol_v1')
    args=parser.parse_args()
    try:run(args.config,args.output,args.workers)
    except BaseException as e:
        write_json(Path(args.output)/'driver_error.json',dict(error=repr(e),time=time.time()))
        raise
