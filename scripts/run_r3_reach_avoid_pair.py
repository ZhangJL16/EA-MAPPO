"""Resumable 50k/50k development pair; never auto-promotes to long training."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import signal
import tempfile
import time
import traceback

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

from experiments.jacobian_energy_bridge.sac import JacobianBridgeSAC
from experiments.r3_collision.training import ReachAvoidEnv, ReachAvoidReplay, ReachAvoidSAC
from scripts.run_r3_collision_interface_audit import (
    ROOT, BASE, AuditEnv, atomic_json, environment_contract, sha256,
)
from scripts.train_uav_energy_delivery_sac import (
    _configure_navigation_worker_threads, generate_stratified_navigation_tasks,
)

AUDIT = ROOT/'artifacts/r3_collision_interface_audit_50pairs_20260905_v1'
CHECKPOINT = BASE/'phase1_navigation/checkpoint_transition_500000.zip'
STOP = [False]


def parameter_hash(module) -> str:
    digest = hashlib.sha256()
    for key,value in module.state_dict().items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def calibrate() -> dict:
    result = json.loads((AUDIT/'RESULT.json').read_text())
    if result['decision'] != 'READY_FOR_MATCHED_TRAINING':
        raise RuntimeError('interface gate has not passed')
    rows = [json.loads(p.read_text()) for p in (AUDIT/'records').glob('*robust.json')]
    h = np.array([r['correction_integral'] for r in rows if r['safe_goal']])
    scale = float(np.quantile(h,.9))
    if scale <= 0:
        raise RuntimeError('no usable continuous signal')
    kappa = float(np.log(2)/scale)
    return dict(kappa=kappa,gamma=result['gamma'],calibration='half weight at development H90',
        success_count=len(h),h90=scale,old_kappa=result['kappa'],
        old_weight_below_001=int((np.exp(-result['kappa']*h)<.01).sum()),
        new_min_success_weight=float(np.exp(-kappa*h.max())),
        new_min_possible_step_weight=float(np.exp(-kappa*.2)),
        source_sha256=sha256(AUDIT/'RESULT.json'))


def save_checkpoint(model, folder: Path) -> None:
    parent = folder/'checkpoints'
    parent.mkdir(exist_ok=True)
    final = parent/f'{model.num_timesteps:08d}'
    if not final.exists():
        temporary = Path(tempfile.mkdtemp(prefix='.saving-',dir=parent))
        model.save(temporary/'model.zip')
        model.save_replay_buffer(temporary/'replay.pkl')
        rng = dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
        with (temporary/'rng.pkl').open('wb') as stream:
            pickle.dump(rng,stream)
        atomic_json(temporary/'metadata.json',dict(transitions=model.num_timesteps,
                    actor_updates=model.actor_updates,mc_updates=model.mc_updates,
                    resume_semantics='model+optimizers+replay+RNG; restart unfinished simulator episodes'))
        # Publish directory only after every member is closed; latest never points to partial files.
        temporary.rename(final)
    atomic_json(folder/'LATEST.json',dict(directory=str(final.relative_to(folder)),transitions=model.num_timesteps))


class EpisodeLog(BaseCallback):
    def __init__(self, folder):
        super().__init__()
        self.folder = folder

    def _on_step(self):
        rows = [dict(transitions=self.num_timesteps,worker=i,**info['ra_episode'])
                for i,info in enumerate(self.locals['infos']) if 'ra_episode' in info]
        if rows:
            with (self.folder/'training_episodes.jsonl').open('a') as stream:
                for row in rows:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
        return True


def make_model(vec, args, gamma):
    old = JacobianBridgeSAC.load(CHECKPOINT,device=args.device)
    model = ReachAvoidSAC('MlpPolicy',vec,policy_kwargs=old.policy_kwargs,
        learning_rate=3e-4,buffer_size=args.transitions,batch_size=args.batch_size,
        learning_starts=args.learning_starts,train_freq=1,gradient_steps=-1,
        tau=old.tau,gamma=gamma,ent_coef=0.0,replay_buffer_class=ReachAvoidReplay,
        warmup_transitions=args.warmup,training_budget=args.transitions,
        actor_lr=3e-5,actor_entropy=.001,seed=args.seed,device=args.device,verbose=0)
    model.actor.load_state_dict(old.actor.state_dict())
    audit = dict(actor_equals_R3=parameter_hash(model.actor)==parameter_hash(old.actor),
        critic_reset=parameter_hash(model.critic)!=parameter_hash(old.critic),
        target_equals_critic=parameter_hash(model.critic)==parameter_hash(model.critic_target),
        critic_count=len(model.critic.q_networks),observation_shape=list(vec.observation_space.shape),
        shared_extractor=model.policy.share_features_extractor,
        actor_hash=parameter_hash(model.actor),critic_hash=parameter_hash(model.critic),
        actor_optimizer_fresh=len(model.actor.optimizer.state)==0,
        critic_optimizer_fresh=len(model.critic.optimizer.state)==0)
    if not all(audit[k] for k in ['actor_equals_R3','critic_reset','target_equals_critic',
                                  'actor_optimizer_fresh','critic_optimizer_fresh']):
        raise AssertionError('invalid warm-start/reset contract')
    if audit['critic_count'] != 2 or audit['shared_extractor']:
        raise AssertionError('exactly two independent-of-actor critics required')
    del old
    return model,audit


def train_arm(args, arm, kappa, gamma):
    folder = Path(args.output)/arm
    folder.mkdir(exist_ok=True)
    if (folder/'TRAINING_COMPLETED.json').exists():
        return True
    latest = json.loads((folder/'LATEST.json').read_text()) if (folder/'LATEST.json').exists() else None
    elapsed_steps = latest['transitions'] if latest else 0
    kwargs = environment_contract()
    kwargs.update(hocbf_sampled_data_robust=True,projection_geometry_enabled=False)
    if args.smoke_steps:
        kwargs.update(phase1_episode_max_policy_steps=args.smoke_steps,max_steps_per_task=args.smoke_steps)
    def factory():
        _configure_navigation_worker_threads()
        return ReachAvoidEnv(kappa=kappa,discount=gamma,**kwargs)
    vec = SubprocVecEnv([factory for _ in range(args.workers)],start_method='forkserver')
    started = time.monotonic()
    try:
        if latest:
            checkpoint = folder/latest['directory']
            model = ReachAvoidSAC.load(checkpoint/'model.zip',env=vec,device=args.device,force_reset=True)
            model.load_replay_buffer(checkpoint/'replay.pkl')
            model.replay_buffer.discard_unfinished_episodes()
            with (checkpoint/'rng.pkl').open('rb') as stream:
                rng=pickle.load(stream)
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'])
            if rng['cuda'] and str(args.device).startswith('cuda'):
                torch.cuda.set_rng_state_all(rng['cuda'])
            vec.seed(args.seed+100000+elapsed_steps)
            atomic_json(folder/'RESUME.json',dict(from_transitions=elapsed_steps,
                restarted_simulator_episodes=True,unix_time=time.time()))
        else:
            model,audit = make_model(vec,args,gamma)
            atomic_json(folder/'INITIALIZATION.json',audit)
        model.set_logger(configure(str(folder/'sb3_logs'),[]))
        callback = EpisodeLog(folder)
        while model.num_timesteps < args.transitions:
            model.learn(min(args.chunk,args.transitions-model.num_timesteps),
                        reset_num_timesteps=False,callback=callback)
            pause = STOP[0] or (Path(args.output)/'PAUSE_REQUEST').exists()
            if model.num_timesteps % args.save_every == 0 or pause or model.num_timesteps >= args.transitions:
                save_checkpoint(model,folder)
            status = dict(status='PAUSED' if pause else 'TRAINING',arm=arm,
                transitions=model.num_timesteps,budget=args.transitions,
                elapsed_this_invocation=time.monotonic()-started,device=str(model.device),
                workers=args.workers,pid=os.getpid(),**model.last_ra_metrics)
            atomic_json(Path(args.output)/'PROGRESS.json',status)
            print(json.dumps(status),flush=True)
            if pause:
                return False
        atomic_json(folder/'TRAINING_COMPLETED.json',dict(transitions=model.num_timesteps,
                    actor_updates=model.actor_updates,mc_updates=model.mc_updates))
        return True
    finally:
        vec.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def evaluate(args,label):
    folder = Path(args.output)/label/'evaluation'
    folder.mkdir(parents=True,exist_ok=True)
    tasks = generate_stratified_navigation_tasks(num_tasks=args.eval_tasks,seed=args.eval_task_seed)
    jobs = []
    for i,task in enumerate(tasks):
        for arm in ('robust','off'):
            if (folder/f'{i:03d}_{arm}.json').exists():
                continue
            jobs.append(dict(case=i,arm=arm,world_seed=args.eval_world_seed+i,distance=task.straight_line_distance,
                distance_bucket=task.distance_bucket,options=dict(start_position=task.start_position,
                    start_velocity=task.initial_velocity,task_point=task.goal_position)))
    if jobs:
        if label=='frozen_R3':
            model=JacobianBridgeSAC.load(CHECKPOINT,device=args.device)
        else:
            base=Path(args.output)/label
            latest=json.loads((base/'LATEST.json').read_text())
            model=ReachAvoidSAC.load(base/latest['directory']/'model.zip',device=args.device)
        model.policy.set_training_mode(False)
        kwargs=environment_contract()
        kwargs['projection_geometry_enabled']=False
        if args.smoke_steps:
            kwargs.update(phase1_episode_max_policy_steps=args.smoke_steps,max_steps_per_task=args.smoke_steps)
        workers=min(args.workers,len(jobs))
        factories=[]
        for rank in range(workers):
            local=jobs[rank::workers]
            def factory(local=local):
                _configure_navigation_worker_threads()
                return AuditEnv(kwargs,local)
            factories.append(factory)
        vec=SubprocVecEnv(factories,start_method='forkserver')
        try:
            obs=vec.reset()
            remaining=len(jobs)
            ticks=0
            while remaining:
                actions,_=model.predict(obs,deterministic=True)
                obs,_,_,infos=vec.step(actions)
                for info in infos:
                    if 'record' in info:
                        r=info['record']
                        atomic_json(folder/f"{r['case']:03d}_{r['arm']}.json",r)
                        remaining-=1
                ticks+=1
                if ticks%100==0:
                    atomic_json(Path(args.output)/'PROGRESS.json',dict(status='EVALUATING',
                        label=label,completed=2*args.eval_tasks-remaining,total=2*args.eval_tasks,pid=os.getpid()))
                if STOP[0] or (Path(args.output)/'PAUSE_REQUEST').exists():
                    atomic_json(Path(args.output)/'PROGRESS.json',dict(status='PAUSED',phase='evaluation',label=label))
                    return False
        finally:
            vec.close()
    records=[json.loads(p.read_text()) for p in folder.glob('[0-9]*_*.json')]
    if len(records)!=2*args.eval_tasks:
        raise AssertionError('incomplete evaluation')
    summary={}
    for arm in ('robust','off'):
        rows=[r for r in records if r['arm']==arm]
        steps=sum(r['steps'] for r in rows)
        safe=[r for r in rows if r['safe_goal']]
        summary[arm]=dict(tasks=len(rows),safe_goals=len(safe),contacts=sum(r['contact'] for r in rows),
            timeouts=sum(not r['safe_goal'] and not r['contact'] for r in rows),
            successful_mean_path_ratio=float(np.mean([r['path_ratio'] for r in safe])) if safe else None,
            intervention_step_rate=sum(r['intervention'] for r in rows)/steps,
            emergency_step_rate=sum(r['emergency'] for r in rows)/steps,
            correction_integral=sum(r['correction_integral'] for r in rows))
    atomic_json(Path(args.output)/label/'EVALUATION.json',summary)
    return True


def run(args):
    output=Path(args.output).resolve()
    args.output=str(output)
    output.mkdir(parents=True,exist_ok=True)
    calibration=calibrate()
    sources=['scripts/run_r3_reach_avoid_pair.py','experiments/r3_collision/training.py',
        'experiments/r3_collision/core.py','scripts/run_r3_collision_interface_audit.py',
        'envs/UAVEnergyDeliverySAC.py','review_bundle/safety/collision/filter.py',
        'review_bundle/safety/collision/hocbf.py','experiments/jacobian_energy_bridge/features.py']
    protocol=dict(version=2,arguments=vars(args),calibration=calibration,
        checkpoint_sha256=sha256(CHECKPOINT),source_hashes={s:sha256(ROOT/s) for s in sources},
        algorithm='SAC-derived entropy-free reach-avoid twin Q; frozen-actor MC initialization',
        primary_comparison='hard kappa=0 vs continuous kappa=calibrated; all other factors matched',
        test_seeds=[args.eval_task_seed,args.eval_world_seed],automatic_scale_up=False)
    path=output/'PROTOCOL.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise RuntimeError('resume protocol/code mismatch; use a new experiment directory')
    atomic_json(path,protocol)
    if (output/'PAUSE_REQUEST').exists():
        raise RuntimeError('remove PAUSE_REQUEST before resuming')
    if (output/'FAILED.json').exists():
        (output/'FAILED.json').rename(output/f'PREVIOUS_FAILURE_{time.time_ns()}.json')
    for arm,kappa in [('hard',0.0),('continuous',calibration['kappa'])]:
        if not train_arm(args,arm,kappa,calibration['gamma']):
            return
    for label in ['hard','continuous','frozen_R3']:
        if not evaluate(args,label):
            return
    # Verify all three models were evaluated on the exact same initial observations/scenes.
    for i in range(args.eval_tasks):
        hashes=set()
        for label in ['hard','continuous','frozen_R3']:
            for arm in ['robust','off']:
                r=json.loads((output/label/'evaluation'/f'{i:03d}_{arm}.json').read_text())
                hashes.add((r['obstacle_hash'],r['initial_observation_hash']))
        if len(hashes)!=1:
            raise AssertionError('evaluation scenes are not matched')
    result=dict(status='COMPLETED',decision='REVIEW_REQUIRED_NO_AUTO_SCALE',
        smoke_only=bool(args.smoke or args.smoke_steps),
        results={label:json.loads((output/label/'EVALUATION.json').read_text())
                 for label in ['hard','continuous','frozen_R3']})
    atomic_json(output/'RESULT.json',result)
    atomic_json(output/'COMPLETED.json',dict(unix_time=time.time(),decision=result['decision']))
    atomic_json(output/'PROGRESS.json',dict(status='COMPLETED'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--transitions',type=int,default=50000)
    parser.add_argument('--warmup',type=int,default=10000)
    parser.add_argument('--learning-starts',type=int,default=256)
    parser.add_argument('--batch-size',type=int,default=256)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--chunk',type=int,default=1000)
    parser.add_argument('--save-every',type=int,default=10000)
    parser.add_argument('--eval-tasks',type=int,default=50)
    parser.add_argument('--eval-task-seed',type=int,default=710001)
    parser.add_argument('--eval-world-seed',type=int,default=720001)
    parser.add_argument('--seed',type=int,default=530001)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--smoke-steps',type=int,default=0)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if (min(args.workers,args.chunk,args.save_every,args.transitions,args.batch_size)<=0
        or args.eval_tasks<5 or args.eval_tasks%5 or not 0<=args.warmup<args.transitions
        or args.chunk%args.workers or args.transitions%args.chunk or args.save_every%args.chunk):
        parser.error('invalid budgets/divisibility/worker count')
    _configure_navigation_worker_threads()
    for sig in [signal.SIGTERM,signal.SIGINT]:
        signal.signal(sig,lambda *_:STOP.__setitem__(0,True))
    try:
        run(args)
    except Exception:
        atomic_json(Path(args.output)/'FAILED.json',dict(traceback=traceback.format_exc()))
        atomic_json(Path(args.output)/'PROGRESS.json',dict(status='FAILED',pid=os.getpid()))
        raise


if __name__=='__main__':
    main()
