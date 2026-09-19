"""Frozen-actor P0 audit. No optimizers, policy training or downstream launches."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import subprocess
import zipfile
import numpy as np
import torch
from stable_baselines3.sac.policies import Actor
from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.correction_supervision import CorrectionRecovery

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT/'artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip'
EXPECTED = 'fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def atomic(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, path)

def actor_for(env):
    if sha(MODEL) != EXPECTED:
        raise RuntimeError('checkpoint hash mismatch')
    extractor = DirectionalLidarExtractor(env.observation_space, remaining_time=True)
    actor = Actor(env.observation_space, env.action_space, [256,256], extractor, extractor.features_dim)
    with zipfile.ZipFile(MODEL) as archive:
        state = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
    actor.load_state_dict({k[6:]: v for k,v in state.items() if k.startswith('actor.')}, strict=True)
    actor.eval().requires_grad_(False)
    return actor

def jobs():
    result=[]
    for band, distance in [('near',100),('medium',300),('far',600)]:
        for obstacles in [4,24]:
            for role in ['pickup','dropoff','return']:
                result.append(dict(id=len(result), seed=219190001+len(result), band=band,
                                   distance=distance, obstacles=obstacles, role=role))
    return result

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true'); p.add_argument('--resume',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    sources=[Path(__file__), ROOT/'envs/UAVEnergyDeliverySAC.py']
    sources+=list((ROOT/'experiments/directional_navigation').glob('*.py'))
    contract=dict(checkpoint=str(MODEL.relative_to(ROOT)),checkpoint_sha256=EXPECTED,training_seed=0,
                  environment_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                  source_hashes={str(x.relative_to(ROOT)):sha(x) for x in sources},
                  jobs=jobs(), horizon=4000, deterministic=True, optimizer_loaded=False,
                  observation_definition='velocity3 + goal direction3 + goal distance1 + LiDAR distances1024 + hit flags1024 + remaining option time1',
                  action_definition='normalized acceleration3; horizontal norm clipped, limits 5/5/3',
                  goal_conditioning='same frozen actor; goal-relative observation; HOCBF enabled unchanged',
                  scope='18-route engineering qualification, one route per cell; NOT population reliability or payload/service qualification',
                  stopping_rule='No P1/P2 promotion without navigation stability AND independently validated return and mission quantile coverage; no automatic training',
                  smoke=args.smoke)
    manifest=args.output/'freeze.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text())!=contract:
            raise RuntimeError('existing output requires identical-contract --resume')
    else: atomic(manifest,contract)
    for job in jobs()[:1] if args.smoke else jobs():
        dest=args.output/f"route_{job['id']:03d}.json"
        if dest.exists(): continue
        env=CorrectionRecovery(seed_start=job['seed'],seed_stride=1,obstacles=job['obstacles'])
        actor=actor_for(env)
        home=env.base.charger_position.copy(); d=job['distance']
        if job['role']=='return': start=home-np.array([d,0,0]); goal=home
        elif job['role']=='pickup': start=home; goal=home-np.array([d,0,0])
        else: start=home-np.array([d,0,0]); goal=start+np.array([0,d,0])
        obs,_=env.reset(seed=job['seed'],options={'start_position':start,'task_point':goal})
        initial=float(env.base.simulation_time); trajectory=[]; row=None
        for step in range(1,2 if args.smoke else 4001):
            with torch.inference_mode(): action=actor(torch.as_tensor(obs[None]),deterministic=True)[0].numpy()
            obs,_,done,truncated,info=env.step(action)
            if not np.isfinite(obs).all() or not np.isfinite(action).all(): raise RuntimeError('nonfinite rollout')
            if step==1 or step%100==0 or done:
                trajectory.append(dict(step=step,position=env.base.agent.pos.tolist(),collision_count=env.collision_count))
            if done or truncated:
                row=info['navigation_episode']; break
        if args.smoke: row={'smoke_only':True,'steps':step}
        if row is None: raise RuntimeError('missing terminal audit')
        atomic(dest,dict(**job,result=row,elapsed_seconds=float(env.base.simulation_time)-initial,
                        trajectory=trajectory,checkpoint_unchanged=sha(MODEL)==EXPECTED))
        env.close()
        print(json.dumps(dict(id=job['id'],role=job['role'],band=job['band'],result=row)),flush=True)
    rows=[json.loads(x.read_text()) for x in sorted(args.output.glob('route_*.json'))]
    atomic(args.output/'status.json',dict(completed=len(rows),planned=1 if args.smoke else len(jobs()),
          phase='P0',p1_authorized_by_gate=False,energy_coverage='NOT_VALIDATED',training_updates=0))

if __name__=='__main__': main()
