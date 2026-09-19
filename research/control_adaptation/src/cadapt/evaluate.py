"""Read-only policy evaluation. No evaluation transitions enter training replay."""
import argparse
import json
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from .environment import DeploymentEnv,load_task,hover_feasibility
from .train import atomic_json


def evaluate(config,reference,output,checkpoint=None,episodes=None):
    task=load_task(reference,config)
    output=Path(output)
    if output.exists():
        raise FileExistsError(output)
    model=SAC.load(Path(checkpoint)/"model.zip",device="cpu") if checkpoint else None
    rows=[]
    for condition,gain in config["deployment_gains"].items():
        for seed in config["evaluation_seeds"][:episodes]:
            env=DeploymentEnv(task,seed,gain)
            try:
                obs,_=env.reset()
                if model is None:
                    from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain
                    symbolic=env.sim.symbolic
                    feedback=compute_lqr_gain(symbolic,symbolic.X_EQ,symbolic.U_EQ,
                        np.diag([1,.1,1,.1,.1,.1]),np.diag([.1,.1]),True)
                done=False
                while not done:
                    if model:
                        action,_=model.predict(obs,deterministic=True)
                    else:
                        # Nominal prior model only, public current state and next reference.
                        physical=-feedback @ (obs[:6]-obs[6:12])+symbolic.U_EQ
                        action=env.sim.normalize_action(physical)
                    obs,_,term,trunc,_=env.step(action)
                    done=term or trunc
                rows.append({"controller":"frozen_sac" if model else "nominal_lqr_with_same_action_box",
                    "condition":condition,"seed":seed,**env.last_episode})
            finally:
                env.close()
    atomic_json(output,{"scope":"DEV evaluation, not adaptation; no training replay writes",
        "checkpoint":str(checkpoint) if checkpoint else None,
        "hover_checks":{k:hover_feasibility(v) for k,v in config["deployment_gains"].items()},
        "rows":rows,"warning":"RMSE is over executed steps; always read with episode length/failure/completion"})
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--reference',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--checkpoint')
    p.add_argument('--episodes',type=int)
    a=p.parse_args()
    result=evaluate(json.loads(Path(a.config).read_text()),a.reference,a.output,a.checkpoint,a.episodes)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
