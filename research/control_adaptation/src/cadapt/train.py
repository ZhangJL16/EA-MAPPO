"""Episode-boundary resumable SB3 SAC pretraining on native public physics.

No automatic adaptation, gain tuning, best-seed selection or curriculum.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import tempfile
import time
import numpy as np
import torch
from stable_baselines3 import SAC
import stable_baselines3
from .environment import load_task,DeploymentEnv


def atomic_json(path,data):
    tmp=Path(str(path)+".tmp")
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    os.replace(tmp,path)


def source_binding(config,reference):
    return {"config":config,"source":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py'))},
        "task_sha256":hashlib.sha256((Path(reference)/config['task_yaml']).read_bytes()).hexdigest(),
        "torch":torch.__version__,"sb3":stable_baselines3.__version__,"numpy":np.__version__,
        "runtime_dependencies":{p:importlib.metadata.version(p) for p in
            ['pybullet','casadi','gymnasium','PyYAML','scipy']}}


def save_boundary(model,env,out,binding,history,seconds):
    # DummyVecEnv has reset after done. Resume reconstructs THAT same fresh state.
    if env.sim.ctrl_step_counter!=0:
        raise RuntimeError("Checkpoint must be at post-reset episode boundary")
    name=f"checkpoint_{model.num_timesteps}"
    tmp=Path(tempfile.mkdtemp(prefix=name+".pending-",dir=out))
    model.save(tmp/"model.zip")
    model.save_replay_buffer(tmp/"replay.pkl")
    rng={"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
        "action_space":env.action_space.np_random.bit_generator.state,
        "cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    torch.save(rng,tmp/"rng.pt")
    atomic_json(tmp/"state.json",{"steps":model.num_timesteps,"next_reset_index":env.reset_index-1,
        "history":history,"training_seconds":seconds,"binding":binding,
        "optimizer_updates":model._n_updates,"replay_size":model.replay_buffer.size(),
        "finite_parameters":all(bool(torch.isfinite(p).all()) for p in model.policy.parameters())})
    os.replace(tmp,out/name)
    atomic_json(out/"latest.json",{"checkpoint":name,"steps":model.num_timesteps})
    return out/name


def run(config,reference,output,seed,device="cpu",stop_after_checkpoint=False,max_episodes=None):
    torch.set_num_threads(config["threads"])
    task=load_task(reference,config)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=True)
    binding=source_binding(config,reference)
    latest=output/"latest.json"
    checkpoint=None
    if latest.exists():
        checkpoint=output/json.loads(latest.read_text())["checkpoint"]
        state=json.loads((checkpoint/"state.json").read_text())
        if state["binding"]!=binding:
            raise ValueError("Changed source/config/dependencies on resume")
        index=state["next_reset_index"]
        history=state["history"]
        previous_seconds=state["training_seconds"]
    else:
        index,history,previous_seconds=0,[],0.
        if (output/"run.json").exists():
            raise ValueError("Existing run without valid checkpoint; inspect instead of silently restarting")
        atomic_json(output/"run.json",{"binding":binding,"seed":seed,"device":device,
            "created_utc":time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),"pid":os.getpid()})
    original_run=json.loads((output/"run.json").read_text())
    if original_run["seed"]!=seed or original_run["device"]!=device:
        raise ValueError("Seed/device mismatch")
    env=DeploymentEnv(task,seed,reset_index=index)
    if checkpoint is None:
        model=SAC("MlpPolicy",env,learning_rate=config["learning_rate"],buffer_size=config["replay_size"],
            learning_starts=config["learning_starts"],batch_size=config["batch_size"],
            tau=.005,gamma=.99,train_freq=(1,"episode"),gradient_steps=-1,
            ent_coef=config["entropy_coefficient"],policy_kwargs={"net_arch":config["hidden"]},
            device=device,seed=seed,verbose=0)
    else:
        model=SAC.load(checkpoint/"model.zip",env=env,device=device,force_reset=True)
        model.load_replay_buffer(checkpoint/"replay.pkl")
        rng=torch.load(checkpoint/"rng.pt",weights_only=False,map_location="cpu")
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        env.action_space.np_random.bit_generator.state=rng["action_space"]
    stop=[False]
    def request_stop(signum,frame):
        stop[0]=True
    signal.signal(signal.SIGTERM,request_stop)
    signal.signal(signal.SIGINT,request_stop)
    start=time.monotonic()
    last_saved=model.num_timesteps if checkpoint else 0
    target=config["first_checkpoint_steps"] if not checkpoint else model.num_timesteps+config["checkpoint_steps"]
    episodes=0
    try:
        while model.num_timesteps<config["base_steps"]:
            # One whole episode, then one gradient update per collected transition.
            model.learn(total_timesteps=1,reset_num_timesteps=False,log_interval=None)
            episodes+=1
            history.append(dict(env.last_episode,total_steps=model.num_timesteps,updates=model._n_updates))
            if not all(bool(torch.isfinite(p).all()) for p in model.policy.parameters()):
                raise FloatingPointError("Nonfinite SAC parameters")
            timeout=time.monotonic()-start>=config["max_invocation_seconds"]
            should_stop=stop[0] or timeout or (max_episodes is not None and episodes>=max_episodes)
            if model.num_timesteps>=target or model.num_timesteps>=config["base_steps"] or should_stop:
                path=save_boundary(model,env,output,binding,history,previous_seconds+time.monotonic()-start)
                last_saved=model.num_timesteps
                tail=history[-10:]
                atomic_json(output/"status.json",{"steps":model.num_timesteps,"checkpoint":str(path),"pid":os.getpid(),
                    "updates":model._n_updates,"recent_return_mean":float(np.mean([x["return"] for x in tail])),
                    "recent_length_mean":float(np.mean([x["steps"] for x in tail])),
                    "state":"paused" if should_stop or stop_after_checkpoint else "training",
                    "finite_parameters":True})
                print(json.dumps(json.loads((output/"status.json").read_text())),flush=True)
                target=model.num_timesteps+config["checkpoint_steps"]
                if should_stop or stop_after_checkpoint:
                    break
        final=json.loads((output/"status.json").read_text())
        final["state"]="completed_base_budget" if model.num_timesteps>=config["base_steps"] else "paused"
        atomic_json(output/"status.json",final)
    finally:
        env.close()
    return last_saved


def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument("--config",required=True)
    p.add_argument("--reference",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--seed",type=int,default=5101)
    p.add_argument("--device",default="cpu")
    p.add_argument("--stop-after-checkpoint",action="store_true")
    p.add_argument("--max-episodes",type=int)
    args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    Path(args.output).mkdir(parents=True,exist_ok=True)
    with (Path(args.output)/'.runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run(cfg,args.reference,args.output,args.seed,args.device,args.stop_after_checkpoint,args.max_episodes)


if __name__=="__main__":
    main()
