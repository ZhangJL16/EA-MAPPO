"""Native safe-control-gym physics/reward with explicit API and privacy boundary."""
import copy
from pathlib import Path
import subprocess
import sys
import gymnasium as gym
import numpy as np
import yaml


def load_task(reference, config):
    reference=Path(reference).resolve()
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=reference,text=True).strip()
    if commit != config["source_commit"]:
        raise ValueError("Wrong safe-control-gym source revision")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=reference,text=True).strip():
        raise ValueError("Modified upstream source")
    if str(reference) not in sys.path:
        sys.path.insert(0,str(reference))
    task=yaml.safe_load((reference/config["task_yaml"]).read_text())["task_config"]
    return task


class PhysicalGain:
    """Evaluator-owned multiplicative gain AFTER denormalization, BEFORE clipping."""
    def __init__(self,gain):
        self._gain=np.array(gain,dtype=float)
        if self._gain.shape != (2,) or np.any(self._gain<=0):
            raise ValueError("Two positive actuator gains required")
    def reset(self,env):
        pass
    def seed(self,env):
        pass
    def apply(self,target,env):
        original=np.asarray(target)
        # Keep the upstream dtype: promoting float32 to float64 can change PWM
        # conversion enough to break even the gain=1 native-parity control.
        return (original*self._gain).astype(original.dtype,copy=False)


def split_done(done, info):
    truncated=bool(info.get("TimeLimit.truncated",False))
    return bool(done and not truncated),truncated


class DeploymentEnv(gym.Env):
    """Policy gets 12 native observations + previous physical obs/action + valid bit.

    Gain, symbolic model, true physical parameters and applied thrust are NOT
    returned in policy info. Full six-dimensional state is PUBLIC in this
    benchmark; do not confuse this with LiDAR-only UAV navigation.
    """
    metadata={"render_modes":[]}
    def __init__(self,task,seed,gain=(1.,1.),reset_index=0):
        from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
        opts=copy.deepcopy(task)
        opts["seed"]=seed
        opts["gui"]=False
        self.sim=Quadrotor(**opts)
        self.sim.disturbances["action"]=PhysicalGain(gain)
        self.action_space=gym.spaces.Box(-1.,1.,(2,),dtype=np.float32)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(21,),dtype=np.float32)
        self.root_seed=seed
        self.reset_index=reset_index
        self.last_native=None
        self.last_episode=None
        self._needs_reset=True

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        # Deterministic episode-indexed reset permits cross-process boundary resume.
        # SB3 re-sends its root seed when loading a model. Episode identity must
        # not jump back to episode zero on resume. Constructor root_seed owns it.
        episode_seed=self.root_seed+100003*self.reset_index
        self.reset_index+=1
        obs,_=self.sim.reset(seed=episode_seed)
        self.last_native=np.asarray(obs,dtype=np.float32)
        if self.last_native.shape != (12,):
            raise ValueError("Unexpected native observation layout")
        self.count=0
        self.total_return=0.
        self.mse_sum=0.
        self.violations=0
        self.saturations=0
        self._needs_reset=False
        return np.concatenate((self.last_native,np.zeros(9,dtype=np.float32))),{}

    def step(self,action):
        if self._needs_reset:
            raise RuntimeError("Reset required")
        action=np.asarray(action,dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Invalid action")
        command=np.clip(action,-1.,1.)
        obs,reward,done,info=self.sim.step(command)
        current=np.asarray(obs,dtype=np.float32)
        public=np.concatenate((current,self.last_native[:6],command,np.ones(1,dtype=np.float32)))
        self.last_native=current
        terminated,truncated=split_done(done,info)
        self._needs_reset=bool(done)
        self.count+=1
        self.total_return+=float(reward)
        self.mse_sum+=float(info["mse"])
        self.violations+=int(info.get("constraint_violation",0))
        saturation=bool(np.any(np.abs(command)>=.999) or np.any(action!=command))
        self.saturations+=int(saturation)
        # Whitelist only numeric diagnostics. SB3 policy itself receives obs only.
        safe={"mse":float(info["mse"]),"constraint_violation":int(info.get("constraint_violation",0)),
            "out_of_bounds":bool(info.get("out_of_bounds",False)),"command_saturated":saturation}
        if done:
            self.last_episode={"return":self.total_return,"steps":self.count,
                "position_rmse":float(np.sqrt(self.mse_sum/self.count)),
                "violation_steps":self.violations,"saturation_steps":self.saturations,
                "terminated":terminated,"truncated":truncated,"reset_index":self.reset_index-1}
            safe["episode_metrics"]=self.last_episode.copy()
        return public,float(reward),terminated,truncated,safe

    def close(self):
        self.sim.close()


def hover_feasibility(gain,scale=.1):
    """Necessary static check ONLY, not trajectory reachability proof."""
    required=(1/np.asarray(gain)-1)/scale
    return {"required_normalized_hover_command":required.tolist(),
        "within_action_box":bool(np.all(np.abs(required)<=1)),
        "scope":"necessary hover condition; trajectory feasibility still requires control evaluation"}
