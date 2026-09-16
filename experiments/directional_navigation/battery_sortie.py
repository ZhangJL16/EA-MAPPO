"""One battery sortie with frozen recovery physics and external neural decisions."""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from review_bundle.safety.switching.commitment import SortieMode
from experiments.directional_navigation.correction_supervision import CorrectionRecovery

CAPACITY = 378.72626091628933  # Historical SYNTHETIC energy units, not Wh.


class BatterySortie(CorrectionRecovery):
    def __init__(self, jobs: list[dict], guard: int = 20000):
        super().__init__(seed_start=1,seed_stride=1,horizon=4000,hocbf=True)
        self.jobs=jobs;self.cursor=0;self.guard=guard
        self.base.set_phase(SACTrainingPhase.ENERGY_MANAGED)
        self.base.configure_calibrated_battery(CAPACITY,reserve_fraction=0.,source='historical_synthetic_capacity')
        self.base.reset_at_charger=True;self.base.mission_switching_enabled=False
        self.base.energy_learning_enabled=False;self.base.phase2_episode_limit=guard
        self.observation_space=gym.spaces.Dict({
            'nav':self.observation_space,'return':self.observation_space,
            'battery':gym.spaces.Box(0.,np.inf,(1,),np.float32),
            'role':gym.spaces.Box(0.,np.inf,(5,),np.float32)})
        self.action_space=gym.spaces.Box(-1.,1.,(4,),np.float32)
        self.inactive=False;self.job={};self.contacts=0;self.return_steps=0

    def observation(self) -> dict:
        if self.inactive:
            return {k:np.zeros(space.shape,np.float32) for k,space in self.observation_space.spaces.items()}
        b=self.base;returning=b.mode is SortieMode.CHARGER_COMMITTED
        nav_time=max(0.,1.-(self.return_steps if returning else b.steps_in_current_task)/4000)
        nav=np.append(b.sac_observation_for_goal(b.active_goal),nav_time).astype(np.float32)
        # Counterfactual start of a fresh 4000-step return option, NOT remaining mission time.
        rth=np.append(b.charger_relative_observation(),1.).astype(np.float32)
        at_home=float(np.linalg.norm(b.agent.pos-b.charger_position)<=b.goal_tolerance)
        return {'nav':nav,'return':rth,'battery':np.array([b.agent.energy],np.float32),
                'role':np.array([self.job['method'],self.job['head_seed'],1,float(returning),at_home],np.float32)}

    def reset(self, *, seed=None, options=None):
        if self.cursor>=len(self.jobs):
            self.inactive=True;return self.observation(),{}
        self.inactive=False;self.job=self.jobs[self.cursor];self.cursor+=1
        self.base.bind_keyed_task_schedule(self.job['world_seed'])
        # No fresh RNG-dependent task stream per method; preserve keyed goals.
        self.base.reset(seed=self.job['world_seed'],options=options)
        self.base.agent.energy=CAPACITY*self.job['soc']
        self.base.cycle_start_energy=self.base.agent.energy
        self.contacts=0;self.return_steps=0;self.commit=None;self.total_steps=0
        return self.observation(),{}

    def commit_return(self) -> None:
        b=self.base
        if b.mode is SortieMode.CHARGER_COMMITTED:return
        self.commit={'step':self.total_steps,'energy':float(b.agent.energy),'position':b.agent.pos.tolist(),
                     'velocity':b.agent.vel.tolist(),'distance':float(np.linalg.norm(b.agent.pos-b.charger_position))}
        b._finalize_goal_trajectory(success=False,censored_reason='neural_return_probability_commit')
        b.mode=SortieMode.CHARGER_COMMITTED;b.agent.goal=b.charger_position
        b.steps_in_current_task=0;self.return_steps=0;b._start_goal_trajectory(b.charger_position)
        # No stop, teleport, recharge, or reset at the decision instant.

    def step(self, action):
        if self.inactive:return self.observation(),0.,False,False,{'inactive':True}
        a=np.asarray(action)
        if a.shape!=(4,) or not np.isfinite(a).all():raise ValueError('acceleration3 plus commitment required')
        if a[3]>0:self.commit_return()
        returning=self.base.mode is SortieMode.CHARGER_COMMITTED
        _,_,terminated,truncated,info=self.base.step(a[:3])
        self.total_steps+=1;self.return_steps+=int(returning)
        contact=bool(info['obstacle_collision'] or info['boundary_contact']);self.contacts+=int(contact)
        components=dict(info['reward_components'])
        if contact:
            for k in ('progress_reward_component','velocity_reward_component','task_completion_reward_component'):components[k]=0.
        cycle=info['battery_cycle_record']
        reached=bool(cycle is not None and cycle.get('return_success',False))
        return_deadline=returning and self.return_steps>=4000 and not reached
        done=bool(terminated or truncated or reached or return_deadline)
        light={'cost':float(contact),'collision_count':self.contacts}
        if done:
            reason='returned' if reached else 'return_deadline' if return_deadline else str(info['end_reason'])
            remaining=float(cycle['remaining_energy_at_cycle_end']) if cycle is not None else float(self.base.agent.energy)
            tasks=int(cycle['tasks_completed_in_cycle']) if reached else int(self.base.tasks_completed)
            light['sortie']={**self.job,'termination':reason,'returned':reached,
                'safe_return':reached and self.contacts==0,'collision_count':self.contacts,
                'steps':self.total_steps,'return_steps':self.return_steps,'tasks_completed':tasks,
                'initial_energy':CAPACITY*self.job['soc'],'remaining_energy':remaining,
                'energy_used':float(self.base.cumulative_virtual_energy),'commit':self.commit,
                'zero_task_return':reached and tasks==0,'energy_exhausted':bool(terminated and reason=='energy_exhausted')}
        return self.observation(),float(sum(components.values()))*.01,done,False,light
