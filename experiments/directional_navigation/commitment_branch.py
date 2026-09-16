"""Offline paired intervention at a frozen controller's actual switch state."""
from __future__ import annotations
import hashlib
import numpy as np
from experiments.directional_navigation.battery_sortie import BatterySortie


class CommitmentBranch(BatterySortie):
    def __init__(self, jobs: list[dict], guard: int = 20000):
        self.branch_phase='prefix';self.anchor_payload=None;self.control_row=None
        self.anchor_info=None;self.anchor_tasks=0
        self.suffix_trace=[];self.control_trace=[]
        super().__init__(jobs,guard)

    def reset(self, *, seed=None, options=None):
        self.branch_phase='prefix';self.anchor_payload=None;self.control_row=None
        self.anchor_info=None;self.anchor_tasks=0
        self.suffix_trace=[];self.control_trace=[]
        return super().reset(seed=seed,options=options)

    def observation(self) -> dict:
        obs=super().observation()
        if not self.inactive and self.branch_phase=='continuation':
            # Skip energy-head decisions only; nav goal and task clock stay real.
            obs['role'][3]=1.
        return obs

    def _capture_anchor(self) -> None:
        obs=super().observation();b=self.base
        self.anchor_tasks=int(b.tasks_completed)
        self.anchor_info={'step':self.total_steps,'tasks_completed':self.anchor_tasks,
            'task_steps':int(b.steps_in_current_task),'energy':float(b.agent.energy),
            'position':b.agent.pos.tolist(),'velocity':b.agent.vel.tolist(),
            'goal':b.current_task_point.tolist(),'contacts':self.contacts,
            'nav_observation':obs['nav'].tolist(),'return_observation':obs['return'].tolist()}
        # Payload has no reference to itself. Both suffixes share this full state.
        self.anchor_payload=super().snapshot()
        self.branch_phase='control'

    def step(self, action):
        if self.inactive:return super().step(action)
        action=np.asarray(action,dtype=np.float32).copy()
        if self.branch_phase=='prefix' and action[3]>0:self._capture_anchor()
        if self.branch_phase=='continuation':action[3]=0.
        obs,reward,done,truncated,info=super().step(action)
        if self.branch_phase!='prefix' and (self.total_steps%16==0 or done):
            b=self.base
            self.suffix_trace.append({'step':self.total_steps,'phase':self.branch_phase,
                'position':b.agent.pos.tolist(),'velocity':b.agent.vel.tolist(),
                'energy':info['sortie']['remaining_energy'] if done else float(b.agent.energy),
                'home_distance':float(np.linalg.norm(b.agent.pos-b.charger_position)),
                'collision_count':self.contacts,'hocbf_calls':int(b.safety_filter_calls),
                'hocbf_interventions':int(b.safety_interventions),
                'nominal_action':action[:3].tolist()})
        if done and self.branch_phase=='control':
            control=info['sortie'];payload=self.anchor_payload
            trace=self.suffix_trace
            digest=hashlib.sha256(payload).hexdigest()
            super().restore(payload)
            self.control_row=control;self.anchor_payload=None;self.branch_phase='continuation'
            self.control_trace=trace;self.suffix_trace=[]
            self.anchor_info['snapshot_sha256']=digest
            obs=self.observation()
            # Explicit cross-branch identity before executing any alternate step.
            assert obs['battery'][0]==np.float32(self.anchor_info['energy'])
            np.testing.assert_array_equal(obs['nav'],np.asarray(self.anchor_info['nav_observation'],np.float32))
            np.testing.assert_array_equal(obs['return'],np.asarray(self.anchor_info['return_observation'],np.float32))
            return obs,reward,False,False,{'control_finished':True}
        if done:
            row=info['sortie']
            paired=self.control_row is not None
            info={'pair':{**self.job,'paired':paired,'anchor':self.anchor_info,
                         'control':self.control_row if paired else row,
                         'continue_then_return':row if paired else None,
                         'control_trace':self.control_trace,'continue_trace':self.suffix_trace}}
            return obs,reward,True,False,info
        if self.branch_phase=='continuation' and self.base.tasks_completed>self.anchor_tasks:
            # Existing service reset is part of task execution; no custom reset.
            self.commit_return();self.branch_phase='alternate_return';obs=self.observation()
        return obs,reward,False,False,info
