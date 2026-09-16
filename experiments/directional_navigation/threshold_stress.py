"""Unlimited-duration, first-depletion stress test; locked contact physics."""
from __future__ import annotations

import numpy as np

from experiments.directional_navigation.battery_sortie import BatterySortie, CAPACITY
from review_bundle.safety.switching.commitment import SortieMode


def stress_jobs():
    return [dict(job_id=i*4+j*2+m, world_seed=1120000001+i, head_seed=0,
                 method=m, method_name=f'soc{int(q*100)}', threshold=q,
                 obstacles=n, soc=1.)
            for i in range(2) for j,n in enumerate((24,48))
            for m,q in enumerate((.2,.4))]


class ThresholdStress(BatterySortie):
    def __init__(self, job: dict):
        super().__init__([job])
        self.base.num_obstacles = job['obstacles']
        # Explicitly authorized experiment-only removal; frozen E1 unchanged.
        self.base.max_steps_per_task = float('inf')
        self.base.phase2_episode_limit = float('inf')

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.recharges = 0
        self.replenished = 0.
        self.last_tasks = 0
        self.cycle_contacts_start = 0
        self.exhausted = False
        return obs, info

    def diagnostic(self):
        b = self.base
        return dict(**self.job, step=self.total_steps, simulation_seconds=b.simulation_time,
                    tasks=int(b.tasks_completed), recharges=self.recharges,
                    battery=float(b.agent.energy), collision_count=self.contacts,
                    mode=b.mode.value, return_steps=self.return_steps,
                    task_steps=b.steps_in_current_task, position=b.agent.pos.tolist(),
                    velocity=b.agent.vel.tolist(), charger=b.charger_position.tolist(),
                    distance=float(np.linalg.norm(b.agent.pos-b.charger_position)),
                    commit=self.commit, energy_used=b.cumulative_virtual_energy,
                    energy_balance_residual=CAPACITY*self.job['soc']+self.replenished
                        -b.agent.energy-b.cumulative_virtual_energy,
                    hocbf_interventions=int(b.safety_interventions),
                    hocbf_calls=int(b.safety_filter_calls),
                    hocbf_fallbacks=int(b.safety_fallbacks), exhausted=self.exhausted)

    def step(self, action):
        if self.exhausted:
            # Park without resetting so failure snapshot is the actual end state.
            return self.observation(), 0., False, False, {'parked': True}
        was_returning = self.base.mode is SortieMode.CHARGER_COMMITTED
        obs, reward, _, _, info = super().step(action)
        b = self.base
        row = info.pop('sortie', None)
        returned = bool(row and row['returned'])
        self.exhausted = bool(b.agent.energy <= 1e-9)
        cycle = None
        if returned:
            cycle = dict(index=self.recharges, tasks=row['tasks_completed'],
                         arrival_energy=row['remaining_energy'], commit=self.commit,
                         collision_count=self.contacts-self.cycle_contacts_start)
            self.replenished += CAPACITY-row['remaining_energy']
            self.recharges += 1
            self.return_steps = 0
            self.commit = None
            self.cycle_contacts_start = self.contacts
        gain = b.tasks_completed-self.last_tasks
        self.last_tasks = b.tasks_completed
        info.update(task_gain=gain, recharge_event=returned, cycle=cycle,
                    commit_event=not was_returning and bool(action[3]>0),
                    exhausted=self.exhausted,
                    clock_warning=(self.return_steps==4000 or b.steps_in_current_task==4000),
                    diagnostic=self.diagnostic())
        telemetry=b.trajectory_log[-1]
        info['executed_action']=telemetry['executed_action'].copy()
        info['hocbf_intervened']=bool(telemetry['hocbf_intervened'])
        # Bound passive visualization logs; physics, RNG and pending goal labels untouched.
        if self.total_steps % 512 == 0:
            del b.agent_paths[0][:-512]
            del b.trajectory_log[:-512]
        # Depletion is handled by the outer runner; never VecEnv-autoreset failure.
        return self.observation(), reward, False, False, info
