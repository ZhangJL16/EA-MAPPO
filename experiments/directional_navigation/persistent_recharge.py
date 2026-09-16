"""Finite-time repeated recharge evaluation on the unchanged recovery plant."""
from __future__ import annotations

import math
import numpy as np

from experiments.directional_navigation.battery_sortie import BatterySortie, CAPACITY
from review_bundle.safety.switching.commitment import SortieMode

METHODS = ('soc20', 'soc40', 'repaired95')


def persistent_jobs(worlds: int = 60) -> list[dict]:
    if not 1 <= worlds <= 60:
        raise ValueError('use 1..60 registered development worlds')
    return [dict(job_id=i*6+s*3+m, world_seed=1100000001+i,
                 head_seed=0, method=m, method_name=METHODS[m], soc=soc)
            for i in range(worlds) for s, soc in enumerate((1.0, 0.2))
            for m in range(len(METHODS))]


class PersistentRecharge(BatterySortie):
    """Return is a service event; contacts never terminate the physical run.

    Historical per-task and per-return deadlines remain 4000 policy steps.
    Failed/time-limited runs remain in the fixed-time throughput denominator.
    BatterySortie.step supplies the original reward/contact handling unchanged.
    """

    def __init__(self, jobs: list[dict], duration_seconds: float = 3600.0):
        if not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError('positive finite evaluation duration required')
        self.duration_seconds = float(duration_seconds)
        # Every real step advances at least one 0.05s substep. This guard cannot
        # expire before the evaluation clock, including frequent service events.
        super().__init__(jobs, guard=math.ceil(duration_seconds/0.05)+2)
        if self.base.physics_dt != 0.05:
            raise ValueError('unexpected physics clock')
        self.run_finished = False
        self.cycles: list[dict] = []
        self.cycle_contacts_start = 0
        self.recharged_energy = 0.0
        self.first_contact_step: int | None = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.run_finished = False
        self.cycles = []
        self.cycle_contacts_start = 0
        self.recharged_energy = 0.0
        self.first_contact_step = None
        self.last_tasks = 0
        return obs, info

    def step(self, action):
        if self.run_finished:
            raise RuntimeError('reset required after run ends')
        if self.inactive:
            return super().step(action)
        was_returning = self.base.mode is SortieMode.CHARGER_COMMITTED
        _, reward, old_done, _, info = super().step(action)
        b = self.base
        if info['cost'] and self.first_contact_step is None:
            self.first_contact_step = self.total_steps
        tasks = int(b.tasks_completed)
        task_gain = tasks-self.last_tasks
        self.last_tasks = tasks
        old_row = info.pop('sortie', None)
        returned = bool(old_row is not None and old_row['returned'])
        committed = not was_returning and self.commit is not None
        cycle = None
        if returned:
            cycle = dict(cycle_id=len(self.cycles), end_step=self.total_steps,
                         end_seconds=float(b.simulation_time),
                         tasks_completed=int(old_row['tasks_completed']),
                         collision_count=self.contacts-self.cycle_contacts_start,
                         arrival_energy=float(old_row['remaining_energy']),
                         commit=self.commit)
            cycle['safe_return'] = cycle['collision_count'] == 0
            cycle['zero_task_return'] = cycle['tasks_completed'] == 0
            self.cycles.append(cycle)
            self.recharged_energy += CAPACITY-cycle['arrival_energy']
            self.cycle_contacts_start = self.contacts
            # The plant already serviced and selected a task: never reset it.
            self.return_steps = 0
            self.commit = None
        time_complete = b.simulation_time+1e-9 >= self.duration_seconds
        failure = bool(old_done and not returned)
        reason = old_row['termination'] if failure else 'evaluation_time' if time_complete else None
        self.run_finished = failure or time_complete
        info.update(task_gain=task_gain, recharge_event=returned,
                    commit_event=committed, cycle=cycle,
                    step=self.total_steps, simulation_seconds=float(b.simulation_time))
        if self.run_finished:
            available = CAPACITY*self.job['soc']+self.recharged_energy
            info['persistent_run'] = dict(
                **self.job, termination=reason, time_complete=bool(time_complete),
                evaluation_seconds=self.duration_seconds,
                simulation_seconds=float(b.simulation_time), steps=self.total_steps,
                time_overshoot=max(0., float(b.simulation_time)-self.duration_seconds),
                tasks_completed=tasks, tasks_per_simulated_hour=tasks*3600/self.duration_seconds,
                collision_count=self.contacts, first_contact_step=self.first_contact_step,
                energy_exhausted=reason == 'energy_exhausted',
                censored=reason in ('task_step_limit', 'return_deadline', 'episode_emergency_step_guard'),
                initial_energy=CAPACITY*self.job['soc'], remaining_energy=float(b.agent.energy),
                replenished_energy=self.recharged_energy, energy_used=float(b.cumulative_virtual_energy),
                energy_balance_residual=available-float(b.agent.energy)-float(b.cumulative_virtual_energy),
                recharge_count=len(self.cycles), cycles=self.cycles,
                # Being alive at T does not certify that eventual return is feasible.
                alive_at_time_limit=time_complete and not failure,
                mode_at_end=b.mode.value, return_steps=self.return_steps)
        return self.observation(), reward, failure, time_complete and not failure, info


def continuing_targets(task_gain: np.ndarray, next_task_value: np.ndarray,
                       first_failure: np.ndarray, next_failure_probability: np.ndarray,
                       bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Undiscounted finite-time targets; recharge is deliberately NOT a mask.

    This is a target-semantics reference, not an implemented learning algorithm.
    bootstrap is false at the fixed horizon or depletion. Administrative
    censoring is excluded by the caller, not silently assigned zero risk.
    A first-contact analysis label may absorb risk, but must not stop utility.
    """
    arrays = [np.asarray(x) for x in (task_gain, next_task_value, first_failure,
                                    next_failure_probability, bootstrap)]
    g, v, d, r, m = arrays
    if (g.ndim != 1 or any(x.shape != g.shape for x in arrays)
            or any(not np.isfinite(x).all() for x in arrays)
            or np.any(g < 0) or np.any(v < 0)
            or np.any((r < 0) | (r > 1))
            or d.dtype != np.bool_ or m.dtype != np.bool_):
        raise ValueError('aligned finite targets with boolean event/bootstrap masks required')
    return g+m*v, d.astype(float)+(~d)*m*r
