"""P1.1: post-delivery decisions with no task screening.

Inherit the unchanged P1 navigation/station mechanics. The historical P1 class
and its source-bound evidence remain intact; its execute() is never used here.
"""
from dataclasses import asdict

import numpy as np

from research.regenerative_control.continuing import ContinuingUAV


class PostDeliveryUAV(ContinuingUAV):
    def __init__(self, *args, **kwargs):
        self.task_draw_count = 0
        super().__init__(*args, **kwargs)
        del self.abandoned  # no task rejection in this process
        # The inherited constructor's legacy hook was suppressed below.
        self.events.clear()
        self._event('initial_charged_state', regeneration=False)

    def _new_task(self):
        """Suppress legacy draw-after-reset/recharge; expose a forced H-state."""
        if self.task is not None:
            raise RuntimeError('H-state cannot carry a task')
        self.phase = 'home'

    def state(self, *, full=False):
        return dict(super().state(full=full), task_draw_count=self.task_draw_count,
                    semantics='P1.1_post_delivery')

    def _draw_task(self):
        if self.task is not None:
            raise RuntimeError('cannot replace an existing task')
        index = int(self.task_rng.integers(3))
        d = (100, 300, 600)[index]
        home = self.base.charger_position
        self.task = dict(index=index, value=1., pickup=(home-[d,0,0]).tolist(),
                         dropoff=(home+[-d,d,0]).tolist())
        self.task_draw_count += 1
        self.phase = 'task_execution'
        self._event('task_draw')

    def _check_epoch(self, phase):
        if (self.phase != phase or self.task is not None or self.failed
                or self.truncated or self.time >= self.config.cutoff):
            raise ValueError(f'operation requires a live task-free {phase} state')

    def advance_home(self, actor):
        """Mandatory H→first task→D transition; no policy action at home."""
        self._check_epoch('home')
        return self._transition('forced_task', actor)

    def execute(self, action, actor):
        self._check_epoch('decision')
        if action not in ('C', 'R'):
            raise ValueError('D-state supports only C or R')
        return self._transition(action, actor)

    def _transition(self, action, actor):
        start = self.state(full=True)
        used = self.base.cumulative_virtual_energy+self.service_energy
        count, completed = self.contacts, self.completed
        self.macro_count += 1
        self._event('forced_first_task' if action=='forced_task' else 'decision_'+action)
        legs=[]
        if action in ('C', 'forced_task'):
            self._draw_task()  # only after committing to C / mandatory H departure
            legs.append(self._navigate(np.array(self.task['pickup']),actor,'pickup'))
            if not (self.failed or self.truncated):
                legs.append(self._navigate(np.array(self.task['dropoff']),actor,'dropoff'))
                if legs[-1]['goal_reached'] and self.time < self.config.cutoff:
                    self.completed += 1
                    delivered_task = self.task.copy()
                    self.task = None
                    self.phase = 'decision'
                    self._event('delivery_complete', delivered_task=delivered_task)
        else:
            # No task draw, discard, RNG advance or abandonment counter.
            legs.append(self._navigate(self.base.charger_position,actor,'return'))
            if not (self.failed or self.truncated):
                self._station_service()  # unchanged physics/service, ends in H
        return dict(action=action, start_state=start, next_macro_state=self.state(full=True),
            start_energy=start['energy'], end_energy=self.energy,
            elapsed_time=self.time-start['time'],
            energy_used=self.base.cumulative_virtual_energy+self.service_energy-used,
            reward=self.completed-completed, collision_count=self.contacts-count,
            collision=self.contacts>count,
            goal_reached=(legs[-1]['goal_reached'] if action=='R' else self.completed>completed),
            success=not (self.failed or self.truncated),
            timeout=self.failure_reason=='navigation_timeout', censored=self.truncated,
            terminal_reason=self.failure_reason or ('evaluation_cutoff' if self.truncated else 'completed'),
            legs=legs)

    def audit(self):
        b=self.base
        return dict(reset_calls=b.reset_calls, elapsed_time=self.time,
            completed=self.completed, task_draw_count=self.task_draw_count,
            collision_count=self.contacts, energy_failure=self.failure_reason=='energy_depletion',
            failed=self.failed, terminal_reason=self.failure_reason,
            evaluation_cutoff=self.config.cutoff, truncated=self.truncated,
            cycle_count=len(self.cycles),
            energy_balance_residual=self.config.capacity+self.replenished-self.energy
                -b.cumulative_virtual_energy-self.service_energy,
            throughput_at_cutoff=self.completed/self.config.cutoff,
            config=asdict(self.config), map_id=self.map_hash,
            semantics='P1.1_post_delivery', task_rejections=0)
