"""B0–B5. Inputs are public observations, never the environment or future stream."""
from dataclasses import dataclass

import numpy as np

from .config import METHODS


@dataclass(frozen=True)
class Action:
    kind: str
    task_id: int | None = None
    reason: str = ''


class Scheduler:
    def __init__(self, method, model, threshold=0.25):
        if method not in METHODS or not 0 <= threshold <= 1:
            raise ValueError('unknown method or invalid threshold')
        self.method, self.model, self.threshold = method, model, float(threshold)

    def choose(self, observation):
        o = observation
        if not o['decision_required']:
            raise ValueError('no policy action at this state')
        if not o['queue']:
            # Shared transparent idle-period rule, not validation tuning.
            if o['can_recharge'] and o['battery'] / o['capacity'] < self.threshold:
                return Action('recharge', reason='empty_queue_soc_threshold')
            return Action('idle', reason='empty_queue_idle')
        rows = []
        for task in o['queue']:
            duration, energy = self.model.predict(o['position'], task['position'])
            _, return_energy = self.model.predict(task['position'], o['charger_position'])
            rows.append(dict(task=task, duration=duration, energy=energy, total=energy + return_energy,
                             distance=float(np.linalg.norm(np.asarray(task['position']) - o['position']))))
        can_charge = o['can_recharge']
        if self.method == 'reserve_sjf':
            feasible = [r for r in rows if r['total'] < o['battery']]
            if feasible:
                row = min(feasible, key=lambda r: (r['duration'], r['task']['id']))
                return Action('serve', row['task']['id'], 'estimated_reserve_feasible')
            if can_charge:
                return Action('recharge', reason='no_estimated_reserve_feasible_task')
            row = min(rows, key=lambda r: (r['total'], r['task']['id']))
            return Action('serve', row['task']['id'], 'full_station_infeasible_estimate_fallback')
        if can_charge and o['battery'] / o['capacity'] < self.threshold:
            return Action('recharge', reason='soc_threshold')
        key = {'fifo': lambda r: (r['task']['arrival'], r['task']['id']),
               'nearest': lambda r: (r['distance'], r['task']['id']),
               'shortest_time': lambda r: (r['duration'], r['task']['id']),
               'threshold_sjf': lambda r: (r['duration'], r['task']['id']),
               'energy_greedy': lambda r: (r['energy'], r['task']['id'])}[self.method]
        return Action('serve', min(rows, key=key)['task']['id'], self.method)
