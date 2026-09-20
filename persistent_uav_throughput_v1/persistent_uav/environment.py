"""PersistentUAVThroughput-v1: resumable event state machine, no hidden safety mask."""
from dataclasses import asdict
import math

import numpy as np

from .baselines import Action
from .config import ceil_grid
from .streams import stream_hash


class PersistentUAVThroughput:
    def __init__(self, config, navigator, tasks):
        self.config, self.nav = config, navigator
        self._tasks = tuple(tasks)  # private: never included in policy observations
        if list(self._tasks) != sorted(self._tasks, key=lambda t: (t.arrival, t.id)):
            raise ValueError('task stream must be sorted')
        if len({t.id for t in self._tasks}) != len(self._tasks):
            raise ValueError('task IDs must be unique')
        if sum(t.arrival == 0. for t in self._tasks) != config.initial_tasks:
            raise ValueError('initial task count does not match declared configuration')
        if any(t.arrival < 0 or t.arrival > config.cutoff + 1e-8 for t in self._tasks):
            raise ValueError('arrival outside evaluation window')
        self.workload_hash = stream_hash(self._tasks)
        self.cursor = 0
        self.queue = []
        self.current_task = None
        self.mode = 'IDLE'
        self.completed = 0
        self.failure = None
        self.failure_time = None
        self.recharge_failure = False
        self.cutoff_mode = None
        self.done = False
        self.arrivals = self.accepted = self.overflow = 0
        self.initial_accepted = 0
        self.recharge_count = 0
        self.completed_recharges = 0
        self.estimate_fallbacks = 0
        self.time_by_mode = dict(flight=0., charging=0., waiting=0., failed=0.)
        self.queue_time = [0.] * (config.queue_capacity + 1)
        self._queue_clock = 0.
        self.events = []
        self.latencies = []
        self.task_start_contacts = 0
        self.collision_free_completions = 0
        self._arrive_until(0.)

    @property
    def time(self):
        return self.nav.time

    @property
    def decision_required(self):
        at_full_station = (np.linalg.norm(self.nav.position - self.nav.station) <= self.nav.goal_radius
                           and self.nav.energy >= self.config.capacity - 1e-9)
        return not self.done and self.mode == 'IDLE' and (bool(self.queue) or not at_full_station)

    @property
    def can_recharge(self):
        at_station = np.linalg.norm(self.nav.position - self.nav.station) <= self.nav.goal_radius
        return bool(self.decision_required and not (at_station and self.nav.energy >= self.config.capacity - 1e-9))

    def observe(self):
        # No future cursor, layout, oracle cost, random state or stream seed.
        return dict(position=self.nav.position.tolist(), velocity=self.nav.velocity.tolist(),
                    battery=self.nav.energy, capacity=self.config.capacity,
                    time=self.time, remaining_time=max(0., self.config.cutoff - self.time),
                    queue=[asdict(t) for t in self.queue],
                    current_task=None if self.current_task is None else asdict(self.current_task),
                    charger_position=self.nav.station.tolist(), mode=self.mode,
                    decision_required=self.decision_required, can_recharge=self.can_recharge,
                    completed=self.completed)

    def _event(self, kind, **fields):
        self.events.append(dict(time=self.time, event=kind, **fields))

    def _queue_integrate(self, time):
        if time < self._queue_clock - 1e-7:
            raise RuntimeError('queue accounting clock moved backwards')
        self.queue_time[len(self.queue)] += max(0., time - self._queue_clock)
        self._queue_clock = float(time)

    def _arrive_until(self, time, *, include_endpoint=True):
        while self.cursor < len(self._tasks):
            arrival = self._tasks[self.cursor].arrival
            if not (arrival <= time + 1e-8 if include_endpoint else arrival < time - 1e-8):
                break
            task = self._tasks[self.cursor]
            self._queue_integrate(task.arrival)
            self.cursor += 1
            initial = task.original_arrival == 0.
            self.arrivals += int(not initial)
            accepted = len(self.queue) < self.config.queue_capacity
            if accepted:
                self.queue.append(task)
                self.accepted += int(not initial)
                self.initial_accepted += int(initial)
            else:
                self.overflow += int(not initial)
            # Arrival logs have their own timestamp, not a fictitious old position.
            self.events.append(dict(time=task.arrival, event='arrival', task_id=task.id,
                                    original_arrival=task.original_arrival, accepted=accepted,
                                    initial=initial, queue_length=len(self.queue)))
        self._queue_integrate(time)

    def _start_action(self, action):
        if not isinstance(action, Action) or not self.decision_required:
            raise ValueError('an action is legal only at a decision point')
        before = self.observe()
        if action.kind == 'serve':
            candidates = [t for t in self.queue if t.id == action.task_id]
            if not candidates:
                raise ValueError('task not in waiting queue')
            self.current_task = candidates[0]
            self.queue.remove(self.current_task)
            self.nav.start_leg(self.current_task.position)
            self.task_start_contacts = self.nav.contacts
            self.mode = 'SERVING'
            self.estimate_fallbacks += int(action.reason == 'full_station_infeasible_estimate_fallback')
        elif action.kind == 'recharge' and action.task_id is None:
            if not self.can_recharge:
                raise ValueError('zero-time full-station recharge is illegal')
            self.nav.start_leg(self.nav.station)
            self.mode = 'RETURNING'
            self.recharge_count += 1
        elif action.kind == 'idle' and action.task_id is None and not self.queue:
            self.mode = 'WAITING'
        else:
            raise ValueError('serve/recharge with backlog; recharge/idle when empty')
        self._event('decision', observation=before, action=asdict(action))

    def _fail(self, reason):
        self.failure, self.failure_time = reason, self.time
        self.recharge_failure = self.mode in ('RETURNING', 'CHARGING')
        self.mode = 'FAILED'
        self._event('failure', reason=reason, recharge_failure=self.recharge_failure,
                    residual_battery=self.nav.energy)
        # Fixed-T score includes failures. Arrivals continue, service never resumes.
        self.time_by_mode['failed'] += max(0., self.config.cutoff - self.time)
        self.nav.absorb_until(self.config.cutoff)
        self._arrive_until(self.time)
        self.done = True

    def _arrived_at_goal(self):
        self.nav.stop_at_service()
        if self.mode == 'SERVING':
            self.completed += 1
            self.latencies.append(self.time - self.current_task.original_arrival)
            collision_free = self.nav.contacts == self.task_start_contacts
            self.collision_free_completions += int(collision_free)
            self._event('task_completed', task_id=self.current_task.id, collision_free=collision_free,
                        position=self.nav.position.tolist(), battery=self.nav.energy)
            self.current_task = None
            self.mode = 'IDLE'
        elif self.mode == 'RETURNING':
            self.mode = 'CHARGING'
            self._event('charger_arrival', battery=self.nav.energy)
        else:
            raise RuntimeError('unexpected goal arrival')

    def step(self, action=None):
        """One flight policy step or stationary event interval; checkpoint at any return."""
        if self.done:
            raise RuntimeError('episode already ended; no implicit reset')
        completed_before = self.completed
        if self.time >= self.config.cutoff - 1e-8:
            self._cutoff()
            return self.observe(), 0, True
        if self.decision_required:
            if action is None:
                raise ValueError('scheduler action required')
            self._start_action(action)
        elif action is not None:
            raise ValueError('cannot act during a committed option or empty-queue wait')
        if self.mode == 'WAITING' or (self.mode == 'IDLE' and not self.queue):
            self.mode = 'WAITING'
            next_arrival = (self._tasks[self.cursor].arrival if self.cursor < len(self._tasks)
                            else self.config.cutoff)
            duration = min(next_arrival, self.config.cutoff) - self.time
            outcome = self.nav.advance_stationary(duration)
            self.time_by_mode['waiting'] += outcome.duration
            self._arrive_until(self.time)
            if outcome.depleted:
                self._fail('energy_depletion')
            else:
                self.mode = 'IDLE'
        elif self.mode in ('SERVING', 'RETURNING'):
            if self.nav.reached():
                self._arrived_at_goal()
                self._arrive_until(self.time)
            else:
                outcome = self.nav.advance_flight(self.config.cutoff - self.time)
                self.time_by_mode['flight'] += outcome.duration
                # Queue events within the flight step precede its endpoint decision.
                self._arrive_until(self.time, include_endpoint=False)
                if outcome.depleted:
                    self._fail('energy_depletion')
                elif outcome.reached:
                    self._arrived_at_goal()
                elif outcome.timeout:
                    self._fail('navigation_failure')
                if not self.done:
                    self._arrive_until(self.time)
        elif self.mode == 'CHARGING':
            duration = min(self.config.cutoff - self.time,
                           ceil_grid(max(0., self.config.capacity - self.nav.energy) / self.config.recharge_rate))
            outcome = self.nav.advance_stationary(duration, recharge_rate=self.config.recharge_rate)
            self.time_by_mode['charging'] += outcome.duration
            self._arrive_until(self.time, include_endpoint=False)
            if self.nav.energy >= self.config.capacity - 1e-9:
                self.completed_recharges += 1
                self.mode = 'IDLE'
                self._event('charge_complete', battery=self.nav.energy)
            self._arrive_until(self.time)
        if not self.done and self.time >= self.config.cutoff - 1e-8:
            self._cutoff()
        self._assert_invariants()
        return self.observe(), self.completed - completed_before, self.done

    def _cutoff(self):
        self._arrive_until(self.config.cutoff)
        self.cutoff_mode = self.mode
        self.mode = 'CUTOFF'
        self.done = True
        self._event('evaluation_cutoff', interrupted_mode=self.cutoff_mode)

    def _assert_invariants(self):
        if len(self.queue) > self.config.queue_capacity:
            raise RuntimeError('queue exceeded capacity')
        if not -1e-8 <= self.nav.energy <= self.config.capacity + 1e-8:
            raise RuntimeError('invalid battery')
        if self.time > self.config.cutoff + 1e-7 or not math.isfinite(self.time):
            raise RuntimeError('invalid time')
        if self.arrivals != self.accepted + self.overflow:
            raise RuntimeError('arrival conservation failed')
        active = int(self.current_task is not None)
        if self.initial_accepted + self.accepted != self.completed + len(self.queue) + active:
            raise RuntimeError('task conservation failed')

    def summary(self):
        self._assert_invariants()
        return dict(completed=self.completed, evaluation_horizon=self.config.cutoff,
                    simulation_time=self.time, done=self.done, stream_sha256=self.workload_hash,
                    depletion=self.failure == 'energy_depletion', failure=self.failure,
                    failure_time=self.failure_time, navigation_failure=self.failure == 'navigation_failure',
                    recharge_failure=self.recharge_failure,
                    recharge_censored=self.cutoff_mode in ('RETURNING', 'CHARGING'),
                    terminal_residual_battery=self.nav.energy, collision_count=self.nav.contacts,
                    collision_free_completions=self.collision_free_completions,
                    arrivals=self.arrivals, accepted=self.accepted, overflow=self.overflow,
                    overflow_rate=self.overflow / self.arrivals if self.arrivals else None,
                    queue_time=self.queue_time, empty_fraction=self.queue_time[0] / self.config.cutoff,
                    time_by_mode=dict(self.time_by_mode), recharge_attempts=self.recharge_count,
                    completed_recharges=self.completed_recharges,
                    mean_completed_task_latency=float(np.mean(self.latencies)) if self.latencies else None,
                    estimate_infeasible_fallbacks=self.estimate_fallbacks, physical_resets=self.nav.reset_count,
                    returnability='not_an_admissibility_constraint; diagnostic_not_run')
