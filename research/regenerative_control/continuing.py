"""P1 continuing single-agent plant; no energy predictor or learned high-level policy.

Physical navigation is inherited unchanged. The outer process owns task phases,
finite battery, paid station service, and the measurement cutoff. Source NAVIGATION
mode is kept only to suppress the legacy automatic task/charger service branch.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math

import gymnasium as gym
import numpy as np
import torch

from experiments.directional_navigation.recovery import LockedRecoveryPlant
from experiments.directional_navigation.correction_supervision import TeacherConfig


@dataclass(frozen=True)
class Config:
    capacity: float = 60.0  # existing synthetic telemetry units, not Wh
    cutoff: float = 50_000.0  # simulator seconds; measurement only
    option_step_limit: int = 4000
    obstacles: int = 24
    recharge_overhead: float = 10.0
    recharge_rate: float = 1.0
    docking_speed: float = 1.0
    docking_power: float = 0.02

    def __post_init__(self):
        for name in ('capacity', 'cutoff', 'recharge_overhead', 'recharge_rate', 'docking_speed', 'docking_power'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if self.option_step_limit <= 0 or self.obstacles < 0:
            raise ValueError('invalid option limit or obstacle count')


class MacroPlant(LockedRecoveryPlant):
    @property
    def finite_energy_enabled(self):
        # Enables inherited substep depletion without legacy free recharge.
        return True

    @property
    def episode_policy_step_limit(self):
        return math.inf  # no administrative episode horizon inside the plant

    def _initialize_static_obstacles(self, excluded_positions):
        home = self.charger_position
        endpoints = [home - [d, 0, 0] for d in (100, 300, 600)]
        endpoints += [home + [-d, d, 0] for d in (100, 300, 600)]
        super()._initialize_static_obstacles(excluded_positions + endpoints)

    def reset(self, **kwargs):
        self.reset_calls = getattr(self, 'reset_calls', 0) + 1
        return super().reset(**kwargs)


class ContinuingUAV:
    """C accepts the whole pickup→dropoff job. R discards it and recharges.

    Regeneration is conditional on one fixed layout. The task stream is IID,
    always available, uniform over three known endpoint pairs with value one.
    No action is requested within a task. Catastrophic depletion/navigation
    timeout stops operation; there is no rescue/reset that could reward failure.
    """
    def __init__(self, config=Config(), *, layout_seed=319190001, stream_seed=419190001):
        self.config = config
        self.layout_seed, self.stream_seed = int(layout_seed), int(stream_seed)
        self.task_rng = np.random.default_rng(stream_seed)
        self.base = MacroPlant(lidar_enabled=True, lidar_horizontal_sectors=128,
            lidar_vertical_sectors=8, num_obstacles=config.obstacles,
            cbf_enabled=False, projection_geometry_enabled=False,
            max_steps_per_task=config.option_step_limit)
        self.base.configure_calibrated_battery(config.capacity, reserve_fraction=0,
                                               source='P1_declared_synthetic_capacity')
        self.base.mission_switching_enabled = False
        self.base.energy_learning_enabled = False
        self.base.cbf_enabled = True
        self.base.safety_filter = TeacherConfig().make_filter()
        self.base.safety_intervention_penalty = 0.0
        home = self.base.charger_position
        self.base.reset(seed=layout_seed, options={'start_position': home,
                        'task_point': home - [100, 0, 0]})
        self.map_hash = hashlib.sha256(json.dumps(self.base.static_obstacle_layout(),
                                   sort_keys=True).encode()).hexdigest()
        space = self.base.observation_space
        self.observation_space = gym.spaces.Box(np.append(space.low, 0).astype(np.float32),
                                               np.append(space.high, 1).astype(np.float32))
        self.action_space = self.base.action_space
        self.phase = 'decision'
        self.task = None
        self.completed = self.abandoned = self.contacts = self.macro_count = 0
        self.failed = self.truncated = False
        self.failure_reason = None
        self.events, self.cycles = [], []
        self.replenished = self.service_energy = 0.0
        self.cycle_start_time = 0.0
        self.cycle_start_tasks = 0
        self._event('initial_charged_state', regeneration=False)
        self._new_task()

    @property
    def time(self):
        return float(self.base.simulation_time)

    @property
    def energy(self):
        return float(self.base.agent.energy)

    def _new_task(self):
        index = int(self.task_rng.integers(3))
        d = (100, 300, 600)[index]
        home = self.base.charger_position
        self.task = dict(index=index, value=1.0, pickup=(home-[d, 0, 0]).tolist(),
                         dropoff=(home+[-d, d, 0]).tolist())
        self.phase = 'decision'
        self._event('task_available')

    def state(self, *, full=False):
        b = self.base
        result = dict(position=b.agent.pos.tolist(), velocity=b.agent.vel.tolist(),
            energy=self.energy, time=self.time, phase=self.phase, task=copy.deepcopy(self.task),
            map_id=self.map_hash, previous_contact=bool(b.agent.collided),
            completed=self.completed, collision_count=self.contacts,
            failed=self.failed, truncated=self.truncated)
        if full:
            # Privileged audit/possible oracle state; NEVER an actor observation.
            result.update(map=b.static_obstacle_layout(),
                task_rng=copy.deepcopy(self.task_rng.bit_generator.state),
                plant_rng=copy.deepcopy(b.np_random.bit_generator.state),
                plant_steps=int(b.current_step), reset_calls=b.reset_calls)
        return result

    def _event(self, kind, **extra):
        self.events.append(dict(event=kind, **self.state(), reset_calls=self.base.reset_calls, **extra))

    def _fail(self, reason):
        self.failed = True
        self.failure_reason = reason
        self.phase = 'failed'
        self._event(reason)

    def _set_goal(self, goal):
        b = self.base
        b.current_task_point = np.asarray(goal, np.float32).copy()
        b.agent.goal = b.current_task_point.copy()
        b.agent.reached = False
        b.steps_in_current_task = 0
        b._start_goal_trajectory(b.current_task_point)
        # Contact memory, velocity, battery, position and global clock persist.

    def nav_observation(self):
        return np.append(self.base.sac_observation_for_goal(self.base.active_goal),
            max(0., 1-self.base.steps_in_current_task/self.config.option_step_limit)).astype(np.float32)

    def _navigate(self, goal, actor, leg):
        self.phase = leg
        start = self.state(full=True)
        self._set_goal(goal)
        used_before = self.base.cumulative_virtual_energy
        count_before = self.contacts
        reached = False
        if np.linalg.norm(self.base.agent.pos - goal) <= self.base.goal_tolerance:
            reached = True
        while not reached and not (self.failed or self.truncated):
            if self.time >= self.config.cutoff:
                self.truncated = True
                break
            obs = self.nav_observation()
            if not np.isfinite(obs).all():
                raise RuntimeError('nonfinite actor observation')
            with torch.inference_mode():
                action = actor(torch.as_tensor(obs[None]), deterministic=True)[0].cpu().numpy()
            original_substeps = self.base.physics_substeps_per_policy_step
            remaining_steps = int((self.config.cutoff-self.time+1e-10)/self.base.physics_dt)
            if remaining_steps < 1:
                self.truncated = True
                break
            self.base.physics_substeps_per_policy_step = min(original_substeps, remaining_steps)
            try:
                _, _, _, timed_out, info = self.base.step(action)
            finally:
                self.base.physics_substeps_per_policy_step = original_substeps
            contact = bool(info['boundary_contact'] or info['obstacle_collision'])
            self.contacts += int(contact)
            reached = bool(info['is_success'])
            if self.energy <= 1e-10:
                self._fail('energy_depletion')
                reached = False
            elif timed_out:
                self._fail('navigation_timeout')
            if self.time >= self.config.cutoff:
                self.truncated = True
        if reached:
            self._event(leg+'_reached')
        result = dict(option=leg, start_state=start, end_state=self.state(full=True),
            start_energy=start['energy'], end_energy=self.energy,
            energy_used=float(self.base.cumulative_virtual_energy-used_before),
            elapsed_time=self.time-start['time'], goal_reached=reached,
            collision_count=self.contacts-count_before, collision=self.contacts>count_before,
            timeout=self.failure_reason=='navigation_timeout',
            censored=self.truncated and not reached,
            terminal_reason=self.failure_reason or ('goal_reached' if reached else 'evaluation_cutoff'))
        return result

    def _station_service(self):
        """Paid station capture within goal radius, then charging; not env.reset.

        Capture is an explicit ground-service actuator with bounded linear motion,
        speed 1 distance-unit/s, power .02. This is NOT an extra SAC flight policy.
        """
        b, c = self.base, self.config
        distance = float(np.linalg.norm(b.agent.pos-b.charger_position))
        if distance > b.goal_tolerance + 1e-6:
            raise RuntimeError('cannot recharge away from charger')
        self.phase = 'docking'
        self._event('docking_start', docking_distance=distance)
        start = b.agent.pos.copy()
        duration = distance/c.docking_speed
        elapsed = 0.0
        while elapsed < duration - 1e-12:
            dt = min(b.physics_dt, duration-elapsed, c.cutoff-self.time)
            if dt <= 1e-12:
                self.truncated = True
                return
            elapsed += dt
            point = start + (b.charger_position-start)*(elapsed/duration)
            if not b._position_clear_of_obstacles(point):
                raise RuntimeError('station capture corridor obstructed')
            b.agent.pos = point.astype(np.float32)
            b.agent.prev_pos = b.agent.pos.copy()
            b.agent.last_pos = b.agent.pos.copy()
            b.agent.vel = ((b.charger_position-start)/duration).astype(np.float32)
            cost = c.docking_power*dt
            b.agent.energy = max(0., self.energy-cost)
            self.service_energy += cost
            b.simulation_time += dt
            if self.energy <= 1e-10:
                self._fail('energy_depletion')
                return
        b.agent.pos = b.charger_position.copy()
        b.agent.prev_pos = b.agent.pos.copy()
        b.agent.last_pos = b.agent.pos.copy()
        b.agent.vel[:] = 0
        self.phase = 'recharging'
        self._event('recharge_start')
        overhead = min(c.recharge_overhead, max(0., c.cutoff-self.time))
        b.simulation_time += overhead
        if overhead < c.recharge_overhead:
            self.truncated = True
            return
        needed = (c.capacity-self.energy)/c.recharge_rate
        duration = min(needed, max(0., c.cutoff-self.time))
        added = duration*c.recharge_rate
        b.agent.energy += added
        self.replenished += added
        b.simulation_time += duration
        if duration < needed:
            self.truncated = True
            return
        b.agent.energy = c.capacity
        # Contact memory is not erased by service: perform one real clean policy
        # step at home. Its cost/time are then paid back at the charging rate.
        self._set_goal(b.charger_position)
        if c.cutoff-self.time < b.physics_dt:
            self.truncated = True
            return
        before_energy = self.energy
        _, _, _, _, info = b.step(np.zeros(3, np.float32))
        contact = bool(info['boundary_contact'] or info['obstacle_collision'])
        self.contacts += int(contact)
        if contact:
            raise RuntimeError('canonical station state has contact')
        cost = before_energy-self.energy
        duration = min(cost/c.recharge_rate, max(0., c.cutoff-self.time))
        b.simulation_time += duration
        self.replenished += duration*c.recharge_rate
        b.agent.energy += duration*c.recharge_rate
        if duration < cost/c.recharge_rate:
            self.truncated = True
            return
        b.agent.energy = c.capacity
        b.agent.vel[:] = 0
        b._update_lidar()
        self.task = None
        self.phase = 'regenerated'
        self.cycles.append(dict(start=self.cycle_start_time, end=self.time,
            duration=self.time-self.cycle_start_time,
            task_value=self.completed-self.cycle_start_tasks,
            initial_delayed_cycle=len(self.cycles)==0))
        self.cycle_start_time, self.cycle_start_tasks = self.time, self.completed
        self._event('recharge_complete', regeneration=True)
        self._new_task()  # new independent draw, never reseed/replay the RNG
        self.truncated = self.time >= c.cutoff

    def execute(self, action, actor):
        if action not in ('C', 'R') or self.phase != 'decision' or self.failed or self.truncated:
            raise ValueError('C/R only at live pre-task decision epochs')
        if self.time >= self.config.cutoff:
            raise ValueError('measurement cutoff already reached')
        start = self.state(full=True)
        used = self.base.cumulative_virtual_energy + self.service_energy
        count = self.contacts
        completed = self.completed
        self.macro_count += 1
        self._event('decision_'+action)
        legs=[]
        if action == 'C':
            legs.append(self._navigate(np.array(self.task['pickup']), actor, 'pickup'))
            if not (self.failed or self.truncated):
                legs.append(self._navigate(np.array(self.task['dropoff']), actor, 'dropoff'))
                if legs[-1]['goal_reached'] and self.time < self.config.cutoff:
                    self.completed += 1
                    self._event('delivery_complete')
                    self.task = None
                    if not self.truncated:
                        self._new_task()
        else:
            self.abandoned += 1  # offered but unaccepted task is discarded
            self.task = None
            legs.append(self._navigate(self.base.charger_position, actor, 'return'))
            if not (self.failed or self.truncated):
                self._station_service()
        return dict(action=action, start_state=start, next_macro_state=self.state(full=True),
            start_energy=start['energy'], end_energy=self.energy,
            elapsed_time=self.time-start['time'],
            energy_used=self.base.cumulative_virtual_energy+self.service_energy-used,
            reward=self.completed-completed, collision_count=self.contacts-count,
            collision=self.contacts>count,
            goal_reached=(self.completed>completed if action=='C' else legs[-1]['goal_reached']),
            success=not (self.failed or self.truncated),
            timeout=self.failure_reason=='navigation_timeout', censored=self.truncated,
            terminal_reason=self.failure_reason or ('evaluation_cutoff' if self.truncated else 'completed'),
            legs=legs)

    def audit(self):
        b=self.base
        return dict(reset_calls=b.reset_calls, elapsed_time=self.time,
            completed=self.completed, abandoned=self.abandoned, collision_count=self.contacts,
            energy_failure=self.failure_reason=='energy_depletion', failed=self.failed,
            terminal_reason=self.failure_reason, evaluation_cutoff=self.config.cutoff,
            truncated=self.truncated, cycle_count=len(self.cycles),
            energy_balance_residual=self.config.capacity+self.replenished-self.energy
                -b.cumulative_virtual_energy-self.service_energy,
            throughput_at_cutoff=self.completed/self.config.cutoff,
            measured_rate=self.completed/max(self.time,1e-12),
            config=asdict(self.config), map_id=self.map_hash)

    def close(self):
        self.base.close()
