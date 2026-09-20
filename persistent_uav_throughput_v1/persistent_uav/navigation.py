"""Read-only reuse of locked recovery physics, frozen actor and sensor HOCBF.

The new plant overrides only finite-energy/single-leg orchestration and unused
training telemetry. It never uses the legacy automatic goal draw/free refill.
"""
from dataclasses import dataclass
import hashlib
import io
import math
import os
from pathlib import Path
import sys
import zipfile

LEGACY = Path(os.environ.get('PERSISTENT_UAV_LEGACY_ROOT', '/home/zjl/mappo')).resolve()
for path in (LEGACY, LEGACY / 'runtime_support'):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.sac.policies import Actor

from experiments.directional_navigation.features import DirectionalLidarExtractor
from experiments.directional_navigation.correction_supervision import TeacherConfig
from experiments.directional_navigation.recovery import LockedRecoveryPlant

from .config import MAP_SEED, PHYSICS_DT, ceil_grid

MODEL = LEGACY / 'artifacts/hocbf_correction_sac_20260908_v1/sac/checkpoint_000131072/model.zip'
MODEL_SHA = 'fb79482ae7d832b40137ff1a080ee84912eabb95867e118aca80c846f6ae62be'
_ACTOR = None


class FlightPlant(LockedRecoveryPlant):
    @property
    def finite_energy_enabled(self):
        return True

    @property
    def episode_policy_step_limit(self):
        return math.inf

    def _finalize_goal_trajectory(self, **kwargs):
        # Energy-TD training/evaluation telemetry is unused; cost is accumulated
        # from actual physics in cumulative_virtual_energy, unchanged.
        self._pending_goal_transitions.clear()
        return None


@dataclass(frozen=True)
class StepResult:
    duration: float
    energy_used: float
    reached: bool = False
    depleted: bool = False
    timeout: bool = False
    contact: bool = False
    boundary_penalty: float = 0.
    obstacle_penalty: float = 0.


def actor_for(navigator):
    global _ACTOR
    if _ACTOR is None:
        if hashlib.sha256(MODEL.read_bytes()).hexdigest() != MODEL_SHA:
            raise RuntimeError('frozen SAC checksum mismatch')
        torch.set_num_threads(1)
        extractor = DirectionalLidarExtractor(navigator.observation_space, remaining_time=True)
        actor = Actor(navigator.observation_space, navigator.base.action_space,
                      [256, 256], extractor, extractor.features_dim)
        with zipfile.ZipFile(MODEL) as archive:
            weights = torch.load(io.BytesIO(archive.read('policy.pth')), map_location='cpu', weights_only=True)
        actor.load_state_dict({k[6:]: v for k, v in weights.items() if k.startswith('actor.')}, strict=True)
        _ACTOR = actor.eval().requires_grad_(False)
    return _ACTOR


class FrozenNavigator:
    dt = PHYSICS_DT
    goal_radius = 5.

    def __init__(self, capacity, *, option_step_limit=4000, layout=None, start=None, seed=MAP_SEED):
        self.limit = option_step_limit
        self.base = FlightPlant(
            lidar_enabled=True, lidar_horizontal_sectors=128, lidar_vertical_sectors=8,
            num_obstacles=24, cbf_enabled=False, projection_geometry_enabled=False,
            max_steps_per_task=option_step_limit,
        )
        self.base.configure_calibrated_battery(capacity, reserve_fraction=0,
                                              source='PersistentUAVThroughput-v1')
        self.base.mission_switching_enabled = False
        self.base.energy_learning_enabled = False
        self.base.cbf_enabled = True
        self.base.safety_filter = TeacherConfig().make_filter()
        self.base.safety_intervention_penalty = 0.
        # Fixed-map calibration starts may be physically legal but within the
        # legacy map-generation protection halo (20m). Validate the fixed map
        # at its original station; place the independent pilot start using the
        # physical position validator, not map-generation rejection rules.
        options = dict(start_position=self.base.charger_position)
        if layout is not None:
            options['static_obstacles'] = layout
        self.base.reset(seed=seed, options=options)
        if start is not None:
            position = self.base._validate_position(start, 'calibration_start')
            self.base.agent.pos = position.copy()
            self.base.agent.prev_pos = position.copy()
            self.base.agent.last_pos = position.copy()
            self.base.agent.spawn_pos = position.copy()
            self.base.agent_paths = [[position.copy()]]
            self.base._update_lidar()
        space = self.base.observation_space
        self.observation_space = gym.spaces.Box(np.append(space.low, 0).astype(np.float32),
                                               np.append(space.high, 1).astype(np.float32))
        self.capacity = float(capacity)
        self.layout = self.base.static_obstacle_layout()
        self.contacts = 0
        self.policy_steps = 0
        self.hocbf_fallback_steps = 0
        self.reset_count = 1

    @property
    def time(self):
        return float(self.base.simulation_time)

    @property
    def energy(self):
        return float(self.base.agent.energy)

    @property
    def position(self):
        return self.base.agent.pos.copy()

    @property
    def station(self):
        return self.base.charger_position

    @property
    def velocity(self):
        return self.base.agent.vel.copy()

    def start_leg(self, goal):
        b = self.base
        b.current_task_point = np.asarray(goal, dtype=np.float32).copy()
        b.agent.goal = b.current_task_point.copy()
        b.agent.reached = False
        b.steps_in_current_task = 0
        b._start_goal_trajectory(b.current_task_point)
        # Do not reset position, velocity, time, battery, or last-policy contact.

    def reached(self):
        return bool(np.linalg.norm(self.base.active_goal - self.position) <= self.goal_radius)

    def stop_at_service(self):
        self.base.agent.vel[:] = 0.

    def observation(self):
        return np.append(self.base.sac_observation_for_goal(self.base.active_goal),
                         max(0., 1. - self.base.steps_in_current_task / self.limit)).astype(np.float32)

    def advance_flight(self, max_duration):
        available = int(math.floor((max_duration + 1e-8) / self.dt))
        if available < 1:
            raise ValueError('at least one physics substep required')
        obs = self.observation()
        if not np.isfinite(obs).all():
            raise RuntimeError('nonfinite navigator observation')
        with torch.inference_mode():
            action = actor_for(self)(torch.as_tensor(obs[None]), deterministic=True)[0].cpu().numpy()
        if not np.isfinite(action).all():
            raise RuntimeError('nonfinite frozen action')
        b = self.base
        old_substeps = b.physics_substeps_per_policy_step
        b.physics_substeps_per_policy_step = min(old_substeps, available)
        before = self.time
        try:
            _, _, _, truncated, info = b.step(action)
        finally:
            b.physics_substeps_per_policy_step = old_substeps
        self.policy_steps += 1
        contact = bool(info['boundary_contact'] or info['obstacle_collision'])
        self.contacts += int(contact)
        self.hocbf_fallback_steps += int(info['hocbf_fallback_used'])
        # Base telemetry is not the public episode record; bounded storage keeps
        # mid-flight snapshots small without changing actuator/sensor state.
        b._pending_goal_transitions.clear()
        b.trajectory_log.clear()
        b.agent_paths = [[b.agent.pos.copy()]]
        components = info['reward_components']
        return StepResult(self.time - before, float(info['realized_energy_cost']),
                          reached=bool(info['is_success']), depleted=bool(info['energy_exhausted']),
                          timeout=bool(truncated), contact=contact,
                          boundary_penalty=components['boundary_penalty_component'],
                          obstacle_penalty=components['obstacle_penalty_component'])

    def advance_stationary(self, duration, *, recharge_rate=None):
        if not math.isfinite(duration) or duration < 0:
            raise ValueError('stationary duration must lie on the physics grid')
        # Validate proximity, not ceiling: accumulated flight-clock roundoff
        # can put an exact grid interval a few ulps above its intended value.
        grid_duration = round(duration / self.dt) * self.dt
        if abs(duration - grid_duration) > 1e-7:
            raise ValueError('stationary duration must lie on the physics grid')
        if np.linalg.norm(self.velocity) > 1e-8:
            raise RuntimeError('stationary service cannot brake a moving agent for free')
        actual = float(grid_duration)
        used = 0.
        b = self.base
        if recharge_rate is not None:
            if np.linalg.norm(self.position - self.station) > self.goal_radius:
                raise ValueError('cannot charge away from station')
            b.agent.energy = min(self.capacity, self.energy + recharge_rate * actual)
        elif actual > 0 and np.linalg.norm(self.position - self.station) > self.goal_radius:
            # Away from the dock, stationary waiting is hover. At the dock it
            # consumes no battery; only explicit recharge can increase energy.
            cost = b._realized_energy_cost(np.zeros(3), self.dt, velocity=np.zeros(3))
            if cost > 0:
                depletion_time = ceil_grid(self.energy / cost * self.dt)
                actual = min(actual, depletion_time)
            used = min(self.energy, actual / self.dt * cost)
            b.agent.energy = max(0., self.energy - used)
            b.cumulative_virtual_energy += used
        b.simulation_time = round((self.time + actual) / self.dt) * self.dt
        if actual > 0:
            # A positive stationary interval is clean; next flight contact is first contact.
            b.agent.prev_collided = b.agent.collided = False
        return StepResult(actual, used, depleted=self.energy <= 1e-10)

    def absorb_until(self, cutoff):
        self.base.simulation_time = float(cutoff)

    def close(self):
        self.base.close()
