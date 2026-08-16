from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from envs.UAVEnergyDelivery import UAVAgent, UAVEnv as LegacyUAVEnv, eps
from review_bundle.envs.navigation.state import NavigationState
from review_bundle.envs.navigation.telemetry_cost import TelemetryCostConfig, TelemetryCostModel
from review_bundle.safety.energy.critics import MonotoneQuantileCritic
from review_bundle.safety.energy.td import quantile_atom_weights, quantile_huber_loss, quantile_ssp_target
from review_bundle.safety.switching import SortieMode


ENERGY_UNIT = "synthetic_simulation_energy_units"
ENERGY_GAMMA = 1.0
ENERGY_QUANTILES = (0.50, 0.90, 0.95, 0.99)
SWITCH_QUANTILE = 0.95


class SACTrainingPhase(IntEnum):
    NAVIGATION = 1
    TD_PRETRAINING = 2
    ENERGY_MANAGED = 3


class GoalActionProvider(Protocol):
    def __call__(self, sac_goal_observation: np.ndarray) -> np.ndarray: ...


class BatchGoalActionProvider(Protocol):
    def __call__(self, sac_goal_observations: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class EnergyTDTransition:
    state: np.ndarray
    action: np.ndarray
    cost: float
    next_state: np.ndarray
    next_sac_observation: np.ndarray
    historical_next_action: np.ndarray
    goal_reached: bool
    goal_type: str
    censored_goal: bool


@dataclass(frozen=True)
class MissionEnergyEstimate:
    task_q95: float
    return_after_task_q95: float
    mission_q95_composition: float
    return_now_q95: float


class GoalConditionedQuantileTDEnergyEstimator:
    def __init__(
        self,
        *,
        state_dim: int = 7,
        action_dim: int = 3,
        quantile_levels: tuple[float, ...] = ENERGY_QUANTILES,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        learning_starts: int = 512,
        updates_per_transition: int = 1,
        target_tau: float = 0.01,
        initial_base: float = 1.0,
        initial_increment: float = 0.1,
        gamma_energy: float = ENERGY_GAMMA,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if gamma_energy != ENERGY_GAMMA:
            raise ValueError("goal-conditioned SSP energy TD requires gamma_energy=1.0")
        if tuple(quantile_levels) != ENERGY_QUANTILES:
            raise ValueError(f"energy quantiles must be {ENERGY_QUANTILES}")
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state and action dimensions must be positive")
        if batch_size <= 0 or replay_capacity < batch_size or learning_starts < batch_size:
            raise ValueError("replay and learning_starts must accommodate a positive batch")
        if updates_per_transition <= 0 or not 0.0 < target_tau <= 1.0:
            raise ValueError("invalid TD update schedule")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.state_dim + self.action_dim
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        self.hidden_dim = int(hidden_dim)
        self.learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.replay_capacity = int(replay_capacity)
        self.learning_starts = int(learning_starts)
        self.updates_per_transition = int(updates_per_transition)
        self.target_tau = float(target_tau)
        self.initial_base = float(initial_base)
        self.initial_increment = float(initial_increment)
        self.gamma_energy = float(gamma_energy)
        self.device = torch.device(device)
        self.model = MonotoneQuantileCritic(
            self.input_dim,
            self.quantile_levels,
            hidden_dim=self.hidden_dim,
            initial_base=self.initial_base,
            initial_increment=self.initial_increment,
        ).to(self.device)
        self.target_model = MonotoneQuantileCritic(
            self.input_dim,
            self.quantile_levels,
            hidden_dim=self.hidden_dim,
            initial_base=self.initial_base,
            initial_increment=self.initial_increment,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.replay: deque[EnergyTDTransition] = deque(maxlen=self.replay_capacity)
        self.trainable_replay: deque[EnergyTDTransition] = deque(maxlen=self.replay_capacity)
        self.rng = np.random.default_rng(seed)
        self.update_count = 0
        self.last_loss: float | None = None
        self.next_action_provider: BatchGoalActionProvider | None = None
        self.quantile_tensor = torch.as_tensor(self.quantile_levels, dtype=torch.float32, device=self.device)
        self.target_weights = quantile_atom_weights(self.quantile_tensor)

    def set_next_action_provider(self, provider: BatchGoalActionProvider) -> None:
        self.next_action_provider = provider

    def clear_replay(self) -> None:
        self.replay.clear()
        self.trainable_replay.clear()

    def predict_quantiles(self, energy_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        features = self._features(energy_state, action)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).squeeze(0).cpu().numpy()
        return np.asarray(prediction, dtype=np.float64)

    def predict(
        self,
        energy_state: np.ndarray,
        action: np.ndarray,
        quantile: float = SWITCH_QUANTILE,
    ) -> float:
        index = self.quantile_levels.index(float(quantile))
        return float(self.predict_quantiles(energy_state, action)[index])

    def observe_transition(
        self,
        energy_state: np.ndarray,
        action: np.ndarray,
        energy_cost: float,
        next_energy_state: np.ndarray,
        next_sac_observation: np.ndarray,
        next_action: np.ndarray,
        goal_reached: bool,
        goal_type: str,
        *,
        censored_goal: bool = False,
    ) -> float | None:
        state = self._vector(energy_state, self.state_dim, "energy_state")
        control = self._vector(action, self.action_dim, "action")
        next_state = self._vector(next_energy_state, self.state_dim, "next_energy_state")
        next_sac = self._vector(next_sac_observation, self.state_dim, "next_sac_observation")
        next_control = self._vector(next_action, self.action_dim, "next_action")
        cost = float(energy_cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("energy_cost must be finite and nonnegative")
        label = str(goal_type).upper()
        if label not in {"TASK", "CHARGER"}:
            raise ValueError("goal_type must be TASK or CHARGER")
        if bool(goal_reached) and bool(censored_goal):
            raise ValueError("a successful goal terminal cannot be censored")
        transition = EnergyTDTransition(
            state.copy(),
            control.copy(),
            cost,
            next_state.copy(),
            next_sac.copy(),
            next_control.copy(),
            bool(goal_reached),
            label,
            bool(censored_goal),
        )
        self.replay.append(transition)
        if not transition.censored_goal:
            self.trainable_replay.append(transition)
        if transition.censored_goal or len(self.trainable_replay) < self.learning_starts:
            return None
        losses = [self._update_once() for _ in range(self.updates_per_transition)]
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def _update_once(self) -> float:
        indices = self.rng.integers(0, len(self.trainable_replay), size=self.batch_size)
        batch = [self.trainable_replay[int(index)] for index in indices]
        states = torch.as_tensor(np.stack([row.state for row in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([row.action for row in batch]), dtype=torch.float32, device=self.device)
        costs = torch.as_tensor([row.cost for row in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            np.stack([row.next_state for row in batch]), dtype=torch.float32, device=self.device
        )
        terminal = torch.as_tensor([row.goal_reached for row in batch], dtype=torch.bool, device=self.device)
        if self.next_action_provider is None:
            next_actions_array = np.stack([row.historical_next_action for row in batch])
        else:
            next_sac = np.stack([row.next_sac_observation for row in batch])
            next_actions_array = np.asarray(self.next_action_provider(next_sac), dtype=np.float32)
            if next_actions_array.shape != (self.batch_size, self.action_dim):
                raise ValueError("batch goal-action provider returned the wrong shape")
        next_actions = torch.as_tensor(next_actions_array, dtype=torch.float32, device=self.device)
        self.model.train()
        predictions = self.model(torch.cat((states, actions), dim=1))
        with torch.no_grad():
            next_quantiles = self.target_model(torch.cat((next_states, next_actions), dim=1))
            targets = quantile_ssp_target(costs, next_quantiles, terminal, gamma=self.gamma_energy)
        loss = quantile_huber_loss(
            predictions,
            targets,
            self.quantile_tensor,
            target_weights=self.target_weights,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_model.parameters(), self.model.parameters()):
                target_parameter.mul_(1.0 - self.target_tau).add_(parameter, alpha=self.target_tau)
        self.update_count += 1
        return float(loss.detach().cpu().item())

    def checkpoint_payload(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "quantile_levels": self.quantile_levels,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "learning_starts": self.learning_starts,
            "target_tau": self.target_tau,
            "gamma_energy": self.gamma_energy,
            "energy_unit": ENERGY_UNIT,
            "energy_observation": "velocity3_goal_direction3_linear_distance_over_dmax1",
            "model_state_dict": self.model.state_dict(),
            "target_state_dict": self.target_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "replay_size": len(self.replay),
            "trainable_replay_size": len(self.trainable_replay),
        }

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.checkpoint_payload(), destination)

    def _features(self, energy_state: np.ndarray, action: np.ndarray) -> torch.Tensor:
        state = self._vector(energy_state, self.state_dim, "energy_state")
        control = self._vector(action, self.action_dim, "action")
        values = np.concatenate((state, control)).astype(np.float32, copy=False)
        return torch.as_tensor(values[None, :], dtype=torch.float32, device=self.device)

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite ({size},) vector")
        return vector


GoalConditionedTDEnergyEstimator = GoalConditionedQuantileTDEnergyEstimator
OnlineScalarTDEnergyEstimator = GoalConditionedQuantileTDEnergyEstimator


class _SACUAVAgent(UAVAgent):
    def __init__(
        self,
        *,
        position: np.ndarray,
        horizontal_v_max: float,
        vertical_v_max: float,
        horizontal_a_max: float,
        vertical_a_max: float,
        safe_radius: float,
        initial_energy: float,
    ) -> None:
        super().__init__(
            number=0,
            pos=position,
            v_max=horizontal_v_max,
            a_max=horizontal_a_max,
            num_lasers=0,
            l_sensor=0.0,
            safe_radius=safe_radius,
            dim=3,
            initial_energy=initial_energy,
        )
        self.horizontal_v_max = float(horizontal_v_max)
        self.vertical_v_max = float(vertical_v_max)
        self.horizontal_a_max = float(horizontal_a_max)
        self.vertical_a_max = float(vertical_a_max)

    def update_velocity(
        self,
        normalized_action: np.ndarray,
        time_step: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("SAC action must be a finite (3,) vector")
        action = np.clip(action, -1.0, 1.0)
        horizontal = action[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm > 1.0:
            horizontal = horizontal / horizontal_norm
        commanded_acceleration = np.array(
            [
                horizontal[0] * self.horizontal_a_max,
                horizontal[1] * self.horizontal_a_max,
                action[2] * self.vertical_a_max,
            ],
            dtype=np.float32,
        )
        velocity_before = self.vel.copy()
        self.vel += commanded_acceleration * float(time_step)
        horizontal_speed = float(np.linalg.norm(self.vel[:2]))
        if horizontal_speed > self.horizontal_v_max:
            self.vel[:2] *= self.horizontal_v_max / horizontal_speed
        self.vel[2] = np.clip(self.vel[2], -self.vertical_v_max, self.vertical_v_max)
        realized_acceleration = (self.vel - velocity_before) / float(time_step)
        return commanded_acceleration, realized_acceleration.astype(np.float32)


class UAVEnergyDeliverySACEnv(gym.Env, LegacyUAVEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    def __init__(
        self,
        *,
        length: float = 4000.0,
        width: float = 4000.0,
        height: float = 400.0,
        policy_dt: float = 0.20,
        physics_dt: float = 0.05,
        horizontal_v_max: float = 20.0,
        vertical_v_max: float = 5.0,
        horizontal_a_max: float = 5.0,
        vertical_a_max: float = 3.0,
        goal_radius: float = 5.0,
        near_goal_distance: float = 60.0,
        minimum_task_distance: float = 100.0,
        xy_sampling_margin: float = 100.0,
        task_z_min: float = 20.0,
        task_z_max: float = 380.0,
        max_steps_per_task: int = 4000,
        phase1_episode_max_policy_steps: int = 4000,
        phase2_episode_limit: int = 20_000,
        task_completion_reward: float = 100.0,
        progress_reward_weight: float = 1.0,
        velocity_reward_weight: float = 0.1,
        time_penalty: float = 0.01,
        boundary_penalty: float = 1.2,
        obstacle_collision_penalty: float = 1.2,
        safe_radius: float = 0.5,
        operational_energy_capacity: float | None = None,
        energy_reserve_fraction: float = 0.10,
        telemetry_cost_config: TelemetryCostConfig | None = None,
        charger_position: np.ndarray | None = None,
        charger_radius: float | None = None,
        phase: int | SACTrainingPhase = SACTrainingPhase.NAVIGATION,
        lidar_enabled: bool = False,
        lidar_max_range: float = 100.0,
        lidar_min_range: float = 0.5,
        lidar_horizontal_fov: float = 360.0,
        lidar_vertical_fov: float = 60.0,
        lidar_frequency: float = 10.0,
        lidar_horizontal_sectors: int = 128,
        lidar_vertical_sectors: int = 8,
        cbf_enabled: bool = False,
        cbf_frequency: float = 20.0,
        render_mode: str | None = None,
        render_vertical_exaggeration: float = 4.0,
    ) -> None:
        if min(length, width, height) <= 0.0:
            raise ValueError("map dimensions must be positive")
        if policy_dt <= 0.0 or physics_dt <= 0.0:
            raise ValueError("policy_dt and physics_dt must be positive")
        substep_ratio = policy_dt / physics_dt
        if not np.isclose(substep_ratio, round(substep_ratio)):
            raise ValueError("policy_dt must be an integer multiple of physics_dt")
        if min(horizontal_v_max, vertical_v_max, horizontal_a_max, vertical_a_max) <= 0.0:
            raise ValueError("velocity and acceleration limits must be positive")
        if goal_radius <= 0.0 or minimum_task_distance < goal_radius or near_goal_distance <= goal_radius:
            raise ValueError("task distance parameters must exceed goal_radius")
        if not safe_radius <= xy_sampling_margin < min(length, width) / 2.0:
            raise ValueError("xy_sampling_margin leaves no legal XY interior")
        if not safe_radius <= task_z_min < task_z_max <= height - safe_radius:
            raise ValueError("task altitude range is invalid")
        if min(max_steps_per_task, phase1_episode_max_policy_steps, phase2_episode_limit) <= 0:
            raise ValueError("policy-step guards must be positive")
        if operational_energy_capacity is not None and operational_energy_capacity <= 0.0:
            raise ValueError("operational_energy_capacity must be positive when configured")
        if not 0.0 <= energy_reserve_fraction < 1.0:
            raise ValueError("energy_reserve_fraction must lie in [0, 1)")
        if lidar_enabled:
            raise ValueError("current obstacle-free baseline fixes lidar_enabled=False")
        if cbf_enabled:
            raise ValueError("current obstacle-free baseline fixes cbf_enabled=False")
        selected_charger_radius = goal_radius if charger_radius is None else float(charger_radius)
        gym.Env.__init__(self)
        LegacyUAVEnv.__init__(
            self,
            dim_actions=3,
            length=float(length),
            width=float(width),
            height=float(height),
            num_obstacle=0,
            num_hunters=1,
            num_targets=1,
            episode_limit=int(phase2_episode_limit),
            total_orders=1,
            max_active_orders=1,
            initial_energy=1.0 if operational_energy_capacity is None else float(operational_energy_capacity),
            energy_decay_per_step=1.0,
            charging_capacity=1,
            charging_station_count=1,
            charging_radius=selected_charger_radius,
            charging_station_pos=charger_position,
        )
        self.policy_dt = float(policy_dt)
        self.physics_dt = float(physics_dt)
        self.physics_substeps_per_policy_step = int(round(substep_ratio))
        self.time_step = self.policy_dt
        self.horizontal_v_max = float(horizontal_v_max)
        self.vertical_v_max = float(vertical_v_max)
        self.horizontal_a_max = float(horizontal_a_max)
        self.vertical_a_max = float(vertical_a_max)
        self.goal_tolerance = float(goal_radius)
        self.near_goal_distance = float(near_goal_distance)
        self.minimum_task_distance = float(minimum_task_distance)
        self.xy_sampling_margin = float(xy_sampling_margin)
        self.task_z_min = float(task_z_min)
        self.task_z_max = float(task_z_max)
        self.max_steps_per_task = int(max_steps_per_task)
        self.phase1_episode_max_policy_steps = int(phase1_episode_max_policy_steps)
        self.phase2_episode_limit = int(phase2_episode_limit)
        self.max_cycles = self.phase1_episode_max_policy_steps
        self.task_completion_reward = float(task_completion_reward)
        self.progress_reward_weight = float(progress_reward_weight)
        self.velocity_reward_weight = float(velocity_reward_weight)
        self.time_penalty = float(time_penalty)
        self.boundary_penalty = float(boundary_penalty)
        self.obstacle_collision_penalty = float(obstacle_collision_penalty)
        self.safe_radius = float(safe_radius)
        self.operational_energy_capacity = (
            None if operational_energy_capacity is None else float(operational_energy_capacity)
        )
        self.energy_reserve_fraction = float(energy_reserve_fraction)
        self.energy_reserve = (
            None
            if self.operational_energy_capacity is None
            else self.operational_energy_capacity * self.energy_reserve_fraction
        )
        self.battery_capacity_source = (
            None if self.operational_energy_capacity is None else "explicit_test_or_override"
        )
        self.telemetry_cost_model = TelemetryCostModel(telemetry_cost_config)
        self.render_mode = render_mode
        self.render_vertical_exaggeration = float(render_vertical_exaggeration)
        self.phase = SACTrainingPhase(int(phase))
        self.mission_switching_enabled = self.phase is SACTrainingPhase.ENERGY_MANAGED
        self.reset_at_charger = self.phase is SACTrainingPhase.ENERGY_MANAGED
        self.lidar_enabled = False
        self.lidar_max_range = float(lidar_max_range)
        self.lidar_min_range = float(lidar_min_range)
        self.lidar_horizontal_fov = float(lidar_horizontal_fov)
        self.lidar_vertical_fov = float(lidar_vertical_fov)
        self.lidar_frequency = float(lidar_frequency)
        self.lidar_horizontal_sectors = int(lidar_horizontal_sectors)
        self.lidar_vertical_sectors = int(lidar_vertical_sectors)
        self.cbf_enabled = False
        self.cbf_frequency = float(cbf_frequency)
        self.d_max = float(np.linalg.norm([self.length, self.width, self.height]))
        self.current_task_point = np.zeros(3, dtype=np.float32)
        self.mode = SortieMode.TASK
        self.energy_estimator: GoalConditionedQuantileTDEnergyEstimator | None = None
        self.goal_action_provider: GoalActionProvider | None = None
        self.energy_learning_enabled = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-1.0] * 6 + [0.0], dtype=np.float32),
            np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )
        self.orders = []
        self.goals = []
        self.obstacles = []
        self.trajectory_log: list[dict[str, object]] = []
        self.switching_events: list[dict[str, object]] = []
        self.battery_cycle_records: list[dict[str, object]] = []
        self.completed_goal_records: list[dict[str, object]] = []
        self.render_overlay: dict[str, object] = {}
        self._pending_goal_transitions: list[dict[str, object]] = []
        self.last_mission_estimate: MissionEnergyEstimate | None = None
        self.last_quantile_prediction: np.ndarray | None = None
        self.last_td_loss: float | None = None
        self.last_realized_energy = 0.0
        self.last_battery_cycle_record: dict[str, object] | None = None

    @property
    def agent(self) -> _SACUAVAgent:
        if len(self.agents) != 1:
            raise RuntimeError("single-UAV environment is not initialized")
        return self.agents[0]

    @property
    def charger_position(self) -> np.ndarray:
        return self.charging_station_positions[0].copy()

    @property
    def active_goal(self) -> np.ndarray:
        return self.charger_position if self.mode is SortieMode.CHARGER_COMMITTED else self.current_task_point.copy()

    @property
    def finite_energy_enabled(self) -> bool:
        return self.phase is SACTrainingPhase.ENERGY_MANAGED

    @property
    def episode_policy_step_limit(self) -> int:
        return (
            self.phase2_episode_limit
            if self.finite_energy_enabled
            else self.phase1_episode_max_policy_steps
        )

    @property
    def telemetry_cost_config(self) -> dict[str, object]:
        config = self.telemetry_cost_model.config
        return {
            "base_power": float(config.base_power),
            "velocity_coefficients": config.velocity_coefficients.tolist(),
            "acceleration_coefficients": config.acceleration_coefficients.tolist(),
            "compute_power": float(config.compute_power),
            "communication_power": float(config.communication_power),
            "simulation_error": float(config.simulation_error),
            "flight_energy_multiplier": float(config.flight_energy_multiplier),
            "energy_unit": ENERGY_UNIT,
        }

    def bind_energy_learning(
        self,
        *,
        energy_estimator: GoalConditionedQuantileTDEnergyEstimator,
        goal_action_provider: GoalActionProvider,
        training_enabled: bool,
    ) -> None:
        self.energy_estimator = energy_estimator
        self.goal_action_provider = goal_action_provider
        self.energy_learning_enabled = bool(training_enabled)

    def set_phase(self, phase: int | SACTrainingPhase) -> None:
        self.phase = SACTrainingPhase(int(phase))
        self.max_cycles = self.episode_policy_step_limit
        if self.phase is SACTrainingPhase.NAVIGATION:
            self.energy_learning_enabled = False

    def configure_calibrated_battery(
        self,
        capacity: float,
        *,
        reserve_fraction: float | None = None,
        source: str = "phase1_frozen_policy_calibration",
    ) -> None:
        selected_capacity = float(capacity)
        selected_fraction = (
            self.energy_reserve_fraction if reserve_fraction is None else float(reserve_fraction)
        )
        if not np.isfinite(selected_capacity) or selected_capacity <= 0.0:
            raise ValueError("calibrated battery capacity must be finite and positive")
        if not 0.0 <= selected_fraction < 1.0:
            raise ValueError("energy reserve fraction must lie in [0, 1)")
        self.operational_energy_capacity = selected_capacity
        self.energy_reserve_fraction = selected_fraction
        self.energy_reserve = selected_capacity * selected_fraction
        self.battery_capacity_source = str(source)
        if self.agents:
            self.agent.initial_energy = selected_capacity
            self.agent.energy = selected_capacity

    def enable_phase_two(self, *, reserve: float | None = None) -> None:
        if self.energy_estimator is None or self.goal_action_provider is None:
            raise RuntimeError("Phase 2 requires a frozen navigation policy and energy estimator")
        if self.operational_energy_capacity is None:
            raise RuntimeError("Phase 2 requires calibrated battery capacity")
        if reserve is not None:
            if not 0.0 <= reserve < self.operational_energy_capacity:
                raise ValueError("reserve must lie inside battery capacity")
            self.energy_reserve = float(reserve)
            self.energy_reserve_fraction = self.energy_reserve / self.operational_energy_capacity
        self.set_phase(SACTrainingPhase.ENERGY_MANAGED)
        self.mission_switching_enabled = True
        self.reset_at_charger = True

    def enable_battery_validation(self) -> None:
        if self.operational_energy_capacity is None:
            raise RuntimeError("battery validation requires calibrated capacity")
        self.set_phase(SACTrainingPhase.ENERGY_MANAGED)
        self.mission_switching_enabled = False
        self.reset_at_charger = False
        self.energy_learning_enabled = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        gym.Env.reset(self, seed=seed)
        reset_options = {} if options is None else dict(options)
        requested_start = reset_options.pop("start_position", None)
        requested_velocity = reset_options.pop("start_velocity", None)
        requested_task = reset_options.pop("task_point", None)
        if reset_options:
            raise ValueError(f"unsupported reset options: {sorted(reset_options)}")
        default_start = (
            self.charger_position
            if self.finite_energy_enabled and self.reset_at_charger
            else self._sample_legal_position()
        )
        start = self._validate_position(default_start if requested_start is None else requested_start, "start_position")
        velocity = np.zeros(3, dtype=np.float32)
        if requested_velocity is not None:
            velocity = self._validate_velocity(requested_velocity)
        self.agents = [
            _SACUAVAgent(
                position=start,
                horizontal_v_max=self.horizontal_v_max,
                vertical_v_max=self.vertical_v_max,
                horizontal_a_max=self.horizontal_a_max,
                vertical_a_max=self.vertical_a_max,
                safe_radius=self.safe_radius,
                initial_energy=(
                    1.0
                    if self.operational_energy_capacity is None
                    else self.operational_energy_capacity
                ),
            )
        ]
        self.agent.vel = velocity
        self.agent.prev_pos = start.copy()
        self.agent.last_pos = start.copy()
        self.agent.spawn_pos = start.copy()
        self.current_task_point = (
            self._sample_task_point(start)
            if requested_task is None
            else self._validate_task_point(requested_task, start)
        )
        self.tasks_completed = 0
        self.tasks_in_current_battery_cycle = 0
        self.battery_cycles_completed = 0
        self.battery_cycle_id = 0
        self.current_step = 0
        self.steps_in_current_task = 0
        self.mode = SortieMode.TASK
        self.task_to_charger_count = 0
        self.task_stuck_count = 0
        self.boundary_collision_count = 0
        self.consecutive_boundary_contacts = 0
        self.maximum_consecutive_boundary_contacts = 0
        self.obstacle_collision_count = 0
        self.cumulative_virtual_energy = 0.0
        self.cycle_start_global_step = 0
        self.cycle_start_energy = float(self.agent.energy)
        self.cycle_start_simulation_time = 0.0
        self.cycle_policy_steps = 0
        self.cycle_distance_flown = 0.0
        self.simulation_time = 0.0
        self.return_commit_record: dict[str, object] | None = None
        self.agent.goal = self.current_task_point.copy()
        self.agent.reached = False
        self.agent_paths = [[start.copy()]]
        self.trajectory_log = []
        self.switching_events = []
        self.battery_cycle_records = []
        self.completed_goal_records = []
        self._pending_goal_transitions = []
        self.last_mission_estimate = None
        self.last_quantile_prediction = None
        self.last_td_loss = None
        self.last_realized_energy = 0.0
        self.last_battery_cycle_record = None
        self._current_goal_initial_distance = float(np.linalg.norm(self.active_goal - self.agent.pos))
        self._current_goal_path_length = 0.0
        observation = self._active_goal_sac_observation()
        return observation, self._info()

    def step(self, action: np.ndarray):
        if not self.agents:
            raise RuntimeError("reset must be called before step")
        nominal_action = np.asarray(action, dtype=np.float32)
        if nominal_action.shape != (3,) or not np.all(np.isfinite(nominal_action)):
            raise ValueError("action must be a finite (3,) vector")
        nominal_action = np.clip(nominal_action, -1.0, 1.0)
        self.current_step += 1
        self.steps_in_current_task += 1
        self.last_battery_cycle_record = None
        mode_before_decision = self.mode
        switched_at_step_start = False
        if self.finite_energy_enabled and self.mission_switching_enabled and self.mode is SortieMode.TASK:
            switched_at_step_start = self._refresh_mission_decision()
            if switched_at_step_start:
                self._finalize_goal_trajectory(success=False, censored_reason="task_interrupted_by_charger_commitment")
                self.steps_in_current_task = 0
                self._start_goal_trajectory(self.charger_position)
        goal_type = "CHARGER" if self.mode is SortieMode.CHARGER_COMMITTED else "TASK"
        segment_goal = self.active_goal.copy()
        executed_action = (
            self._goal_action(self.sac_observation_for_goal(segment_goal))
            if switched_at_step_start
            else nominal_action
        )
        distance_before = float(np.linalg.norm(segment_goal - self.agent.pos))
        energy_state_before = self.energy_state_for_goal(segment_goal)
        quantiles_before = self._predict_quantiles(segment_goal, executed_action)
        total_energy = 0.0
        boundary_contact = False
        obstacle_collision = False
        physics_substeps = 0
        policy_distance = 0.0
        goal_reached = False
        energy_exhausted = False
        last_commanded_acceleration = np.zeros(3, dtype=np.float32)
        last_realized_acceleration = np.zeros(3, dtype=np.float32)
        for _ in range(self.physics_substeps_per_policy_step):
            substep_start = self.agent.pos.copy()
            (
                last_commanded_acceleration,
                last_realized_acceleration,
            ) = self.agent.update_velocity(executed_action, self.physics_dt)
            velocity_after_propulsion = self.agent.vel.copy()
            self.agent.preview_position(self.physics_dt)
            substep_energy = self._realized_energy_cost(
                last_realized_acceleration,
                self.physics_dt,
                velocity=velocity_after_propulsion,
            )
            substep_boundary, _, _ = self._apply_boundary_constraints(self.agent)
            boundary_contact = boundary_contact or bool(substep_boundary)
            self.agent.pos = self.agent.prev_pos.copy()
            substep_distance = float(np.linalg.norm(self.agent.pos - substep_start))
            self._current_goal_path_length += substep_distance
            policy_distance += substep_distance
            total_energy += substep_energy
            if self.finite_energy_enabled:
                self.agent.energy = max(0.0, float(self.agent.energy) - substep_energy)
            physics_substeps += 1
            energy_exhausted = bool(self.finite_energy_enabled and self.agent.energy <= eps)
            goal_reached = bool(np.linalg.norm(segment_goal - self.agent.pos) <= self.goal_tolerance)
            if goal_reached or energy_exhausted:
                break
        transition_dt = physics_substeps * self.physics_dt
        self.simulation_time += transition_dt
        if self.finite_energy_enabled:
            self.cycle_policy_steps += 1
            self.cycle_distance_flown += policy_distance
        self.last_realized_energy = float(total_energy)
        self.cumulative_virtual_energy += float(total_energy)
        if boundary_contact:
            self.boundary_collision_count += 1
            self.consecutive_boundary_contacts += 1
            self.maximum_consecutive_boundary_contacts = max(
                self.maximum_consecutive_boundary_contacts,
                self.consecutive_boundary_contacts,
            )
        else:
            self.consecutive_boundary_contacts = 0
        if obstacle_collision:
            self.obstacle_collision_count += 1
        distance_after = float(np.linalg.norm(segment_goal - self.agent.pos))
        progress = distance_before - distance_after
        task_completed_now = bool(goal_type == "TASK" and goal_reached and not energy_exhausted)
        charger_reached_now = bool(goal_type == "CHARGER" and goal_reached and not energy_exhausted)
        instantaneous_service_reset = False
        completed_goal_evaluation: dict[str, object] | None = None
        battery_cycle_end = False
        switched_now = switched_at_step_start
        next_energy_state = self.energy_state_for_goal(segment_goal)
        next_sac_for_old_goal = self.sac_observation_for_goal(segment_goal)
        next_action = (
            np.zeros(3, dtype=np.float32)
            if goal_reached or energy_exhausted
            else self._goal_action(next_sac_for_old_goal, allow_unbound=True)
        )
        self._pending_goal_transitions.append(
            {
                "state": energy_state_before,
                "action": executed_action.copy(),
                "cost": float(total_energy),
                "next_state": next_energy_state,
                "next_sac_observation": next_sac_for_old_goal,
                "next_action": next_action,
                "goal_reached": bool(goal_reached),
                "goal_type": goal_type,
                "quantiles": None if quantiles_before is None else quantiles_before.copy(),
                "boundary_contact": bool(boundary_contact),
            }
        )
        single_task_phase = self.phase in {
            SACTrainingPhase.NAVIGATION,
            SACTrainingPhase.TD_PRETRAINING,
        }
        if task_completed_now and single_task_phase:
            completed_goal_evaluation = self._finalize_goal_trajectory(success=True)
            self.tasks_completed += 1
            self.agent.reached = True
        elif task_completed_now:
            completed_goal_evaluation = self._finalize_goal_trajectory(success=True)
            self.tasks_completed += 1
            if self.finite_energy_enabled:
                self.tasks_in_current_battery_cycle += 1
            self.agent.vel[:] = 0.0
            instantaneous_service_reset = True
            self.current_task_point = self._sample_task_point(self.agent.pos)
            self.steps_in_current_task = 0
            self.mode = SortieMode.TASK
            self.agent.goal = self.current_task_point.copy()
            self._start_goal_trajectory(self.current_task_point)
        elif charger_reached_now:
            completed_goal_evaluation = self._finalize_goal_trajectory(success=True)
            self.last_battery_cycle_record = self._complete_battery_cycle()
            battery_cycle_end = True
            instantaneous_service_reset = True
        task_stuck = bool(
            not task_completed_now
            and self.mode is SortieMode.TASK
            and not energy_exhausted
            and self.steps_in_current_task >= self.max_steps_per_task
        )
        episode_guard = bool(
            not task_completed_now
            and not energy_exhausted
            and self.current_step >= self.episode_policy_step_limit
        )
        if energy_exhausted:
            self._finalize_goal_trajectory(success=False, censored_reason="energy_exhausted")
            self.last_battery_cycle_record = self._failed_battery_cycle_record("energy_exhausted")
            battery_cycle_end = True
        elif task_stuck:
            self._finalize_goal_trajectory(success=False, censored_reason="task_stuck")
            self.task_stuck_count += 1
            if self.finite_energy_enabled:
                self.last_battery_cycle_record = self._failed_battery_cycle_record("task_stuck")
                battery_cycle_end = True
        elif episode_guard:
            self._finalize_goal_trajectory(success=False, censored_reason="episode_guard")
            if self.finite_energy_enabled:
                self.last_battery_cycle_record = self._failed_battery_cycle_record(
                    "episode_emergency_step_guard"
                )
                battery_cycle_end = True
        if (
            self.finite_energy_enabled
            and self.mission_switching_enabled
            and not any((energy_exhausted, task_stuck, episode_guard, charger_reached_now))
        ):
            if self.mode is SortieMode.TASK and not task_completed_now:
                switched_after_motion = self._refresh_mission_decision()
                if switched_after_motion:
                    switched_now = True
                    self._finalize_goal_trajectory(
                        success=False,
                        censored_reason="task_interrupted_by_charger_commitment",
                    )
                    self.steps_in_current_task = 0
                    self._start_goal_trajectory(self.charger_position)
        self.agent.goal = self.active_goal.copy()
        self.agent_paths[0].append(self.agent.pos.copy())
        self.agent.collided = bool(boundary_contact or obstacle_collision)
        reward_components = self._reward_components(
            progress=progress,
            goal=segment_goal,
            velocity=self.agent.vel,
            task_completed=task_completed_now,
            boundary_contact=boundary_contact,
            obstacle_collision=obstacle_collision,
        )
        reward = float(sum(reward_components.values()))
        terminated = bool(energy_exhausted or (task_completed_now and single_task_phase))
        truncated = bool((task_stuck or episode_guard) and not terminated)
        end_reason = (
            "energy_exhausted"
            if energy_exhausted
            else "goal_reached"
            if task_completed_now and single_task_phase
            else "task_step_limit"
            if task_stuck
            else "episode_emergency_step_guard"
            if episode_guard
            else None
        )
        info = self._info(
            task_completed_now=task_completed_now,
            charger_reached_now=charger_reached_now,
            battery_cycle_end=battery_cycle_end,
            switched_now=switched_now,
            boundary_contact=boundary_contact,
            obstacle_collision=obstacle_collision,
            realized_energy_cost=total_energy,
            transition_dt=transition_dt,
            physics_substeps=physics_substeps,
            instantaneous_service_reset=instantaneous_service_reset,
            task_stuck=task_stuck,
            end_reason=end_reason,
            reward_components=reward_components,
            completed_goal_evaluation=completed_goal_evaluation,
        )
        info.update(
            {
                "progress": float(progress),
                "commanded_acceleration": last_commanded_acceleration.copy(),
                "realized_acceleration": last_realized_acceleration.copy(),
                "physical_acceleration": last_realized_acceleration.copy(),
                "executed_action": executed_action.copy(),
                "mode_before": mode_before_decision.value,
                "active_goal_before": segment_goal.copy(),
                "energy_exhausted": energy_exhausted,
                "censored_return": bool(energy_exhausted and goal_type == "CHARGER"),
                "is_success": bool(task_completed_now and single_task_phase),
                "navigation_failure": bool(task_stuck or episode_guard),
            }
        )
        self.trajectory_log.append(
            {
                "step": self.current_step,
                "position": self.agent.pos.copy(),
                "velocity": self.agent.vel.copy(),
                "nominal_action": nominal_action.copy(),
                "executed_action": executed_action.copy(),
                "commanded_acceleration": last_commanded_acceleration.copy(),
                "realized_acceleration": last_realized_acceleration.copy(),
                "physical_acceleration": last_realized_acceleration.copy(),
                "active_goal": self.active_goal.copy(),
                "task_point": self.current_task_point.copy(),
                "mode": self.mode.value,
                "remaining_energy": float(self.agent.energy),
                "realized_energy_cost": float(total_energy),
                "transition_dt": float(transition_dt),
                "physics_substeps": int(physics_substeps),
                "task_completed_now": task_completed_now,
                "charger_reached_now": charger_reached_now,
                "energy_exhausted": energy_exhausted,
                "boundary_contact": bool(boundary_contact),
                "obstacle_collision": bool(obstacle_collision),
                "instantaneous_service_reset": instantaneous_service_reset,
            }
        )
        return self._active_goal_sac_observation(), reward, terminated, truncated, info

    def sac_observation_for_goal(
        self,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        target = self._validate_position(goal, "goal")
        velocity_feature, direction, distance = self._goal_features(target, position=position, velocity=velocity)
        distance_feature = np.log1p(distance) / np.log1p(self.d_max)
        observation = np.concatenate((velocity_feature, direction, [np.clip(distance_feature, 0.0, 1.0)]))
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def energy_state_for_goal(
        self,
        goal: np.ndarray,
        *,
        position: np.ndarray | None = None,
        velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        target = self._validate_position(goal, "energy_goal")
        velocity_feature, direction, distance = self._goal_features(target, position=position, velocity=velocity)
        state = np.concatenate((velocity_feature, direction, [np.clip(distance / self.d_max, 0.0, 1.0)]))
        return np.clip(state, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def observation_for_goal(self, goal: np.ndarray) -> np.ndarray:
        return self.sac_observation_for_goal(goal)

    def charger_relative_observation(self) -> np.ndarray:
        return self.sac_observation_for_goal(self.charger_position)

    def charger_energy_state(self) -> np.ndarray:
        return self.energy_state_for_goal(self.charger_position)

    def _active_goal_sac_observation(self) -> np.ndarray:
        return self.sac_observation_for_goal(self.active_goal)

    def _goal_features(
        self,
        goal: np.ndarray,
        *,
        position: np.ndarray | None,
        velocity: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        current_position = self.agent.pos if position is None else self._validate_position(position, "position")
        current_velocity = self.agent.vel if velocity is None else self._validate_velocity(velocity)
        velocity_feature = np.array(
            [
                current_velocity[0] / self.horizontal_v_max,
                current_velocity[1] / self.horizontal_v_max,
                current_velocity[2] / self.vertical_v_max,
            ],
            dtype=np.float32,
        )
        delta = goal - current_position
        distance = float(np.linalg.norm(delta))
        direction = (delta / max(distance, eps)).astype(np.float32)
        return velocity_feature, direction, distance

    def _realized_energy_cost(
        self,
        acceleration: np.ndarray,
        duration: float,
        *,
        velocity: np.ndarray | None = None,
    ) -> float:
        propulsion_velocity = self.agent.vel if velocity is None else np.asarray(velocity, dtype=np.float32)
        state = NavigationState(
            position=self.agent.pos,
            velocity=propulsion_velocity,
            energy=float(self.agent.energy),
            timestamp=float(self.current_step * self.policy_dt),
        )
        return self.telemetry_cost_model.realized_cost(state, acceleration, duration)

    def _goal_action(self, observation: np.ndarray, *, allow_unbound: bool = False) -> np.ndarray:
        if self.goal_action_provider is None:
            if allow_unbound:
                return np.zeros(3, dtype=np.float32)
            raise RuntimeError("goal action provider is not bound")
        action = np.asarray(self.goal_action_provider(observation.copy()), dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("goal action provider must return a finite (3,) action")
        return np.clip(action, -1.0, 1.0)

    def _predict_quantiles(self, goal: np.ndarray, action: np.ndarray) -> np.ndarray | None:
        if self.energy_estimator is None:
            return None
        prediction = self.energy_estimator.predict_quantiles(self.energy_state_for_goal(goal), action)
        self.last_quantile_prediction = prediction.copy()
        return prediction

    def mission_energy_estimate(self) -> MissionEnergyEstimate:
        if self.energy_estimator is None or self.goal_action_provider is None:
            raise RuntimeError("mission estimation requires energy estimator and frozen navigation policy")
        task_sac = self.sac_observation_for_goal(self.current_task_point)
        task_action = self._goal_action(task_sac)
        task_q95 = self.energy_estimator.predict(
            self.energy_state_for_goal(self.current_task_point), task_action, quantile=SWITCH_QUANTILE
        )
        return_position = self.current_task_point.copy()
        return_velocity = np.zeros(3, dtype=np.float32)
        return_after_sac = self.sac_observation_for_goal(
            self.charger_position,
            position=return_position,
            velocity=return_velocity,
        )
        return_after_action = self._goal_action(return_after_sac)
        return_after_q95 = self.energy_estimator.predict(
            self.energy_state_for_goal(
                self.charger_position,
                position=return_position,
                velocity=return_velocity,
            ),
            return_after_action,
            quantile=SWITCH_QUANTILE,
        )
        return_now_sac = self.sac_observation_for_goal(self.charger_position)
        return_now_action = self._goal_action(return_now_sac)
        return_now_q95 = self.energy_estimator.predict(
            self.energy_state_for_goal(self.charger_position),
            return_now_action,
            quantile=SWITCH_QUANTILE,
        )
        return MissionEnergyEstimate(
            float(task_q95),
            float(return_after_q95),
            float(task_q95 + return_after_q95),
            float(return_now_q95),
        )

    def _refresh_mission_decision(self) -> bool:
        if self.mode is SortieMode.CHARGER_COMMITTED:
            self.agent.goal = self.charger_position
            return False
        estimate = self.mission_energy_estimate()
        self.last_mission_estimate = estimate
        remaining = float(self.agent.energy)
        immediate_margin = remaining - estimate.return_now_q95 - self.energy_reserve
        mission_margin = remaining - estimate.mission_q95_composition - self.energy_reserve
        reason = None
        if immediate_margin <= 0.0:
            reason = "immediate_return_energy_boundary"
        elif mission_margin <= 0.0:
            reason = "compositional_mission_energy_boundary"
        if reason is None:
            return False
        self.mode = SortieMode.CHARGER_COMMITTED
        self.task_to_charger_count += 1
        self.agent.goal = self.charger_position
        event = {
            "global_step": int(self.current_step),
            "battery_cycle_id": int(self.battery_cycle_id),
            "position": self.agent.pos.copy(),
            "velocity": self.agent.vel.copy(),
            "distance_to_charger": float(np.linalg.norm(self.charger_position - self.agent.pos)),
            "remaining_energy": remaining,
            "E_task_95": estimate.task_q95,
            "E_return_after_task_95": estimate.return_after_task_q95,
            "E_mission_95": estimate.mission_q95_composition,
            "E_return_now_95": estimate.return_now_q95,
            "reserve": self.energy_reserve,
            "continuation_margin": min(immediate_margin, mission_margin),
            "mode_before": SortieMode.TASK.value,
            "mode_after": SortieMode.CHARGER_COMMITTED.value,
            "reason": reason,
        }
        self.return_commit_record = event.copy()
        self.switching_events.append(event)
        return True

    def _complete_battery_cycle(self) -> dict[str, object]:
        commit = {} if self.return_commit_record is None else self.return_commit_record
        record = {
            "battery_cycle_id": int(self.battery_cycle_id),
            "cycle_start_global_step": int(self.cycle_start_global_step),
            "cycle_end_global_step": int(self.current_step),
            "cycle_start_transition": int(self.cycle_start_global_step),
            "cycle_end_transition": int(self.current_step),
            "cycle_simulation_time": float(self.simulation_time - self.cycle_start_simulation_time),
            "cycle_policy_steps": int(self.cycle_policy_steps),
            "distance_flown": float(self.cycle_distance_flown),
            "tasks_completed_in_cycle": int(self.tasks_in_current_battery_cycle),
            "tasks_completed": int(self.tasks_in_current_battery_cycle),
            "total_energy_used": float(self.cycle_start_energy - self.agent.energy),
            "remaining_energy_at_cycle_end": float(self.agent.energy),
            "remaining_energy_fraction_at_charger": float(
                self.agent.energy / self.operational_energy_capacity
            ),
            "return_commit_step": commit.get("global_step"),
            "distance_to_charger_at_commit": commit.get("distance_to_charger"),
            "predicted_energy_at_commit": commit.get("E_return_now_95"),
            "remaining_energy_at_commit": commit.get("remaining_energy"),
            "energy_at_switch": commit.get("remaining_energy"),
            "E_mission_95_at_switch": commit.get("E_mission_95"),
            "E_return_now_95_at_switch": commit.get("E_return_now_95"),
            "reserve_at_commit": commit.get("reserve"),
            "charger_reached": True,
            "energy_exhausted": False,
            "emergency_time_limit": False,
            "return_success": True,
        }
        self.battery_cycle_records.append(record)
        self.battery_cycles_completed += 1
        self.battery_cycle_id += 1
        self.tasks_in_current_battery_cycle = 0
        self.cycle_start_global_step = self.current_step
        if self.operational_energy_capacity is None:
            raise RuntimeError("battery cycle cannot complete without calibrated capacity")
        self.cycle_start_energy = self.operational_energy_capacity
        self.cycle_start_simulation_time = self.simulation_time
        self.cycle_policy_steps = 0
        self.cycle_distance_flown = 0.0
        self.return_commit_record = None
        self.task_to_charger_count = 0
        self.agent.vel[:] = 0.0
        self.agent.energy = self.operational_energy_capacity
        self.mode = SortieMode.TASK
        self.current_task_point = self._sample_task_point(self.agent.pos)
        self.steps_in_current_task = 0
        self.agent.goal = self.current_task_point.copy()
        self._start_goal_trajectory(self.current_task_point)
        return record

    def _failed_battery_cycle_record(self, reason: str) -> dict[str, object]:
        if not self.finite_energy_enabled or self.operational_energy_capacity is None:
            raise RuntimeError("failed battery-cycle records require finite calibrated energy")
        commit = {} if self.return_commit_record is None else self.return_commit_record
        record = {
            "battery_cycle_id": int(self.battery_cycle_id),
            "cycle_start_global_step": int(self.cycle_start_global_step),
            "cycle_end_global_step": int(self.current_step),
            "cycle_start_transition": int(self.cycle_start_global_step),
            "cycle_end_transition": int(self.current_step),
            "cycle_simulation_time": float(self.simulation_time - self.cycle_start_simulation_time),
            "cycle_policy_steps": int(self.cycle_policy_steps),
            "distance_flown": float(self.cycle_distance_flown),
            "tasks_completed_in_cycle": int(self.tasks_in_current_battery_cycle),
            "tasks_completed": int(self.tasks_in_current_battery_cycle),
            "total_energy_used": float(self.cycle_start_energy - self.agent.energy),
            "remaining_energy_at_cycle_end": float(self.agent.energy),
            "remaining_energy_fraction_at_charger": None,
            "return_commit_step": commit.get("global_step"),
            "distance_to_charger_at_commit": commit.get("distance_to_charger"),
            "predicted_energy_at_commit": commit.get("E_return_now_95"),
            "remaining_energy_at_commit": commit.get("remaining_energy"),
            "energy_at_switch": commit.get("remaining_energy"),
            "E_mission_95_at_switch": commit.get("E_mission_95"),
            "E_return_now_95_at_switch": commit.get("E_return_now_95"),
            "reserve_at_commit": commit.get("reserve"),
            "charger_reached": False,
            "energy_exhausted": reason == "energy_exhausted",
            "emergency_time_limit": reason == "episode_emergency_step_guard",
            "return_success": False,
            "end_reason": reason,
        }
        self.battery_cycle_records.append(record)
        return record

    def _start_goal_trajectory(self, goal: np.ndarray) -> None:
        self._pending_goal_transitions = []
        self._current_goal_initial_distance = float(np.linalg.norm(goal - self.agent.pos))
        self._current_goal_path_length = 0.0

    def _finalize_goal_trajectory(
        self,
        *,
        success: bool,
        censored_reason: str | None = None,
    ) -> dict[str, object] | None:
        if not self._pending_goal_transitions:
            return None
        rows = self._pending_goal_transitions
        if self.energy_learning_enabled and self.energy_estimator is not None:
            for row in rows:
                loss = self.energy_estimator.observe_transition(
                    row["state"],
                    row["action"],
                    row["cost"],
                    row["next_state"],
                    row["next_sac_observation"],
                    row["next_action"],
                    row["goal_reached"],
                    row["goal_type"],
                    censored_goal=not success,
                )
                if loss is not None:
                    self.last_td_loss = float(loss)
        result = None
        if success:
            costs = np.asarray([float(row["cost"]) for row in rows], dtype=np.float64)
            returns = np.cumsum(costs[::-1])[::-1]
            prediction_rows = [row["quantiles"] for row in rows]
            if all(prediction is not None for prediction in prediction_rows):
                predictions = np.stack(prediction_rows).astype(np.float64)
                q50_error = predictions[:, 0] - returns
                boundary_contacts = np.asarray(
                    [bool(row.get("boundary_contact", False)) for row in rows],
                    dtype=bool,
                )
                consecutive_boundary_contacts = 0
                max_consecutive_boundary_contacts = 0
                for contact in boundary_contacts:
                    consecutive_boundary_contacts = (
                        consecutive_boundary_contacts + 1 if contact else 0
                    )
                    max_consecutive_boundary_contacts = max(
                        max_consecutive_boundary_contacts,
                        consecutive_boundary_contacts,
                    )
                result = {
                    "goal_type": rows[0]["goal_type"],
                    "initial_goal_distance": float(self._current_goal_initial_distance),
                    "distance_bucket": self._distance_bucket(self._current_goal_initial_distance),
                    "policy_steps": len(rows),
                    "path_length": float(self._current_goal_path_length),
                    "path_ratio": float(
                        self._current_goal_path_length
                        / max(self._current_goal_initial_distance, eps)
                    ),
                    "path_efficiency_fraction": float(
                        self._current_goal_initial_distance / max(self._current_goal_path_length, eps)
                    ),
                    "true_total_energy": float(returns[0]),
                    "had_boundary_contact": bool(np.any(boundary_contacts)),
                    "boundary_contact_steps": int(np.sum(boundary_contacts)),
                    "boundary_contact_fraction": float(np.mean(boundary_contacts)),
                    "max_consecutive_boundary_contacts": int(
                        max_consecutive_boundary_contacts
                    ),
                    "finite_predictions": bool(np.all(np.isfinite(predictions))),
                    "quantile_ordering_valid": bool(
                        np.all(np.diff(predictions, axis=1) >= -1e-7)
                    ),
                    "td_mae": float(np.mean(np.abs(q50_error))),
                    "td_rmse": float(np.sqrt(np.mean(q50_error**2))),
                    "td_bias": float(np.mean(q50_error)),
                    "td_underestimation_rate": float(np.mean(predictions[:, 0] < returns)),
                    "td_overestimation_rate": float(np.mean(predictions[:, 0] > returns)),
                    "td_mean_underestimation_magnitude": float(
                        np.mean(np.maximum(returns - predictions[:, 0], 0.0))
                    ),
                    "td_relative_error": float(
                        np.mean(np.abs(q50_error) / np.maximum(returns, 1e-8))
                    ),
                    "q50_coverage": float(np.mean(returns <= predictions[:, 0])),
                    "q90_coverage": float(np.mean(returns <= predictions[:, 1])),
                    "q95_coverage": float(np.mean(returns <= predictions[:, 2])),
                    "q99_coverage": float(np.mean(returns <= predictions[:, 3])),
                }
                self.completed_goal_records.append(result)
        self._pending_goal_transitions = []
        if not success and censored_reason is not None:
            return {"success": False, "censored_reason": censored_reason, "policy_steps": len(rows)}
        return result

    @staticmethod
    def _distance_bucket(distance: float) -> str:
        if distance < 500.0:
            return "100-500"
        if distance < 1500.0:
            return "500-1500"
        if distance < 2500.0:
            return "1500-2500"
        if distance <= 4000.0:
            return "2500-4000"
        return ">4000"

    def _reward_components(
        self,
        *,
        progress: float,
        goal: np.ndarray,
        velocity: np.ndarray,
        task_completed: bool,
        boundary_contact: bool,
        obstacle_collision: bool,
    ) -> dict[str, float]:
        delta = goal - self.agent.pos
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, eps)
        velocity_toward_goal = float(np.dot(velocity, direction)) / self.horizontal_v_max
        return {
            "progress_reward_component": self.progress_reward_weight * float(progress),
            "velocity_reward_component": self.velocity_reward_weight * max(0.0, velocity_toward_goal),
            "task_completion_reward_component": self.task_completion_reward * float(task_completed),
            "time_penalty_component": -self.time_penalty,
            "boundary_penalty_component": -self.boundary_penalty * float(boundary_contact),
            "obstacle_penalty_component": -self.obstacle_collision_penalty * float(obstacle_collision),
        }

    @staticmethod
    def _zero_reward_components() -> dict[str, float]:
        return {
            "progress_reward_component": 0.0,
            "velocity_reward_component": 0.0,
            "task_completion_reward_component": 0.0,
            "time_penalty_component": 0.0,
            "boundary_penalty_component": 0.0,
            "obstacle_penalty_component": 0.0,
        }

    def _sample_legal_position(self) -> np.ndarray:
        return np.array(
            [
                self.np_random.uniform(self.xy_sampling_margin, self.length - self.xy_sampling_margin),
                self.np_random.uniform(self.xy_sampling_margin, self.width - self.xy_sampling_margin),
                self.np_random.uniform(self.task_z_min, self.task_z_max),
            ],
            dtype=np.float32,
        )

    def _sample_task_point(self, reference_position: np.ndarray) -> np.ndarray:
        reference = self._validate_position(reference_position, "reference_position")
        for _ in range(self.sample_retry_limit):
            candidate = self._sample_legal_position()
            if np.linalg.norm(candidate - reference) >= self.minimum_task_distance:
                return candidate
        raise RuntimeError("failed to sample a task point at the configured minimum distance")

    def _validate_task_point(self, value: np.ndarray, reference_position: np.ndarray) -> np.ndarray:
        point = self._validate_position(value, "task_point")
        if np.linalg.norm(point - reference_position) < self.minimum_task_distance:
            raise ValueError("task_point violates minimum_task_distance")
        return point

    def _validate_position(self, value: np.ndarray, name: str) -> np.ndarray:
        position = np.asarray(value, dtype=np.float32)
        upper = np.array([self.length, self.width, self.height], dtype=np.float32) - self.safe_radius
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError(f"{name} must be a finite (3,) vector")
        if np.any(position < self.safe_radius) or np.any(position > upper):
            raise ValueError(f"{name} lies outside the legal map interior")
        return position.copy()

    def _validate_velocity(self, value: np.ndarray) -> np.ndarray:
        velocity = np.asarray(value, dtype=np.float32)
        if velocity.shape != (3,) or not np.all(np.isfinite(velocity)):
            raise ValueError("velocity must be a finite (3,) vector")
        if np.linalg.norm(velocity[:2]) > self.horizontal_v_max + eps:
            raise ValueError("horizontal velocity exceeds its physical limit")
        if abs(float(velocity[2])) > self.vertical_v_max + eps:
            raise ValueError("vertical velocity exceeds its physical limit")
        return velocity.copy()

    def _info(
        self,
        *,
        task_completed_now: bool = False,
        charger_reached_now: bool = False,
        battery_cycle_end: bool = False,
        switched_now: bool = False,
        boundary_contact: bool = False,
        obstacle_collision: bool = False,
        realized_energy_cost: float = 0.0,
        transition_dt: float = 0.0,
        physics_substeps: int = 0,
        instantaneous_service_reset: bool = False,
        task_stuck: bool = False,
        end_reason: str | None = None,
        reward_components: dict[str, float] | None = None,
        completed_goal_evaluation: dict[str, object] | None = None,
    ) -> dict[str, object]:
        estimate = self.last_mission_estimate
        quantiles = self.last_quantile_prediction
        return {
            "phase": int(self.phase),
            "mode": self.mode.value,
            "active_goal": self.active_goal.copy(),
            "current_task_point": self.current_task_point.copy(),
            "charger_position": self.charger_position,
            "tasks_completed": int(self.tasks_completed),
            "tasks_in_current_battery_cycle": int(self.tasks_in_current_battery_cycle),
            "task_completed_now": bool(task_completed_now),
            "charger_reached_now": bool(charger_reached_now),
            "battery_cycle_end": bool(battery_cycle_end),
            "battery_cycle_id": int(self.battery_cycle_id),
            "battery_cycle_record": self.last_battery_cycle_record,
            "switched_now": bool(switched_now),
            "task_to_charger_count": int(self.task_to_charger_count),
            "remaining_energy": float(self.agent.energy),
            "remaining_energy_fraction": (
                None
                if self.operational_energy_capacity is None
                else float(self.agent.energy / self.operational_energy_capacity)
            ),
            "battery_capacity": self.operational_energy_capacity,
            "battery_capacity_source": self.battery_capacity_source,
            "energy_reserve": self.energy_reserve,
            "energy_reserve_fraction": self.energy_reserve_fraction,
            "realized_energy_cost": float(realized_energy_cost),
            "energy_unit": ENERGY_UNIT,
            "cumulative_virtual_energy": float(self.cumulative_virtual_energy),
            "transition_dt": float(transition_dt),
            "interval_mean_power": (
                0.0 if transition_dt <= 0.0 else float(realized_energy_cost / transition_dt)
            ),
            "physics_substeps": int(physics_substeps),
            "instantaneous_service_reset": bool(instantaneous_service_reset),
            "td_loss": self.last_td_loss,
            "td_update_count": 0 if self.energy_estimator is None else int(self.energy_estimator.update_count),
            "td_replay_size": 0 if self.energy_estimator is None else len(self.energy_estimator.replay),
            "Q50": None if quantiles is None else float(quantiles[0]),
            "Q90": None if quantiles is None else float(quantiles[1]),
            "Q95": None if quantiles is None else float(quantiles[2]),
            "Q99": None if quantiles is None else float(quantiles[3]),
            "E_task_95": None if estimate is None else estimate.task_q95,
            "E_return_after_task_95": None if estimate is None else estimate.return_after_task_q95,
            "E_mission_95": None if estimate is None else estimate.mission_q95_composition,
            "E_return_now_95": None if estimate is None else estimate.return_now_q95,
            "boundary_contact": bool(boundary_contact),
            "boundary_collision_count": int(self.boundary_collision_count),
            "consecutive_boundary_contacts": int(self.consecutive_boundary_contacts),
            "maximum_consecutive_boundary_contacts": int(
                self.maximum_consecutive_boundary_contacts
            ),
            "obstacle_collision": bool(obstacle_collision),
            "obstacle_collision_count": int(self.obstacle_collision_count),
            "task_stuck": bool(task_stuck),
            "task_stuck_count": int(self.task_stuck_count),
            "end_reason": end_reason,
            "reward_components": self._zero_reward_components() if reward_components is None else reward_components,
            "completed_goal_evaluation": completed_goal_evaluation,
            "trajectory_length": len(self.agent_paths[0]),
            "episode_policy_steps": int(self.current_step),
            "speed": float(np.linalg.norm(self.agent.vel)),
            "horizontal_speed": float(np.linalg.norm(self.agent.vel[:2])),
            "vertical_speed": abs(float(self.agent.vel[2])),
            "distance_to_active_goal": float(np.linalg.norm(self.active_goal - self.agent.pos)),
            "goal_initial_distance": float(self._current_goal_initial_distance),
            "goal_path_length": float(self._current_goal_path_length),
            "goal_path_ratio": float(
                self._current_goal_path_length / max(self._current_goal_initial_distance, eps)
            ),
        }

    def summary(self) -> dict[str, object]:
        return {
            "phase": int(self.phase),
            "mode": self.mode.value,
            "tasks_completed": int(self.tasks_completed),
            "battery_cycle_id": int(self.battery_cycle_id),
            "battery_cycles_completed": int(self.battery_cycles_completed),
            "tasks_in_current_battery_cycle": int(self.tasks_in_current_battery_cycle),
            "remaining_energy": float(self.agent.energy),
            "cumulative_virtual_energy": float(self.cumulative_virtual_energy),
            "current_policy_step": int(self.current_step),
            "task_stuck_count": int(self.task_stuck_count),
            "boundary_collision_count": int(self.boundary_collision_count),
            "obstacle_collision_count": int(self.obstacle_collision_count),
        }

    def get_trajectory_log(self) -> list[dict[str, object]]:
        return [
            {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in row.items()}
            for row in self.trajectory_log
        ]

    def set_render_overlay(self, **values: object) -> None:
        self.render_overlay = dict(values)

    def _render_title(self) -> str:
        distance = float(np.linalg.norm(self.active_goal - self.agent.pos))
        estimate = self.last_mission_estimate
        values = {
            "step": self.current_step,
            "phase": int(self.phase),
            "mode": self.mode.value,
            "tasks": self.tasks_completed,
            "speed": float(np.linalg.norm(self.agent.vel)),
            "distance": distance,
            "energy": self.last_realized_energy,
            "Q95": None if self.last_quantile_prediction is None else float(self.last_quantile_prediction[2]),
            "mission": None if estimate is None else estimate.mission_q95_composition,
            "return": None if estimate is None else estimate.return_now_q95,
            "battery": "disabled" if not self.finite_energy_enabled else f"{self.agent.energy:.2f}",
            "cycle": self.battery_cycle_id,
            **self.render_overlay,
        }
        return (
            f"step={values['step']} phase={values['phase']} mode={values['mode']} tasks={values['tasks']}\n"
            f"speed={values['speed']:.2f}m/s distance={values['distance']:.1f}m "
            f"step_energy={values['energy']:.4f} Q95={values['Q95']}\n"
            f"mission95={values['mission']} return95={values['return']} "
            f"battery={values['battery']} cycle={values['cycle']}"
        )

    def _set_integer_ticks(self, axis, is_3d: bool = False) -> None:
        axis.set_xticks(np.linspace(0.0, self.length, 5))
        axis.set_yticks(np.linspace(0.0, self.width, 5))
        if is_3d:
            axis.set_zticks(np.linspace(0.0, self.height, 5))

    def _render_2d(self, axis) -> None:
        color = self._agent_color(0)
        trajectory = self._agent_trajectory(0, self.agent)
        if len(trajectory) > 1:
            axis.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.9, linewidth=1.8)
        axis.scatter(*self.current_task_point[:2], c=["#2ca02c"], marker="X", s=125, edgecolors="black", label="current task")
        axis.scatter(*self.agent.pos[:2], c=[color], s=90, edgecolors="black", label="UAV")
        axis.scatter(*self.charger_position[:2], c=["#1f77b4"], marker="P", s=145, edgecolors="black", label="charging station")
        axis.set_xlim(0.0, self.length)
        axis.set_ylim(0.0, self.width)
        axis.set_aspect("equal", adjustable="box")
        self._set_integer_ticks(axis)
        self._set_axis_labels(axis)
        self._style_axes(axis)
        self._add_render_legend(axis)
        axis.set_title(self._render_title(), fontsize=8)

    def _render_3d(self, axis) -> None:
        color = self._agent_color(0)
        trajectory = self._agent_trajectory(0, self.agent)
        if len(trajectory) > 1:
            axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, alpha=0.9, linewidth=1.8)
        axis.scatter(*self.current_task_point, c=["#2ca02c"], marker="X", s=125, edgecolors="black", label="current task")
        axis.scatter(*self.agent.pos, c=[color], s=70, edgecolors="black", label="UAV")
        axis.scatter(*self.charger_position, c=["#1f77b4"], marker="P", s=130, edgecolors="black", label="charging station")
        axis.set_xlim(0.0, self.length)
        axis.set_ylim(0.0, self.width)
        axis.set_zlim(0.0, self.height)
        self._set_integer_ticks(axis, is_3d=True)
        self._set_axis_labels(axis, is_3d=True)
        self._set_vertical_z_view(axis)
        self._style_axes(axis, is_3d=True)
        self._add_render_legend(axis, is_3d=True)
        axis.set_title(self._render_title(), fontsize=8)
        try:
            axis.set_box_aspect((self.length, self.width, self.height * self.render_vertical_exaggeration))
        except AttributeError:
            pass

    def render(self, view: str | None = None):
        return LegacyUAVEnv.render(self, show=self.render_mode == "human", view=view)

    def close(self) -> None:
        LegacyUAVEnv.close(self)


UAVEnergyDeliverySAC = UAVEnergyDeliverySACEnv
UAVEnv = UAVEnergyDeliverySACEnv


__all__ = [
    "ENERGY_GAMMA",
    "ENERGY_QUANTILES",
    "ENERGY_UNIT",
    "SWITCH_QUANTILE",
    "EnergyTDTransition",
    "GoalConditionedQuantileTDEnergyEstimator",
    "GoalConditionedTDEnergyEstimator",
    "MissionEnergyEstimate",
    "OnlineScalarTDEnergyEstimator",
    "SACTrainingPhase",
    "UAVEnergyDeliverySAC",
    "UAVEnergyDeliverySACEnv",
    "UAVEnv",
]
