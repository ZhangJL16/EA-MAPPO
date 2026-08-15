from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Protocol

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from torch import nn

from envs.UAVEnergyDelivery import UAVAgent, UAVEnv as LegacyUAVEnv, eps
from review_bundle.safety.energy.critics import ScalarEnergyCritic
from review_bundle.safety.energy.td import scalar_ssp_target
from review_bundle.safety.switching import EnergySwitchController, SortieMode


class SACTrainingPhase(IntEnum):
    NAVIGATION = 1
    ENERGY_MANAGED = 2


class ChargerActionProvider(Protocol):
    def __call__(self, charger_observation: np.ndarray) -> np.ndarray: ...


class ReturnEnergyEstimator(Protocol):
    def predict(self, charger_observation: np.ndarray, charger_action: np.ndarray) -> float: ...

    def observe_transition(
        self,
        charger_observation: np.ndarray,
        charger_action: np.ndarray,
        energy_cost: float,
        next_charger_observation: np.ndarray,
        next_charger_action: np.ndarray,
        charger_hit: bool,
    ) -> float | None: ...


@dataclass(frozen=True)
class EnergyTDTransition:
    state: np.ndarray
    action: np.ndarray
    cost: float
    next_state: np.ndarray
    next_action: np.ndarray
    charger_hit: bool


class OnlineScalarTDEnergyEstimator:
    def __init__(
        self,
        *,
        state_dim: int = 7,
        action_dim: int = 3,
        hidden_dim: int = 128,
        learning_rate: float = 3e-4,
        batch_size: int = 128,
        replay_capacity: int = 100_000,
        learning_starts: int = 512,
        updates_per_transition: int = 1,
        target_tau: float = 0.01,
        initial_output: float = 1.0,
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state and action dimensions must be positive")
        if batch_size <= 0 or replay_capacity < batch_size:
            raise ValueError("replay capacity must accommodate a positive batch")
        if learning_starts < batch_size:
            raise ValueError("learning_starts must be at least batch_size")
        if updates_per_transition <= 0:
            raise ValueError("updates_per_transition must be positive")
        if not 0.0 < target_tau <= 1.0:
            raise ValueError("target_tau must lie in (0, 1]")
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.input_dim = self.state_dim + self.action_dim
        self.batch_size = int(batch_size)
        self.learning_starts = int(learning_starts)
        self.updates_per_transition = int(updates_per_transition)
        self.target_tau = float(target_tau)
        self.device = torch.device(device)
        self.model = ScalarEnergyCritic(
            self.input_dim,
            hidden_dim=hidden_dim,
            initial_output=initial_output,
        ).to(self.device)
        self.target_model = ScalarEnergyCritic(
            self.input_dim,
            hidden_dim=hidden_dim,
            initial_output=initial_output,
        ).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.replay: deque[EnergyTDTransition] = deque(maxlen=int(replay_capacity))
        self.rng = np.random.default_rng(seed)
        self.update_count = 0

    def predict(self, charger_observation: np.ndarray, charger_action: np.ndarray) -> float:
        features = self._features(charger_observation, charger_action)
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(features).item()
        return float(prediction)

    def observe_transition(
        self,
        charger_observation: np.ndarray,
        charger_action: np.ndarray,
        energy_cost: float,
        next_charger_observation: np.ndarray,
        next_charger_action: np.ndarray,
        charger_hit: bool,
    ) -> float | None:
        state = self._vector(charger_observation, self.state_dim, "charger_observation")
        action = self._vector(charger_action, self.action_dim, "charger_action")
        next_state = self._vector(next_charger_observation, self.state_dim, "next_charger_observation")
        next_action = self._vector(next_charger_action, self.action_dim, "next_charger_action")
        cost = float(energy_cost)
        if not np.isfinite(cost) or cost < 0.0:
            raise ValueError("energy_cost must be finite and nonnegative")
        self.replay.append(
            EnergyTDTransition(
                state.copy(),
                action.copy(),
                cost,
                next_state.copy(),
                next_action.copy(),
                bool(charger_hit),
            )
        )
        if len(self.replay) < self.learning_starts:
            return None
        losses = [self._update_once() for _ in range(self.updates_per_transition)]
        return float(np.mean(losses))

    def _update_once(self) -> float:
        indices = self.rng.integers(0, len(self.replay), size=self.batch_size)
        batch = [self.replay[int(index)] for index in indices]
        states = torch.as_tensor(np.stack([row.state for row in batch]), device=self.device)
        actions = torch.as_tensor(np.stack([row.action for row in batch]), device=self.device)
        costs = torch.as_tensor([row.cost for row in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([row.next_state for row in batch]), device=self.device)
        next_actions = torch.as_tensor(np.stack([row.next_action for row in batch]), device=self.device)
        charger_hits = torch.as_tensor([row.charger_hit for row in batch], device=self.device)
        self.model.train()
        predictions = self.model(torch.cat((states, actions), dim=1))
        with torch.no_grad():
            next_values = self.target_model(torch.cat((next_states, next_actions), dim=1))
            targets = scalar_ssp_target(costs, next_values, charger_hits, gamma=1.0)
        loss = nn.functional.mse_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            for target_parameter, parameter in zip(self.target_model.parameters(), self.model.parameters()):
                target_parameter.mul_(1.0 - self.target_tau).add_(parameter, alpha=self.target_tau)
        self.update_count += 1
        return float(loss.detach().cpu().item())

    def _features(self, observation: np.ndarray, action: np.ndarray) -> torch.Tensor:
        state = self._vector(observation, self.state_dim, "charger_observation")
        control = self._vector(action, self.action_dim, "charger_action")
        return torch.as_tensor(
            np.concatenate((state, control), dtype=np.float32)[None, :],
            device=self.device,
        )

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite ({size},) vector")
        return vector


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

    def update_velocity(self, normalized_action: np.ndarray, time_step: float) -> np.ndarray:
        action = np.asarray(normalized_action, dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("SAC action must be a finite (3,) vector")
        action = np.clip(action, -1.0, 1.0)
        horizontal = action[:2]
        horizontal_norm = float(np.linalg.norm(horizontal))
        if horizontal_norm > 1.0:
            horizontal = horizontal / horizontal_norm
        acceleration = np.array(
            [
                horizontal[0] * self.horizontal_a_max,
                horizontal[1] * self.horizontal_a_max,
                action[2] * self.vertical_a_max,
            ],
            dtype=np.float32,
        )
        self.vel += acceleration * float(time_step)
        horizontal_speed = float(np.linalg.norm(self.vel[:2]))
        if horizontal_speed > self.horizontal_v_max:
            self.vel[:2] *= self.horizontal_v_max / (horizontal_speed + eps)
        self.vel[2] = np.clip(self.vel[2], -self.vertical_v_max, self.vertical_v_max)
        return acceleration


class UAVEnergyDeliverySACEnv(gym.Env, LegacyUAVEnv):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    def __init__(
        self,
        *,
        length: float = 4000.0,
        width: float = 4000.0,
        height: float = 4.0,
        dt: float = 0.2,
        horizontal_v_max: float = 12.0,
        vertical_v_max: float = 3.0,
        horizontal_a_max: float = 3.0,
        vertical_a_max: float = 2.0,
        goal_radius: float = 5.0,
        near_goal_distance: float = 80.0,
        minimum_task_distance: float = 80.0,
        sampling_margin: float = 5.0,
        episode_limit: int = 10_000,
        task_completion_reward: float = 10.0,
        progress_reward_weight: float = 1.0,
        velocity_reward_weight: float = 0.1,
        time_penalty: float = 0.01,
        boundary_penalty: float = 1.2,
        safe_radius: float = 0.05,
        operational_energy_capacity: float = 100.0,
        energy_cost_per_step: float | None = None,
        energy_depletion_fraction: float = 0.5,
        charger_position: np.ndarray | None = None,
        charger_radius: float | None = None,
        phase: int | SACTrainingPhase = SACTrainingPhase.NAVIGATION,
        render_mode: str | None = None,
    ) -> None:
        if length <= 0.0 or width <= 0.0 or height <= 0.0:
            raise ValueError("map dimensions must be positive")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if min(horizontal_v_max, vertical_v_max, horizontal_a_max, vertical_a_max) <= 0.0:
            raise ValueError("velocity and acceleration limits must be positive")
        if goal_radius <= 0.0 or near_goal_distance <= goal_radius:
            raise ValueError("near_goal_distance must exceed a positive goal_radius")
        if minimum_task_distance < goal_radius:
            raise ValueError("minimum_task_distance must be at least goal_radius")
        if sampling_margin < safe_radius:
            raise ValueError("sampling_margin must cover the UAV radius")
        if episode_limit <= 0:
            raise ValueError("episode_limit must be positive")
        if operational_energy_capacity <= 0.0:
            raise ValueError("operational_energy_capacity must be positive")
        if energy_cost_per_step is None:
            depletion_steps = max(1.0, float(episode_limit) * float(energy_depletion_fraction))
            energy_cost_per_step = operational_energy_capacity / depletion_steps
        if energy_cost_per_step <= 0.0:
            raise ValueError("energy_cost_per_step must be positive")
        selected_charger_radius = goal_radius if charger_radius is None else float(charger_radius)
        if selected_charger_radius <= 0.0:
            raise ValueError("charger_radius must be positive")
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
            episode_limit=int(episode_limit),
            total_orders=1,
            max_active_orders=1,
            initial_energy=float(operational_energy_capacity),
            energy_decay_per_step=float(energy_cost_per_step),
            charging_capacity=1,
            charging_station_count=1,
            charging_radius=selected_charger_radius,
            charging_station_pos=charger_position,
        )
        self.time_step = float(dt)
        self.horizontal_v_max = float(horizontal_v_max)
        self.vertical_v_max = float(vertical_v_max)
        self.horizontal_a_max = float(horizontal_a_max)
        self.vertical_a_max = float(vertical_a_max)
        self.goal_tolerance = float(goal_radius)
        self.near_goal_distance = float(near_goal_distance)
        self.minimum_task_distance = float(minimum_task_distance)
        self.sampling_margin = float(sampling_margin)
        self.max_cycles = int(episode_limit)
        self.task_completion_reward = float(task_completion_reward)
        self.progress_reward_weight = float(progress_reward_weight)
        self.velocity_reward_weight = float(velocity_reward_weight)
        self.time_penalty = float(time_penalty)
        self.boundary_penalty = float(boundary_penalty)
        self.safe_radius = float(safe_radius)
        self.operational_energy_capacity = float(operational_energy_capacity)
        self.energy_cost_per_step = float(energy_cost_per_step)
        self.render_mode = render_mode
        self.phase = SACTrainingPhase(int(phase))
        self.current_task_point = np.zeros(3, dtype=np.float32)
        self.tasks_completed = 0
        self.battery_cycles_completed = 0
        self.mode = SortieMode.TASK
        self.switch_controller: EnergySwitchController | None = None
        self.energy_estimator: ReturnEnergyEstimator | None = None
        self.charger_action_provider: ChargerActionProvider | None = None
        self.last_energy_prediction: float | None = None
        self.last_td_loss: float | None = None
        self.trajectory_log: list[dict[str, object]] = []
        self.orders = []
        self.goals = []
        self.obstacles = []
        self.action_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(
            np.array([-1.0] * 6 + [0.0], dtype=np.float32),
            np.ones(7, dtype=np.float32),
            dtype=np.float32,
        )

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
        if self.mode is SortieMode.CHARGER_COMMITTED:
            return self.charger_position
        return self.current_task_point.copy()

    @property
    def finite_energy_enabled(self) -> bool:
        return self.phase is SACTrainingPhase.ENERGY_MANAGED

    def enable_phase_two(
        self,
        *,
        energy_estimator: ReturnEnergyEstimator,
        charger_action_provider: ChargerActionProvider,
        reserve: float,
    ) -> None:
        self.phase = SACTrainingPhase.ENERGY_MANAGED
        self.energy_estimator = energy_estimator
        self.charger_action_provider = charger_action_provider
        self.switch_controller = EnergySwitchController(reserve=reserve)
        self.mode = SortieMode.TASK

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
        if self.phase is SACTrainingPhase.ENERGY_MANAGED:
            if self.energy_estimator is None or self.charger_action_provider is None or self.switch_controller is None:
                raise RuntimeError("Phase 2 requires an energy estimator, charger policy, and switch controller")
            self.switch_controller.complete_charge_and_start_sortie()
            default_start = self.charger_position
        else:
            default_start = self._sample_legal_position()
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
                initial_energy=self.operational_energy_capacity,
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
        self.current_step = 0
        self.mode = SortieMode.TASK
        self.last_energy_prediction = None
        self.last_td_loss = None
        self.agent.goal = self.current_task_point.copy()
        self.agent.reached = False
        self.agent_paths = [[start.copy()]]
        self.trajectory_log = []
        if self.phase is SACTrainingPhase.ENERGY_MANAGED:
            self._refresh_energy_decision()
        observation = self._active_goal_observation()
        return observation, self._info(
            task_completed_now=False,
            switched_now=self.mode is SortieMode.CHARGER_COMMITTED,
            boundary_contact=False,
            energy_cost=0.0,
            battery_cycle_end_reason=None,
        )

    def step(self, action: np.ndarray):
        if not self.agents:
            raise RuntimeError("reset must be called before step")
        control = np.asarray(action, dtype=np.float32)
        if control.shape != (3,) or not np.all(np.isfinite(control)):
            raise ValueError("action must be a finite (3,) vector")
        control = np.clip(control, -1.0, 1.0)
        self.current_step += 1
        mode_before = self.mode
        active_goal_before = self.active_goal
        distance_before = float(np.linalg.norm(active_goal_before - self.agent.pos))
        charger_observation_before = self.charger_relative_observation()
        acceleration = self.agent.update_velocity(control, self.time_step)
        self.agent.preview_position(self.time_step)
        boundary_contact, _, _ = self._apply_boundary_constraints(self.agent)
        self.agent.pos = self.agent.prev_pos.copy()
        distance_after = float(np.linalg.norm(active_goal_before - self.agent.pos))
        progress = distance_before - distance_after
        energy_cost = self.energy_cost_per_step
        if self.finite_energy_enabled:
            self.agent.energy = max(0.0, self.agent.energy - energy_cost)
        task_completed_now = False
        if mode_before is SortieMode.TASK and distance_after <= self.goal_tolerance:
            task_completed_now = True
            self.tasks_completed += 1
            retained_position = self.agent.pos.copy()
            retained_velocity = self.agent.vel.copy()
            self.current_task_point = self._sample_task_point(retained_position)
            self.agent.pos = retained_position
            self.agent.prev_pos = retained_position.copy()
            self.agent.vel = retained_velocity
        charger_hit = bool(
            mode_before is SortieMode.CHARGER_COMMITTED
            and np.linalg.norm(self.agent.pos - self.charger_position) <= self.charging_radius
        )
        energy_exhausted = bool(self.finite_energy_enabled and self.agent.energy <= eps)
        terminated = charger_hit or energy_exhausted
        battery_cycle_end_reason = "charger_reached" if charger_hit else "energy_exhausted" if energy_exhausted else None
        if terminated:
            self.battery_cycles_completed += 1
        next_charger_observation = self.charger_relative_observation()
        if mode_before is SortieMode.CHARGER_COMMITTED and self.energy_estimator is not None:
            next_charger_action = (
                np.zeros(3, dtype=np.float32)
                if charger_hit
                else self._charger_action(next_charger_observation)
            )
            self.last_td_loss = self.energy_estimator.observe_transition(
                charger_observation_before,
                control,
                energy_cost,
                next_charger_observation,
                next_charger_action,
                charger_hit,
            )
        switched_now = False
        if self.finite_energy_enabled and not terminated:
            switched_now = self._refresh_energy_decision()
        self.agent.goal = self.active_goal
        self.agent_paths[0].append(self.agent.pos.copy())
        self.agent.collided = bool(boundary_contact)
        goal_direction = active_goal_before - self.agent.pos
        goal_norm = float(np.linalg.norm(goal_direction))
        velocity_alignment = 0.0
        if goal_norm > eps:
            velocity_alignment = float(np.dot(self.agent.vel, goal_direction / goal_norm)) / self.horizontal_v_max
        reward = self.progress_reward_weight * progress
        reward += self.velocity_reward_weight * max(0.0, velocity_alignment)
        reward -= self.time_penalty
        reward -= self.boundary_penalty * float(boundary_contact)
        reward += self.task_completion_reward * float(task_completed_now)
        truncated = bool(
            self.phase is SACTrainingPhase.NAVIGATION
            and self.current_step >= self.max_cycles
            and not terminated
        )
        info = self._info(
            task_completed_now=task_completed_now,
            switched_now=switched_now,
            boundary_contact=boundary_contact,
            energy_cost=energy_cost,
            battery_cycle_end_reason=battery_cycle_end_reason,
        )
        info.update(
            {
                "progress": progress,
                "physical_acceleration": acceleration.copy(),
                "mode_before": mode_before.value,
                "active_goal_before": active_goal_before.copy(),
                "charger_observation": next_charger_observation.copy(),
            }
        )
        self.trajectory_log.append(
            {
                "step": self.current_step,
                "position": self.agent.pos.copy(),
                "velocity": self.agent.vel.copy(),
                "action": control.copy(),
                "physical_acceleration": acceleration.copy(),
                "active_goal": self.active_goal.copy(),
                "task_point": self.current_task_point.copy(),
                "mode": self.mode.value,
                "remaining_energy": float(self.agent.energy),
                "energy_cost": float(energy_cost),
                "task_completed_now": task_completed_now,
                "charger_hit": charger_hit,
                "energy_exhausted": energy_exhausted,
                "boundary_contact": bool(boundary_contact),
            }
        )
        return self._active_goal_observation(), float(reward), terminated, truncated, info

    def observation_for_goal(self, goal: np.ndarray) -> np.ndarray:
        target = self._validate_position(goal, "goal")
        velocity = np.array(
            [
                self.agent.vel[0] / self.horizontal_v_max,
                self.agent.vel[1] / self.horizontal_v_max,
                self.agent.vel[2] / self.vertical_v_max,
            ],
            dtype=np.float32,
        )
        delta = target - self.agent.pos
        distance = float(np.linalg.norm(delta))
        direction = delta / max(distance, eps)
        observation = np.concatenate(
            (
                velocity,
                direction.astype(np.float32),
                np.array([np.clip(distance / self.near_goal_distance, 0.0, 1.0)], dtype=np.float32),
            )
        )
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def charger_relative_observation(self) -> np.ndarray:
        return self.observation_for_goal(self.charger_position)

    def _active_goal_observation(self) -> np.ndarray:
        return self.observation_for_goal(self.active_goal)

    def _refresh_energy_decision(self) -> bool:
        if self.energy_estimator is None or self.charger_action_provider is None or self.switch_controller is None:
            raise RuntimeError("energy management is not bound")
        charger_observation = self.charger_relative_observation()
        charger_action = self._charger_action(charger_observation)
        predicted_return = self.energy_estimator.predict(charger_observation, charger_action)
        self.last_energy_prediction = float(predicted_return)
        decision = self.switch_controller.decide(
            remaining_energy=float(self.agent.energy),
            predicted_return_energy=float(predicted_return),
            task_goal=self.current_task_point,
            charger_goal=self.charger_position,
        )
        self.mode = decision.mode
        self.agent.goal = decision.active_goal.astype(np.float32)
        return bool(decision.switched_now)

    def _charger_action(self, charger_observation: np.ndarray) -> np.ndarray:
        if self.charger_action_provider is None:
            raise RuntimeError("charger action provider is not bound")
        action = np.asarray(self.charger_action_provider(charger_observation.copy()), dtype=np.float32)
        if action.shape != (3,) or not np.all(np.isfinite(action)):
            raise ValueError("charger action provider must return a finite (3,) action")
        return np.clip(action, -1.0, 1.0)

    def _sample_legal_position(self) -> np.ndarray:
        low = np.array([self.sampling_margin] * 3, dtype=np.float32)
        high = np.array(
            [
                self.length - self.sampling_margin,
                self.width - self.sampling_margin,
                self.height - self.sampling_margin,
            ],
            dtype=np.float32,
        )
        if high[2] <= low[2]:
            low[2] = self.safe_radius
            high[2] = self.height - self.safe_radius
        if np.any(high <= low):
            raise ValueError("sampling margin leaves no legal map interior")
        return self.np_random.uniform(low, high).astype(np.float32)

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
            raise ValueError("start_velocity must be a finite (3,) vector")
        horizontal_speed = float(np.linalg.norm(velocity[:2]))
        if horizontal_speed > self.horizontal_v_max + eps or abs(float(velocity[2])) > self.vertical_v_max + eps:
            raise ValueError("start_velocity exceeds the configured physical limits")
        return velocity.copy()

    def _info(
        self,
        *,
        task_completed_now: bool,
        switched_now: bool,
        boundary_contact: bool,
        energy_cost: float,
        battery_cycle_end_reason: str | None,
    ) -> dict[str, object]:
        return {
            "phase": int(self.phase),
            "mode": self.mode.value,
            "active_goal": self.active_goal.copy(),
            "current_task_point": self.current_task_point.copy(),
            "charger_position": self.charger_position,
            "tasks_completed": int(self.tasks_completed),
            "task_completed_now": bool(task_completed_now),
            "switched_now": bool(switched_now),
            "task_to_charger_count": 0 if self.switch_controller is None else self.switch_controller.task_to_charger_count,
            "remaining_energy": float(self.agent.energy),
            "energy_cost": float(energy_cost),
            "energy_prediction": self.last_energy_prediction,
            "td_loss": self.last_td_loss,
            "boundary_contact": bool(boundary_contact),
            "battery_cycle_end_reason": battery_cycle_end_reason,
            "trajectory_length": len(self.agent_paths[0]),
        }

    def summary(self) -> dict[str, object]:
        return {
            "phase": int(self.phase),
            "mode": self.mode.value,
            "tasks_completed": int(self.tasks_completed),
            "battery_cycles_completed": int(self.battery_cycles_completed),
            "remaining_energy": float(self.agent.energy),
            "current_step": int(self.current_step),
            "trajectory_length": len(self.agent_paths[0]),
            "task_to_charger_count": 0 if self.switch_controller is None else self.switch_controller.task_to_charger_count,
        }

    def get_trajectory_log(self) -> list[dict[str, object]]:
        copied = []
        for row in self.trajectory_log:
            copied.append(
                {
                    key: value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in row.items()
                }
            )
        return copied

    def _set_integer_ticks(self, ax, is_3d: bool = False) -> None:
        ax.set_xticks(np.linspace(0.0, self.length, 5))
        ax.set_yticks(np.linspace(0.0, self.width, 5))
        if is_3d:
            ax.set_zticks(np.linspace(0.0, self.height, 5))

    def _render_2d(self, ax) -> None:
        color = self._agent_color(0)
        trajectory = self._agent_trajectory(0, self.agent)
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.9, linewidth=1.8)
        ax.scatter(
            self.current_task_point[0],
            self.current_task_point[1],
            c=["#2ca02c"],
            marker="X",
            s=125,
            edgecolors="black",
            linewidths=0.8,
            label="current task",
        )
        ax.scatter(
            self.agent.pos[0],
            self.agent.pos[1],
            c=[color],
            s=90,
            edgecolors="black",
            linewidths=0.8,
            label="UAV",
        )
        station = self.charger_position
        ax.scatter(
            station[0],
            station[1],
            c=["#1f77b4"],
            marker="P",
            s=145,
            edgecolors="black",
            linewidths=0.8,
            label="charging station",
        )
        ax.set_xlim(0.0, self.length)
        ax.set_ylim(0.0, self.width)
        ax.set_aspect("equal", adjustable="box")
        self._set_integer_ticks(ax)
        self._set_axis_labels(ax)
        self._style_axes(ax)
        self._add_render_legend(ax)

    def _render_3d(self, ax) -> None:
        color = self._agent_color(0)
        trajectory = self._agent_trajectory(0, self.agent)
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color=color, alpha=0.9, linewidth=1.8)
        ax.scatter(
            *self.current_task_point,
            c=["#2ca02c"],
            marker="X",
            s=125,
            edgecolors="black",
            linewidths=0.8,
            label="current task",
        )
        ax.scatter(
            *self.agent.pos,
            c=[color],
            s=70,
            edgecolors="black",
            linewidths=0.8,
            label="UAV",
        )
        station = self.charger_position
        ax.scatter(
            *station,
            c=["#1f77b4"],
            marker="P",
            s=130,
            edgecolors="black",
            linewidths=0.8,
            label="charging station",
        )
        ax.set_xlim(0.0, self.length)
        ax.set_ylim(0.0, self.width)
        ax.set_zlim(0.0, self.height)
        self._set_integer_ticks(ax, is_3d=True)
        self._set_axis_labels(ax, is_3d=True)
        self._set_vertical_z_view(ax)
        self._style_axes(ax, is_3d=True)
        self._add_render_legend(ax, is_3d=True)
        try:
            ax.set_box_aspect((self.length, self.width, self.height))
        except AttributeError:
            pass

    def render(self, view: str | None = None):
        show = self.render_mode == "human"
        return LegacyUAVEnv.render(self, show=show, view=view)

    def close(self) -> None:
        LegacyUAVEnv.close(self)


UAVEnergyDeliverySAC = UAVEnergyDeliverySACEnv
UAVEnv = UAVEnergyDeliverySACEnv


__all__ = [
    "OnlineScalarTDEnergyEstimator",
    "SACTrainingPhase",
    "UAVEnergyDeliverySAC",
    "UAVEnergyDeliverySACEnv",
    "UAVEnv",
]
