import itertools
import math
import os
import random

import numpy as np
from gymnasium import spaces

from common.auction import auction_assign_min_cost


boundary_length = 4.0
boundary_width = 4.0
boundary_height = 4.0
default_num_obstacles = 10
default_obstacle_radius_range = (0.16, 0.24)
default_initial_energy = 100.0
default_energy_depletion_fraction = 0.5
default_charging_station_count = None
default_charging_capacity = 2
default_charging_radius = 0.18
eps = 1e-6
TASK_IDLE = 0
TASK_ORDER = 1
TASK_CHARGE = 2
HIGH_MODE_CHARGE = 0
HIGH_MODE_ORDER = 1
legend_font_size_xy = 26
legend_font_size_3d = 26
ORDER_EXEC_DIAGNOSTIC_ATTRS = (
    "order_diag_order_steps",
    "order_diag_pickup_steps",
    "order_diag_delivery_steps",
    "order_diag_pickup_successes",
    "order_diag_delivery_successes",
    "order_diag_progress_sum",
    "order_diag_positive_progress_steps",
    "order_diag_regress_steps",
    "order_diag_target_distance_sum",
    "order_diag_charge_steps",
    "order_diag_idle_steps",
    "order_diag_order_obstacle_collisions",
    "order_diag_order_agent_collisions",
    "order_diag_charge_obstacle_collisions",
    "order_diag_charge_agent_collisions",
    "order_diag_idle_obstacle_collisions",
    "order_diag_idle_agent_collisions",
)


def _laser_angles(num_lasers):
    return np.linspace(0.0, 2.0 * math.pi, num_lasers, endpoint=False, dtype=np.float32)


def update_lasers_to_boundary(agent_pos, l_sensor, num_lasers, length, width):
    origin = np.asarray(agent_pos, dtype=np.float32)
    distances = np.full(num_lasers, l_sensor, dtype=np.float32)

    for idx, angle in enumerate(_laser_angles(num_lasers)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        ray_limits = []

        if abs(direction[0]) > eps:
            x_target = length if direction[0] > 0 else 0.0
            t_x = (x_target - origin[0]) / direction[0]
            if t_x >= 0:
                y_at_x = origin[1] + t_x * direction[1]
                if -eps <= y_at_x <= width + eps:
                    ray_limits.append(t_x)

        if abs(direction[1]) > eps:
            y_target = width if direction[1] > 0 else 0.0
            t_y = (y_target - origin[1]) / direction[1]
            if t_y >= 0:
                x_at_y = origin[0] + t_y * direction[0]
                if -eps <= x_at_y <= length + eps:
                    ray_limits.append(t_y)

        if ray_limits:
            distances[idx] = min(l_sensor, float(min(ray_limits)))

    return distances


def update_lasers_to_obstacle(agent_pos, obstacle_pos, radius, l_sensor, num_lasers):
    origin = np.asarray(agent_pos, dtype=np.float32)
    center = np.asarray(obstacle_pos, dtype=np.float32)
    distances = np.full(num_lasers, l_sensor, dtype=np.float32)

    oc = origin - center
    oc_dot = float(np.dot(oc, oc))
    radius_sq = float(radius * radius)

    for idx, angle in enumerate(_laser_angles(num_lasers)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        b = 2.0 * float(np.dot(direction, oc))
        c = oc_dot - radius_sq
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            continue

        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / 2.0
        t1 = (-b + sqrt_disc) / 2.0
        valid_ts = [t for t in (t0, t1) if t >= 0.0]
        if valid_ts:
            distances[idx] = min(l_sensor, float(min(valid_ts)))

    return distances


class UAVAgent:
    def __init__(
        self,
        number,
        pos,
        v_max,
        a_max,
        num_lasers,
        l_sensor,
        safe_radius,
        dim,
        initial_energy=default_initial_energy,
    ):
        self.number = number
        self.pos = np.asarray(pos, dtype=np.float32)
        self.prev_pos = np.asarray(pos, dtype=np.float32)
        self.last_pos = np.asarray(pos, dtype=np.float32)
        self.goal = np.asarray(pos, dtype=np.float32)
        self.vel = np.zeros(dim, dtype=np.float32)
        self.v_max = v_max
        self.a_max = a_max
        self.num_lasers = num_lasers
        self.l_sensor = l_sensor
        self.lasers = np.full((num_lasers,), l_sensor, dtype=np.float32)
        self.safe_radius = safe_radius
        self.dim = dim
        self.reached = True
        self.collided = False
        self.prev_collided = False
        self.assigned_order_id = None
        self.assigned_order_slot = None
        self.auction_order_slot = None
        self.charge_station_idx = None
        self.charge_slot_idx = None
        self.task_target = self.pos.copy()
        self.subgoal = self.pos.copy()
        self.original_subgoal = self.pos.copy()
        self.subgoal_test = False
        self.carrying_order = False
        self.current_task_type = TASK_IDLE
        self.completed_orders = 0
        self.initial_energy = float(initial_energy)
        self.energy = float(initial_energy)

    def has_energy(self):
        return self.energy > eps

    def consume_energy(self, amount):
        if not self.has_energy():
            self.energy = 0.0
            return
        self.energy = round(max(0.0, self.energy - float(amount)), 1)

    def charge_energy(self, amount):
        self.energy = round(
            min(self.initial_energy, self.energy + float(amount)),
            1,
        )

    def update_velocity(self, action, time_step):
        if not self.has_energy():
            self.vel[:] = 0.0
            return
        action = np.asarray(action, dtype=np.float32)
        norm = np.linalg.norm(action)
        if norm > self.a_max:
            action = action / (norm + eps) * self.a_max
        self.vel += action * time_step
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel = self.vel / (speed + eps) * self.v_max

    def preview_position(self, time_step):
        self.last_pos = self.pos.copy()
        if not self.has_energy():
            self.vel[:] = 0.0
            self.prev_pos = self.pos.copy()
            return
        self.prev_pos = self.pos + self.vel * time_step


class GoalPoint:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float32)


class DeliveryOrder:
    PENDING = "pending"
    ACTIVE = "active"
    ASSIGNED = "assigned"
    PICKED = "picked"
    COMPLETED = "completed"

    def __init__(self, order_id, pickup_pos, dropoff_pos):
        self.order_id = int(order_id)
        self.pickup_pos = np.asarray(pickup_pos, dtype=np.float32)
        self.dropoff_pos = np.asarray(dropoff_pos, dtype=np.float32)
        self.status = self.PENDING
        self.assigned_agent = None


class Obstacle:
    def __init__(self, length, width, radius_range=default_obstacle_radius_range):
        self.pos = np.array(
            [
                np.random.uniform(0.45, length - 0.45),
                np.random.uniform(0.45, width - 0.45),
            ],
            dtype=np.float32,
        )
        self.radius = float(np.random.uniform(*radius_range))
        self.vel = np.zeros(2, dtype=np.float32)


class UAVEnv:
    def __init__(
        self,
        dim_actions,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=6,
        num_targets=1,
        episode_limit=200,
        reset_retry_limit=20,
        sample_retry_limit=100,
        obstacle_crash_penalty=10.0,
        total_orders=8,
        max_active_orders=4,
        pickup_reward=3.0,
        delivery_reward=8.0,
        initial_energy=default_initial_energy,
        energy_decay_per_step=None,
        energy_depletion_fraction=default_energy_depletion_fraction,
        charging_capacity=default_charging_capacity,
        charging_station_count=default_charging_station_count,
        charging_radius=default_charging_radius,
        charging_rate=None,
        charging_station_pos=None,
        charge_mode_fraction=0.5,
        charge_dense_reward_scale=1.0,
        auction_enabled=True,
        fixed_charge_threshold_enabled=False,
        fixed_charge_threshold=0.35,
        fixed_charge_release_threshold=0.65,
    ):
        if dim_actions not in (2, 3):
            raise ValueError("Dimension must be 2 or 3")
        if int(total_orders) <= 0:
            raise ValueError("total_orders must be positive")
        if int(max_active_orders) <= 0:
            raise ValueError("max_active_orders must be positive")

        self.dim_actions = dim_actions
        self.length = length
        self.width = width
        self.height = height if dim_actions == 3 else 0.0
        self.num_obstacle = num_obstacle
        self.num_agents = num_hunters
        self.episode_limit = episode_limit
        self.reset_retry_limit = reset_retry_limit
        self.sample_retry_limit = sample_retry_limit
        self.obstacle_crash_penalty = float(obstacle_crash_penalty)
        self.total_orders = int(total_orders)
        self.max_active_orders = min(int(max_active_orders), self.total_orders)
        self.low_task_shape = 0
        # High-level action is encoded as [mode_id, progress].
        # mode_id is discrete: 0 selects charging, 1 selects order service.
        self.high_level_n_actions = 2
        self.high_level_mode_n_actions = 2
        self.charge_mode_fraction = float(np.clip(charge_mode_fraction, 0.01, 0.99))
        self.charge_mode_threshold = -1.0 + 2.0 * self.charge_mode_fraction
        self.charge_mode_center = -1.0 + self.charge_mode_fraction
        self.order_mode_center = self.charge_mode_fraction
        self._last_high_mode_train_mask = np.ones((self.num_agents, 1), dtype=np.float32)
        self.charge_dense_reward_scale = float(charge_dense_reward_scale)
        self.auction_enabled = bool(auction_enabled)
        self.fixed_charge_threshold_enabled = bool(fixed_charge_threshold_enabled)
        self.fixed_charge_threshold = float(fixed_charge_threshold)
        self.fixed_charge_release_threshold = float(fixed_charge_release_threshold)
        self.charge_action_id = -1
        self.pickup_reward = float(pickup_reward)
        self.delivery_reward = float(delivery_reward)
        self.initial_energy = float(initial_energy)
        self.energy_depletion_fraction = float(energy_depletion_fraction)
        if energy_decay_per_step is None:
            depletion_steps = max(
                1.0,
                float(episode_limit) * self.energy_depletion_fraction,
            )
            energy_decay_per_step = self.initial_energy / depletion_steps
        self.energy_decay_per_step = round(float(energy_decay_per_step), 1)
        if charging_station_count is None:
            charging_station_count = max(1, (int(self.num_agents) + 1) // 2)
        self.charging_station_count = int(max(1, charging_station_count))
        if charging_capacity is None:
            charging_capacity = 2
        self.charging_capacity = int(charging_capacity)
        self.charging_radius = float(charging_radius)
        self.charging_rate = round(
            float(
                4.0 * self.energy_decay_per_step
                if charging_rate is None
                else charging_rate
            ),
            1,
        )
        self.charging_station_positions = self._init_charging_station_positions(
            charging_station_pos
        )
        self.charging_station_pos = self.charging_station_positions[0].copy()
        self.charging_agent_ids = []
        self.charging_station_agent_ids = {
            station_idx: [] for station_idx in range(self.charging_station_count)
        }
        self.charging_slot_reservations = self._empty_charge_reservations()

        self.time_step = 0.4
        self.meta_period = 5
        self.reachable_subgoal_scale = 1.0
        self.intrinsic_reward_scale = 1.0
        self.intrinsic_distance_weight = 0.05
        self.intrinsic_success_bonus = 1.0
        self.delivery_intrinsic_progress_bonus = 0.0
        self.intrinsic_collision_penalty = 0.0
        self.order_progress_override = None
        self.energy_shield_enabled = False
        self.energy_margin_reserve_ratio = 0.05
        self.charge_queue_enabled = False
        self.charge_queue_radius = 0.24
        self.charge_dock_radius = 0.05
        self.charge_energy_threshold = 0.35
        self.charge_release_threshold = 0.65
        self.goal_tolerance = 0.12
        self.charge_dock_radius = min(
            0.65 * self.goal_tolerance,
            0.65 * self.charging_radius,
        )
        self.min_subgoal_progress = 1.25 * self.goal_tolerance
        self.v_max = 0.16
        self.a_max = 0.05
        self.safe_radius = 0.05
        self.risk_warning_margin = 0.06
        self.guard_prediction_margin = 0.04
        self.guard_prediction_horizon = 4
        self.velocity_reward_weight = 0.9
        self.obstacle_collision_penalty = 1.2
        self.agent_collision_penalty = 1.5
        self.repeat_collision_scale = 0.35
        self.l_sensor = 0.35
        self.num_lasers = 16
        self.msg_shape = self.dim_actions * 2 + 3

        self.max_cycles = episode_limit
        self.agents = []
        self.goals = []
        self.orders = []
        self.order_slots = [None for _ in range(self.max_active_orders)]
        self.available_order_slots = []
        self.active_order_ids = []
        self.next_order_id_to_activate = 0
        self.completed_order_count = 0
        self._last_step_order_status_before = [None for _ in range(self.num_agents)]
        self.obstacles = []
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.current_step = 0
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.collision_count = 0.0
        self.obstacle_collision_count = 0.0
        self.agent_collision_count = 0.0
        self._reset_high_level_diagnostics()
        self._reset_order_execution_diagnostics()
        self._render_fig = None
        self._render_has_shown = False

        obs_dim = (
            2 * self.dim_actions
            + self.num_lasers
            + (self.dim_actions + 1)
            + 1
            + self.low_task_shape
        )
        self.action_space = {
            f"agent_{i}": spaces.Box(
                low=-self.a_max,
                high=self.a_max,
                shape=(self.dim_actions,),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }
        self.observation_space = {
            f"agent_{i}": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }

    def _space_scale(self):
        return np.array(
            [self.length, self.width, self.height][: self.dim_actions],
            dtype=np.float32,
        )

    def _horizontal_pos(self, pos):
        return np.asarray(pos[:2], dtype=np.float32)

    def _cylinder_distance(self, pos, obstacle):
        return np.linalg.norm(self._horizontal_pos(pos) - obstacle.pos)

    def _distance_to_goal(self, agent):
        return float(np.linalg.norm(agent.goal - agent.pos))

    def set_meta_period(self, meta_period):
        self.meta_period = max(1, int(meta_period))

    def set_hrl_parameters(
        self,
        reachable_subgoal_scale=None,
        intrinsic_reward_scale=None,
        intrinsic_distance_weight=None,
        intrinsic_success_bonus=None,
        delivery_intrinsic_progress_bonus=None,
        intrinsic_collision_penalty=None,
        order_progress_override=None,
        energy_shield_enabled=None,
        energy_margin_reserve_ratio=None,
        charge_queue_enabled=None,
        charge_queue_radius=None,
    ):
        if reachable_subgoal_scale is not None:
            self.reachable_subgoal_scale = float(reachable_subgoal_scale)
        if intrinsic_reward_scale is not None:
            self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        if intrinsic_distance_weight is not None:
            self.intrinsic_distance_weight = float(intrinsic_distance_weight)
        if intrinsic_success_bonus is not None:
            self.intrinsic_success_bonus = float(intrinsic_success_bonus)
        if delivery_intrinsic_progress_bonus is not None:
            self.delivery_intrinsic_progress_bonus = float(
                delivery_intrinsic_progress_bonus
            )
        if intrinsic_collision_penalty is not None:
            self.intrinsic_collision_penalty = float(intrinsic_collision_penalty)
        if order_progress_override is not None:
            self.order_progress_override = float(
                np.clip(order_progress_override, -1.0, 1.0)
            )
        if energy_shield_enabled is not None:
            self.energy_shield_enabled = bool(energy_shield_enabled)
        if energy_margin_reserve_ratio is not None:
            self.energy_margin_reserve_ratio = float(
                np.clip(energy_margin_reserve_ratio, 0.0, 1.0)
            )
        if charge_queue_enabled is not None:
            self.charge_queue_enabled = bool(charge_queue_enabled)
        if charge_queue_radius is not None:
            min_radius = self.goal_tolerance + self.safe_radius + 0.02
            max_radius = 0.45 * float(np.max(self._space_scale()))
            self.charge_queue_radius = float(
                np.clip(charge_queue_radius, min_radius, max_radius)
            )

    def _max_reachable_subgoal_distance(self, agent):
        return (
            float(agent.v_max)
            * float(self.time_step)
            * float(max(1, self.meta_period))
            * float(self.reachable_subgoal_scale)
        )

    def project_to_reachable_subgoal(self, agent, target):
        target = np.asarray(target, dtype=np.float32)[: self.dim_actions]
        delta = target - agent.pos
        dist = float(np.linalg.norm(delta))
        max_dist = self._max_reachable_subgoal_distance(agent)
        if dist > max_dist > eps:
            delta = delta / (dist + eps) * max_dist
        return (agent.pos + delta).astype(np.float32)

    def _set_agent_subgoal(self, agent, target, task_type, project=True):
        if int(task_type) != TASK_CHARGE:
            self._release_charge_reservation(agent)
        target = np.asarray(target, dtype=np.float32)[: self.dim_actions]
        agent.task_target = target.copy()
        subgoal = self.project_to_reachable_subgoal(agent, target) if project else target
        agent.goal = subgoal.copy()
        agent.subgoal = subgoal.copy()
        agent.original_subgoal = subgoal.copy()
        agent.current_task_type = int(task_type)
        agent.reached = self._distance_to_goal(agent) <= self.goal_tolerance
        if agent.reached and self._is_at_task_terminal_target(agent):
            agent.vel[:] = 0.0
        return True

    def _clip_position_to_bounds(self, pos):
        pos = np.asarray(pos, dtype=np.float32).copy()
        upper = self._space_scale() - self.safe_radius
        lower = np.full(self.dim_actions, self.safe_radius, dtype=np.float32)
        return np.minimum(np.maximum(pos, lower), upper).astype(np.float32)

    def _set_agent_subgoal_from_delta(self, agent, delta, task_type, task_target=None):
        if int(task_type) != TASK_CHARGE:
            self._release_charge_reservation(agent)
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        if delta.size < self.dim_actions:
            delta = np.pad(delta, (0, self.dim_actions - delta.size))
        delta = delta[: self.dim_actions]
        norm = float(np.linalg.norm(delta))
        if norm > 1.0:
            delta = delta / (norm + eps)
        max_dist = self._max_reachable_subgoal_distance(agent)
        subgoal = self._clip_position_to_bounds(agent.pos + delta * max_dist)
        if task_target is None:
            task_target = subgoal
        agent.task_target = np.asarray(task_target, dtype=np.float32)[: self.dim_actions].copy()
        agent.goal = subgoal.copy()
        agent.subgoal = subgoal.copy()
        agent.original_subgoal = subgoal.copy()
        agent.current_task_type = int(task_type)
        agent.reached = self._distance_to_goal(agent) <= self.goal_tolerance
        if agent.reached and self._is_at_task_terminal_target(agent):
            agent.vel[:] = 0.0
        return True

    def _set_agent_subgoal_on_target_line(self, agent, action, task_type, task_target=None):
        if int(task_type) != TASK_CHARGE:
            self._release_charge_reservation(agent)
        if task_target is None:
            task_target = agent.pos.copy()
        target = np.asarray(task_target, dtype=np.float32)[: self.dim_actions]
        to_target = target - agent.pos
        target_dist = float(np.linalg.norm(to_target))

        if task_type == TASK_IDLE or target_dist <= eps:
            subgoal = agent.pos.copy()
        else:
            action = np.asarray(action, dtype=np.float32).reshape(-1)
            scalar = float(action[0]) if action.size else 0.0
            fraction = 0.5 * (np.clip(scalar, -1.0, 1.0) + 1.0)

            max_reachable = self._max_reachable_subgoal_distance(agent)
            line_length = min(target_dist, max_reachable)
            min_progress = min(line_length, self.min_subgoal_progress)
            if line_length > min_progress + eps:
                progress = min_progress + fraction * (line_length - min_progress)
            else:
                progress = line_length

            direction = to_target / (target_dist + eps)
            subgoal = agent.pos + direction * progress
            subgoal = self._clip_position_to_bounds(subgoal)

        agent.task_target = target.copy()
        agent.goal = subgoal.copy()
        agent.subgoal = subgoal.copy()
        agent.original_subgoal = subgoal.copy()
        agent.current_task_type = int(task_type)
        agent.reached = self._distance_to_goal(agent) <= self.goal_tolerance
        if agent.reached and self._is_at_task_terminal_target(agent):
            agent.vel[:] = 0.0
        return True

    def _is_at_task_terminal_target(self, agent):
        if agent.current_task_type == TASK_IDLE:
            return True
        if agent.current_task_type == TASK_CHARGE:
            return self._distance_to_charging_station(agent) <= self.goal_tolerance
        if (
            agent.current_task_type == TASK_ORDER
            and agent.assigned_order_id is not None
        ):
            order = self.orders[agent.assigned_order_id]
            if order.status == DeliveryOrder.ASSIGNED:
                target = order.pickup_pos
            elif order.status == DeliveryOrder.PICKED:
                target = order.dropoff_pos
            else:
                return True
            return bool(np.linalg.norm(agent.pos - target) <= self.goal_tolerance)
        target = getattr(agent, "task_target", None)
        if target is None:
            return True
        return bool(
            np.linalg.norm(
                agent.pos - np.asarray(target, dtype=np.float32)[: self.dim_actions]
            )
            <= self.goal_tolerance
        )

    def _advance_intermediate_subgoal_if_needed(self, agent):
        if not agent.reached or self._is_at_task_terminal_target(agent):
            return False
        if not self._agent_has_motion_task(agent):
            return False

        task_target = (
            getattr(agent, "task_target", self._nearest_charging_station_pos(agent.pos))
            if agent.current_task_type == TASK_CHARGE
            else getattr(agent, "task_target", agent.goal)
        )
        self._set_agent_subgoal_on_target_line(
            agent,
            np.array([1.0], dtype=np.float32),
            task_type=agent.current_task_type,
            task_target=task_target,
        )
        return not agent.reached

    def _charge_option_complete(self, agent):
        if agent.current_task_type != TASK_CHARGE:
            return True
        energy_ratio = agent.energy / (agent.initial_energy + eps)
        return bool(
            self._agent_ready_to_charge(agent)
            and energy_ratio >= self.charge_release_threshold
        )

    def _agent_ready_to_charge(self, agent):
        if agent.current_task_type != TASK_CHARGE:
            return False
        if not agent.reached:
            return False
        if self._distance_to_charging_station(agent) > self.goal_tolerance:
            return False
        return bool(np.linalg.norm(agent.vel) <= eps)

    def _agent_charge_locked(self, agent):
        return (
            agent.current_task_type == TASK_CHARGE
            and not self._charge_option_complete(agent)
        )

    def _agent_order_locked(self, agent):
        if agent.assigned_order_id is None or agent.assigned_order_slot is None:
            return False
        order = self._slot_order(agent.assigned_order_slot)
        return order is not None and order.status in (
            DeliveryOrder.ASSIGNED,
            DeliveryOrder.PICKED,
        )

    def _parse_high_level_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.high_level_mode_n_actions > 1 and action.size >= 2:
            mode_id = int(
                np.clip(
                    np.rint(float(action[0])),
                    HIGH_MODE_CHARGE,
                    HIGH_MODE_ORDER,
                )
            )
            return mode_id, float(action[1])

        mode_scalar = float(action[0]) if action.size else 0.0
        progress_scalar = float(action[1]) if action.size >= 2 else mode_scalar
        mode_id = (
            HIGH_MODE_CHARGE
            if mode_scalar < self.charge_mode_threshold
            else HIGH_MODE_ORDER
        )
        return mode_id, progress_scalar

    def _reset_high_level_diagnostics(self):
        self.high_diag_decision_count = 0.0
        self.high_diag_sampled_charge = 0.0
        self.high_diag_sampled_order = 0.0
        self.high_diag_executed_charge = 0.0
        self.high_diag_executed_order = 0.0
        self.high_diag_executed_idle = 0.0
        self.high_diag_order_locked = 0.0
        self.high_diag_charge_locked = 0.0
        self.high_diag_mode_train_mask_sum = 0.0
        self.high_diag_mode_train_mask_count = 0.0
        self.high_diag_auction_calls = 0.0
        self.high_diag_auction_candidates = 0.0
        self.high_diag_auction_orders = 0.0
        self.high_diag_auction_assignments = 0.0
        self.high_diag_order_commit_attempts = 0.0
        self.high_diag_order_commit_successes = 0.0
        self.high_diag_charge_fallbacks = 0.0

    def _high_diag_rate(self, numerator, denominator=None):
        denom = self.high_diag_decision_count if denominator is None else denominator
        return float(numerator / denom) if denom > eps else 0.0

    def _high_level_diagnostics_summary(self):
        return {
            "high_decision_count": float(self.high_diag_decision_count),
            "sampled_charge_rate": self._high_diag_rate(
                self.high_diag_sampled_charge
            ),
            "sampled_order_rate": self._high_diag_rate(
                self.high_diag_sampled_order
            ),
            "executed_charge_rate": self._high_diag_rate(
                self.high_diag_executed_charge
            ),
            "executed_order_rate": self._high_diag_rate(
                self.high_diag_executed_order
            ),
            "executed_idle_rate": self._high_diag_rate(
                self.high_diag_executed_idle
            ),
            "order_locked_rate": self._high_diag_rate(self.high_diag_order_locked),
            "charge_locked_rate": self._high_diag_rate(self.high_diag_charge_locked),
            "mode_train_mask_mean": self._high_diag_rate(
                self.high_diag_mode_train_mask_sum,
                self.high_diag_mode_train_mask_count,
            ),
            "auction_calls": float(self.high_diag_auction_calls),
            "auction_candidate_mean": self._high_diag_rate(
                self.high_diag_auction_candidates,
                self.high_diag_auction_calls,
            ),
            "auction_order_mean": self._high_diag_rate(
                self.high_diag_auction_orders,
                self.high_diag_auction_calls,
            ),
            "auction_assignment_mean": self._high_diag_rate(
                self.high_diag_auction_assignments,
                self.high_diag_auction_calls,
            ),
            "order_commit_attempt_rate": self._high_diag_rate(
                self.high_diag_order_commit_attempts
            ),
            "order_commit_success_rate": self._high_diag_rate(
                self.high_diag_order_commit_successes
            ),
            "charge_fallback_rate": self._high_diag_rate(
                self.high_diag_charge_fallbacks
            ),
        }

    def _reset_order_execution_diagnostics(self):
        self.order_diag_order_steps = 0.0
        self.order_diag_pickup_steps = 0.0
        self.order_diag_delivery_steps = 0.0
        self.order_diag_pickup_successes = 0.0
        self.order_diag_delivery_successes = 0.0
        self.order_diag_progress_sum = 0.0
        self.order_diag_positive_progress_steps = 0.0
        self.order_diag_regress_steps = 0.0
        self.order_diag_target_distance_sum = 0.0
        self.order_diag_charge_steps = 0.0
        self.order_diag_idle_steps = 0.0
        self.order_diag_order_obstacle_collisions = 0.0
        self.order_diag_order_agent_collisions = 0.0
        self.order_diag_charge_obstacle_collisions = 0.0
        self.order_diag_charge_agent_collisions = 0.0
        self.order_diag_idle_obstacle_collisions = 0.0
        self.order_diag_idle_agent_collisions = 0.0

    def _order_exec_rate(self, numerator, denominator):
        return float(numerator / denominator) if denominator > eps else 0.0

    def _order_execution_diagnostics_summary(self):
        order_collisions = (
            self.order_diag_order_obstacle_collisions
            + self.order_diag_order_agent_collisions
        )
        charge_collisions = (
            self.order_diag_charge_obstacle_collisions
            + self.order_diag_charge_agent_collisions
        )
        idle_collisions = (
            self.order_diag_idle_obstacle_collisions
            + self.order_diag_idle_agent_collisions
        )
        return {
            "order_exec_steps": float(self.order_diag_order_steps),
            "order_pickup_steps": float(self.order_diag_pickup_steps),
            "order_delivery_steps": float(self.order_diag_delivery_steps),
            "order_pickup_success_count": float(self.order_diag_pickup_successes),
            "order_delivery_success_count": float(self.order_diag_delivery_successes),
            "order_progress_mean": self._order_exec_rate(
                self.order_diag_progress_sum,
                self.order_diag_order_steps,
            ),
            "order_positive_progress_rate": self._order_exec_rate(
                self.order_diag_positive_progress_steps,
                self.order_diag_order_steps,
            ),
            "order_regress_rate": self._order_exec_rate(
                self.order_diag_regress_steps,
                self.order_diag_order_steps,
            ),
            "order_target_distance_mean": self._order_exec_rate(
                self.order_diag_target_distance_sum,
                self.order_diag_order_steps,
            ),
            "pickup_success_per_pickup_step": self._order_exec_rate(
                self.order_diag_pickup_successes,
                self.order_diag_pickup_steps,
            ),
            "delivery_success_per_delivery_step": self._order_exec_rate(
                self.order_diag_delivery_successes,
                self.order_diag_delivery_steps,
            ),
            "charge_exec_steps": float(self.order_diag_charge_steps),
            "idle_exec_steps": float(self.order_diag_idle_steps),
            "order_obstacle_collision_count": float(
                self.order_diag_order_obstacle_collisions
            ),
            "order_agent_collision_count": float(
                self.order_diag_order_agent_collisions
            ),
            "charge_obstacle_collision_count": float(
                self.order_diag_charge_obstacle_collisions
            ),
            "charge_agent_collision_count": float(
                self.order_diag_charge_agent_collisions
            ),
            "idle_obstacle_collision_count": float(
                self.order_diag_idle_obstacle_collisions
            ),
            "idle_agent_collision_count": float(
                self.order_diag_idle_agent_collisions
            ),
            "order_collision_rate": self._order_exec_rate(
                order_collisions,
                self.order_diag_order_steps,
            ),
            "charge_collision_rate": self._order_exec_rate(
                charge_collisions,
                self.order_diag_charge_steps,
            ),
            "idle_collision_rate": self._order_exec_rate(
                idle_collisions,
                self.order_diag_idle_steps,
            ),
        }

    def _order_execution_diagnostics_state(self):
        return {
            attr: float(getattr(self, attr, 0.0))
            for attr in ORDER_EXEC_DIAGNOSTIC_ATTRS
        }

    def _energy_required_for_distance(self, agent, distance):
        step_distance = max(float(agent.v_max) * float(self.time_step), eps)
        travel_steps = max(0.0, float(distance)) / step_distance
        return travel_steps * float(self.energy_decay_per_step)

    def _energy_required_ratio_for_distance(self, agent, distance):
        required_energy = self._energy_required_for_distance(agent, distance)
        return float(required_energy / (agent.initial_energy + eps))

    def _selected_order_for_agent(self, agent):
        slot_idx = agent.assigned_order_slot
        if slot_idx is None:
            slot_idx = agent.auction_order_slot
        order = self._slot_order(slot_idx)
        if order is None or order.status == DeliveryOrder.COMPLETED:
            return None
        return order

    def _finish_order_then_charge_distance_for_order(self, agent, order):
        if order is None:
            return 0.0

        current = agent.pos
        distance = 0.0
        if order.status == DeliveryOrder.PICKED or agent.carrying_order:
            distance += float(np.linalg.norm(current - order.dropoff_pos))
            current = order.dropoff_pos
        else:
            distance += float(np.linalg.norm(current - order.pickup_pos))
            distance += float(np.linalg.norm(order.pickup_pos - order.dropoff_pos))
            current = order.dropoff_pos

        distance += self._distance_to_nearest_charging_station_from_pos(current)
        return distance

    def _finish_order_then_charge_distance(self, agent):
        return self._finish_order_then_charge_distance_for_order(
            agent, self._selected_order_for_agent(agent)
        )

    def _energy_margin_for_order(self, agent, order):
        if order is None:
            return 0.0
        energy_ratio = float(agent.energy / (agent.initial_energy + eps))
        required_ratio = self._energy_required_ratio_for_distance(
            agent, self._finish_order_then_charge_distance_for_order(agent, order)
        )
        margin = energy_ratio - required_ratio - self.energy_margin_reserve_ratio
        return float(np.clip(margin, -1.0, 1.0))

    def _energy_margin_order_for_agent(self, agent):
        order = self._selected_order_for_agent(agent)
        return self._energy_margin_for_order(agent, order)

    def _order_slot_energy_feasible(self, agent, slot_idx):
        if not self.energy_shield_enabled:
            return True
        order = self._slot_order(slot_idx)
        return self._energy_margin_for_order(agent, order) >= 0.0

    def _selected_order_energy_feasible(self, agent):
        if not self.energy_shield_enabled:
            return True
        return self._energy_margin_order_for_agent(agent) >= 0.0

    def _has_available_order(self):
        return any(self._slot_order(slot_idx) is not None for slot_idx in self.available_order_slots)

    def _has_energy_feasible_available_order(self, agent):
        return any(
            self._slot_order(slot_idx) is not None
            and self._order_slot_energy_feasible(agent, slot_idx)
            for slot_idx in self.available_order_slots
        )

    def _has_selected_order_for_agent(self, agent):
        return 1.0 if self._selected_order_for_agent(agent) is not None else 0.0

    @staticmethod
    def _nonlinear_risk_from_overlap(overlap, scale):
        overlap = float(max(0.0, overlap))
        if overlap <= 0.0:
            return 0.0
        normalized = overlap / (float(scale) + eps)
        return float(normalized * normalized)

    def _sample_position(self, lower_margin=0.2):
        low = np.full(self.dim_actions, lower_margin, dtype=np.float32)
        high = self._space_scale() - lower_margin
        return np.random.uniform(low, high).astype(np.float32)

    def _is_valid_spawn(self, pos, others, radius):
        for obstacle in self.obstacles:
            if self._cylinder_distance(pos, obstacle) <= radius + obstacle.radius + 0.05:
                return False
        for other in others:
            if np.linalg.norm(pos - other.pos) <= radius + other.safe_radius + 0.08:
                return False
        return True

    def _is_valid_obstacle(self, obstacle, others):
        boundary_margin = obstacle.radius + self.safe_radius + 0.05
        if obstacle.pos[0] - boundary_margin < 0 or obstacle.pos[0] + boundary_margin > self.length:
            return False
        if obstacle.pos[1] - boundary_margin < 0 or obstacle.pos[1] + boundary_margin > self.width:
            return False

        for other in others:
            min_dist = obstacle.radius + other.radius + self.safe_radius + 0.08
            if np.linalg.norm(obstacle.pos - other.pos) <= min_dist:
                return False
        return True

    def _is_valid_goal(self, goal, existing_goals, agents):
        for obstacle in self.obstacles:
            if self._cylinder_distance(goal, obstacle) <= self.goal_tolerance + obstacle.radius + 0.05:
                return False
        for existing in existing_goals:
            if np.linalg.norm(goal - existing.pos) <= 0.25:
                return False
        for agent in agents:
            if np.linalg.norm(goal - agent.pos) <= 0.35:
                return False
        return True

    def _is_valid_delivery_point(self, point, existing_points, agents):
        for obstacle in self.obstacles:
            if self._cylinder_distance(point, obstacle) <= self.goal_tolerance + obstacle.radius + 0.05:
                return False
        for existing in existing_points:
            if np.linalg.norm(point - existing) <= 0.25:
                return False
        for agent in agents:
            if np.linalg.norm(point - agent.pos) <= 0.35:
                return False
        return True

    def _initialize_obstacles(self):
        obstacles = []
        for obstacle_idx in range(self.num_obstacle):
            for _ in range(self.sample_retry_limit):
                obstacle = Obstacle(self.length, self.width)
                if self._is_valid_obstacle(obstacle, obstacles):
                    obstacles.append(obstacle)
                    break
            else:
                raise RuntimeError(
                    f'Failed to place obstacle {obstacle_idx} after {self.sample_retry_limit} attempts.'
                )
        self.obstacles = obstacles

    def _initialize_agents(self):
        self.agents = []
        for agent_idx in range(self.num_agents):
            for _ in range(self.sample_retry_limit):
                pos = self._sample_position()
                candidate = UAVAgent(
                    number=agent_idx,
                    pos=pos,
                    v_max=self.v_max,
                    a_max=self.a_max,
                    num_lasers=self.num_lasers,
                    l_sensor=self.l_sensor,
                    safe_radius=self.safe_radius,
                    dim=self.dim_actions,
                    initial_energy=self.initial_energy,
                )
                if self._is_valid_spawn(pos, self.agents, self.safe_radius):
                    self.agents.append(candidate)
                    break
            else:
                raise RuntimeError(
                    f'Failed to place UAV {agent_idx} after {self.sample_retry_limit} attempts.'
                )

    def _initialize_orders(self):
        self.orders = []
        self.goals = []
        self.order_slots = [None for _ in range(self.max_active_orders)]
        self.available_order_slots = []
        self.active_order_ids = []
        self.next_order_id_to_activate = 0
        self.completed_order_count = 0

    def _slot_order(self, slot_idx):
        if slot_idx is None:
            return None
        slot_idx = int(slot_idx)
        if slot_idx < 0 or slot_idx >= self.max_active_orders:
            return None
        order_id = self.order_slots[slot_idx]
        if order_id is None or order_id >= len(self.orders):
            return None
        return self.orders[order_id]

    def _sync_order_lists(self):
        self.active_order_ids = []
        self.available_order_slots = []
        self.goals = []
        for slot_idx, order_id in enumerate(self.order_slots):
            if order_id is None or order_id >= len(self.orders):
                self.order_slots[slot_idx] = None
                continue
            order = self.orders[order_id]
            if order.status == DeliveryOrder.COMPLETED:
                self.order_slots[slot_idx] = None
                continue
            self.active_order_ids.append(order.order_id)
            if order.status == DeliveryOrder.ACTIVE and order.assigned_agent is None:
                self.available_order_slots.append(slot_idx)
                self.goals.append(GoalPoint(order.pickup_pos))

    def _active_delivery_points(self):
        points = []
        for order_id in self.order_slots:
            if order_id is None or order_id >= len(self.orders):
                continue
            order = self.orders[order_id]
            if order.status == DeliveryOrder.COMPLETED:
                continue
            points.extend([order.pickup_pos, order.dropoff_pos])
        return points

    def _sample_delivery_order(self, order_id):
        existing_points = self._active_delivery_points()

        pickup = None
        for _ in range(self.sample_retry_limit):
            candidate = self._sample_position(lower_margin=0.3)
            if self._is_valid_delivery_point(candidate, existing_points, self.agents):
                pickup = candidate
                break
        if pickup is None:
            raise RuntimeError(
                f'Failed to place pickup point for order {order_id} after {self.sample_retry_limit} attempts.'
            )

        dropoff = None
        for _ in range(self.sample_retry_limit):
            candidate = self._sample_position(lower_margin=0.3)
            if self._is_valid_delivery_point(
                candidate, existing_points + [pickup], self.agents
            ):
                dropoff = candidate
                break
        if dropoff is None:
            raise RuntimeError(
                f'Failed to place dropoff point for order {order_id} after {self.sample_retry_limit} attempts.'
            )

        return DeliveryOrder(order_id, pickup, dropoff)

    def _active_order_count(self):
        active_statuses = {
            DeliveryOrder.ACTIVE,
            DeliveryOrder.ASSIGNED,
            DeliveryOrder.PICKED,
        }
        return sum(
            1
            for order_id in self.order_slots
            if order_id is not None
            and order_id < len(self.orders)
            and self.orders[order_id].status in active_statuses
        )

    def _all_orders_completed(self):
        return self.completed_order_count >= self.total_orders

    def _sync_goals_from_orders(self):
        self._sync_order_lists()

    def _activate_orders(self):
        for slot_idx in range(self.max_active_orders):
            if self.next_order_id_to_activate >= self.total_orders:
                break
            if self.order_slots[slot_idx] is not None:
                continue
            order = self._sample_delivery_order(self.next_order_id_to_activate)
            order.status = DeliveryOrder.ACTIVE
            order.assigned_agent = None
            self.orders.append(order)
            self.order_slots[slot_idx] = order.order_id
            self.next_order_id_to_activate += 1
        self._sync_order_lists()

    def _available_orders(self):
        return [
            self.orders[order_id]
            for order_id in self.order_slots
            if order_id is not None
            and order_id < len(self.orders)
            and self.orders[order_id].status == DeliveryOrder.ACTIVE
            and self.orders[order_id].assigned_agent is None
        ]

    def _set_agent_idle(self, agent):
        self._release_charge_reservation(agent)
        agent.assigned_order_id = None
        agent.assigned_order_slot = None
        agent.auction_order_slot = None
        agent.carrying_order = False
        agent.vel[:] = 0.0
        agent.subgoal_test = False
        return self._set_agent_subgoal(agent, agent.pos.copy(), TASK_IDLE, project=False)

    def _set_agent_charging(self, agent):
        if not agent.has_energy():
            return False
        agent.auction_order_slot = None
        self._reserve_charge_slot(agent)
        return self._set_agent_subgoal(
            agent,
            self._charge_slot_target(agent, None),
            TASK_CHARGE,
        )

    def _assign_order_slot_to_agent(self, agent, slot_idx):
        self._release_charge_reservation(agent)
        slot_idx = int(slot_idx)
        order = self._slot_order(slot_idx)
        if order is None or not agent.has_energy():
            return False

        owns_slot = (
            agent.assigned_order_slot == slot_idx
            and agent.assigned_order_id == order.order_id
            and order.assigned_agent == agent.number
        )
        if owns_slot:
            if order.status == DeliveryOrder.PICKED or agent.carrying_order:
                agent.carrying_order = True
                target = order.dropoff_pos
            else:
                agent.carrying_order = False
                target = order.pickup_pos
            return self._set_agent_subgoal(agent, target, TASK_ORDER)

        if agent.assigned_order_id is not None:
            return False
        if order.status != DeliveryOrder.ACTIVE or order.assigned_agent is not None:
            return False

        order.status = DeliveryOrder.ASSIGNED
        order.assigned_agent = agent.number
        agent.assigned_order_id = order.order_id
        agent.assigned_order_slot = slot_idx
        agent.carrying_order = False
        agent.auction_order_slot = None
        self._set_agent_subgoal(agent, order.pickup_pos, TASK_ORDER)
        self._sync_order_lists()
        return True

    def _commit_auction_order_subgoal(self, agent):
        slot_idx = agent.assigned_order_slot
        if slot_idx is None:
            slot_idx = agent.auction_order_slot
        if slot_idx is None:
            return False
        return self._assign_order_slot_to_agent(agent, int(slot_idx))

    def _select_greedy_order_slot(self, agent):
        best_slot = None
        best_cost = float("inf")
        for slot_idx in self.available_order_slots:
            order = self._slot_order(slot_idx)
            if order is None:
                continue
            if not self._order_slot_energy_feasible(agent, slot_idx):
                continue
            travel = float(np.linalg.norm(agent.pos - order.pickup_pos))
            travel += float(np.linalg.norm(order.pickup_pos - order.dropoff_pos))
            if travel < best_cost:
                best_cost = travel
                best_slot = int(slot_idx)
        return best_slot

    def _maybe_set_greedy_order_slot(self, agent):
        if self.auction_enabled:
            return
        if agent.assigned_order_slot is not None or agent.auction_order_slot is not None:
            return
        slot_idx = self._select_greedy_order_slot(agent)
        if slot_idx is not None:
            agent.auction_order_slot = slot_idx

    def run_order_auction(self, record=False):
        self._activate_orders()
        for agent in self.agents:
            if (
                agent.current_task_type == TASK_CHARGE
                and self._charge_option_complete(agent)
                and agent.assigned_order_slot is None
            ):
                self._set_agent_idle(agent)
            if agent.assigned_order_slot is None:
                agent.auction_order_slot = None

        if not self.auction_enabled:
            if record:
                self.high_diag_auction_calls += 1.0
                self.high_diag_auction_candidates += 0.0
                self.high_diag_auction_orders += float(len(self.available_order_slots))
            return {}

        candidate_agents = [
            agent
            for agent in self.agents
            if agent.has_energy()
            and agent.assigned_order_slot is None
            and not self._agent_charge_locked(agent)
        ]
        order_slots = list(self.available_order_slots)
        if record:
            self.high_diag_auction_calls += 1.0
            self.high_diag_auction_candidates += float(len(candidate_agents))
            self.high_diag_auction_orders += float(len(order_slots))
        if not candidate_agents or not order_slots:
            return {}

        cost_matrix = np.zeros((len(candidate_agents), len(order_slots)), dtype=np.float32)
        for agent_row, agent in enumerate(candidate_agents):
            for order_col, slot_idx in enumerate(order_slots):
                order = self._slot_order(slot_idx)
                if order is None:
                    cost_matrix[agent_row, order_col] = 1e6
                    continue
                if not self._order_slot_energy_feasible(agent, slot_idx):
                    cost_matrix[agent_row, order_col] = 1e6
                    continue
                travel = np.linalg.norm(agent.pos - order.pickup_pos)
                travel += np.linalg.norm(order.pickup_pos - order.dropoff_pos)
                energy_ratio = agent.energy / (agent.initial_energy + eps)
                energy_penalty = max(0.0, 0.35 - energy_ratio) * max(self.length, self.width)
                cost_matrix[agent_row, order_col] = float(travel + energy_penalty)

        assignments = auction_assign_min_cost(cost_matrix)
        result = {}
        for agent_row, order_col in assignments.items():
            if cost_matrix[int(agent_row), int(order_col)] >= 1e5:
                continue
            agent = candidate_agents[int(agent_row)]
            slot_idx = int(order_slots[int(order_col)])
            agent.auction_order_slot = slot_idx
            result[agent.number] = slot_idx
        if record:
            self.high_diag_auction_assignments += float(len(result))
        return result

    def prepare_high_level_decision(self):
        return self.run_order_auction(record=False)

    def _assign_orders(self):
        self._activate_orders()

    def _remove_active_order(self, order_id):
        for slot_idx, active_id in enumerate(self.order_slots):
            if active_id == order_id:
                self.order_slots[slot_idx] = None
        self._sync_order_lists()

    def _advance_order_if_reached(self, agent, current_dist):
        if (
            agent.current_task_type != TASK_ORDER
            or agent.assigned_order_id is None
        ):
            return 0.0

        order = self.orders[agent.assigned_order_id]
        if order.status == DeliveryOrder.ASSIGNED:
            if np.linalg.norm(agent.pos - order.pickup_pos) > self.goal_tolerance:
                return 0.0
            order.status = DeliveryOrder.PICKED
            agent.carrying_order = True
            self._set_agent_subgoal(agent, agent.pos.copy(), TASK_IDLE, project=False)
            agent.assigned_order_id = order.order_id
            agent.assigned_order_slot = self.order_slots.index(order.order_id)
            agent.carrying_order = True
            agent.current_task_type = TASK_ORDER
            agent.reached = True
            return self.pickup_reward

        if order.status == DeliveryOrder.PICKED:
            if np.linalg.norm(agent.pos - order.dropoff_pos) > self.goal_tolerance:
                return 0.0
            order.status = DeliveryOrder.COMPLETED
            order.assigned_agent = None
            self._remove_active_order(order.order_id)
            self.completed_order_count += 1
            agent.completed_orders += 1
            self._set_agent_idle(agent)
            return self.delivery_reward

        return 0.0

    def _agent_has_motion_task(self, agent):
        if agent.current_task_type == TASK_CHARGE:
            return True
        return (
            agent.current_task_type == TASK_ORDER
            and agent.assigned_order_id is not None
        )

    def _agent_high_level_features(self, agent):
        scale = self._space_scale() + eps
        energy = np.array(
            [agent.energy / (agent.initial_energy + eps)],
            dtype=np.float32,
        )
        risk = np.array(
            [
                min(
                    1.0,
                    float(self.safe_value[agent.number]) / (agent.safe_radius + eps),
                )
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                agent.pos / scale,
                agent.vel / (agent.v_max + eps),
                energy,
                risk,
            ]
        ).astype(np.float32)

    def _order_slot_features(self):
        scale = self._space_scale() + eps
        features = []
        for slot_idx in range(self.max_active_orders):
            order = self._slot_order(slot_idx)
            if order is not None and order.status != DeliveryOrder.COMPLETED:
                features.extend(
                    [
                        order.pickup_pos / scale,
                        order.dropoff_pos / scale,
                    ]
                )
            else:
                features.extend(
                    [
                        np.zeros(self.dim_actions, dtype=np.float32),
                        np.zeros(self.dim_actions, dtype=np.float32),
                    ]
                )
        return np.concatenate(features).astype(np.float32)

    def _charging_station_features(self):
        scale = self._space_scale() + eps
        station_center = np.mean(self.charging_station_positions, axis=0)
        return (station_center / scale).astype(np.float32)

    def _order_target_for_agent(self, agent):
        slot_idx = agent.assigned_order_slot
        if slot_idx is None:
            slot_idx = agent.auction_order_slot
        order = self._slot_order(slot_idx)
        if order is None:
            return None
        if order.status == DeliveryOrder.PICKED or agent.carrying_order:
            return order.dropoff_pos.copy()
        return order.pickup_pos.copy()

    def _target_delta_features(self, agent, target):
        if target is None:
            return np.zeros(self.dim_actions + 1, dtype=np.float32)
        scale = self._space_scale() + eps
        delta = np.asarray(target, dtype=np.float32)[: self.dim_actions] - agent.pos
        distance = np.linalg.norm(delta) / (np.linalg.norm(scale) + eps)
        return np.concatenate(
            [delta / scale, np.array([distance], dtype=np.float32)]
        ).astype(np.float32)

    def _other_agent_features(self, observer, other):
        scale = self._space_scale() + eps
        rel_pos = (other.pos - observer.pos) / scale
        rel_vel = (other.vel - observer.vel) / (observer.v_max + eps)
        distance = np.array(
            [np.linalg.norm(other.pos - observer.pos) / (np.linalg.norm(scale) + eps)],
            dtype=np.float32,
        )
        energy = np.array(
            [other.energy / (other.initial_energy + eps)],
            dtype=np.float32,
        )
        risk = np.array(
            [
                min(
                    1.0,
                    float(self.safe_value[other.number]) / (other.safe_radius + eps),
                )
            ],
            dtype=np.float32,
        )
        return np.concatenate([rel_pos, rel_vel, energy, distance, risk]).astype(
            np.float32
        )

    def _relative_order_slot_features(self, agent):
        scale = self._space_scale() + eps
        features = []
        for slot_idx in range(self.max_active_orders):
            order = self._slot_order(slot_idx)
            if order is not None and order.status != DeliveryOrder.COMPLETED:
                pickup_delta = order.pickup_pos - agent.pos
                dropoff_delta = order.dropoff_pos - agent.pos
                features.extend(
                    [
                        pickup_delta / scale,
                        np.array(
                            [
                                np.linalg.norm(pickup_delta)
                                / (np.linalg.norm(scale) + eps)
                            ],
                            dtype=np.float32,
                        ),
                        dropoff_delta / scale,
                        np.array(
                            [
                                np.linalg.norm(dropoff_delta)
                                / (np.linalg.norm(scale) + eps)
                            ],
                            dtype=np.float32,
                        ),
                    ]
                )
            else:
                features.extend(
                    [
                        np.zeros(self.dim_actions, dtype=np.float32),
                        np.zeros(1, dtype=np.float32),
                        np.zeros(self.dim_actions, dtype=np.float32),
                        np.zeros(1, dtype=np.float32),
                    ]
                )
        return np.concatenate(features).astype(np.float32)

    def _high_level_agent_obs(self, agent):
        scale = self._space_scale() + eps
        other_features = [
            self._other_agent_features(agent, other)
            for other in self.agents
            if other is not agent
        ]
        if not other_features:
            other_features = [np.zeros(2 * self.dim_actions + 3, dtype=np.float32)]
        charging_delta = self._target_delta_features(
            agent,
            self._nearest_charging_station_pos(agent.pos),
        )
        lasers = np.asarray(agent.lasers, dtype=np.float32) / (agent.l_sensor + eps)
        return np.concatenate(
            [
                self._agent_high_level_features(agent),
                self._target_delta_features(agent, self._order_target_for_agent(agent)),
                self._target_delta_features(agent, self._assigned_target_for_agent(agent)),
                charging_delta,
                np.concatenate(other_features).astype(np.float32),
                self._relative_order_slot_features(agent),
                lasers,
            ]
        ).astype(np.float32)

    def _assigned_target_for_agent(self, agent):
        if agent.assigned_order_slot is None:
            return None
        order = self._slot_order(agent.assigned_order_slot)
        if order is None:
            return None
        if order.status == DeliveryOrder.PICKED or agent.carrying_order:
            return order.dropoff_pos.copy()
        return order.pickup_pos.copy()

    def get_high_level_state(self):
        agent_features = [self._agent_high_level_features(agent) for agent in self.agents]
        auction_targets = [
            self._target_delta_features(agent, self._order_target_for_agent(agent))
            for agent in self.agents
        ]
        assigned_targets = [
            self._target_delta_features(agent, self._assigned_target_for_agent(agent))
            for agent in self.agents
        ]
        return np.concatenate(
            [
                np.concatenate(agent_features).astype(np.float32),
                np.concatenate(auction_targets).astype(np.float32),
                np.concatenate(assigned_targets).astype(np.float32),
                self._order_slot_features(),
                self._charging_station_features(),
            ]
        ).astype(np.float32)

    def get_high_level_obs(self):
        observations = []
        for agent in self.agents:
            observations.append(self._high_level_agent_obs(agent))
        return np.stack(observations, axis=0)

    def get_available_order_mask(self):
        mask = np.zeros(self.max_active_orders, dtype=np.float32)
        for slot_idx in self.available_order_slots:
            if 0 <= int(slot_idx) < self.max_active_orders:
                mask[int(slot_idx)] = 1.0
        return mask

    def get_high_level_avail_agent_actions(self, agent_id):
        del agent_id
        return np.ones(self.high_level_n_actions, dtype=np.float32)

    def get_high_level_avail_actions(self):
        return np.stack(
            [
                self.get_high_level_avail_agent_actions(agent_id)
                for agent_id in range(self.num_agents)
            ],
            axis=0,
        )

    def get_high_level_energy_margins(self):
        return np.asarray(
            [[self._energy_margin_order_for_agent(agent)] for agent in self.agents],
            dtype=np.float32,
        )

    def get_high_level_energy_order_masks(self):
        return np.asarray(
            [[self._has_selected_order_for_agent(agent)] for agent in self.agents],
            dtype=np.float32,
        )

    def get_high_level_mode_training_mask(self):
        return np.asarray(self._last_high_mode_train_mask, dtype=np.float32).copy()

    def get_current_high_level_actions(self):
        actions = []
        for agent in self.agents:
            mode_value = (
                HIGH_MODE_CHARGE
                if agent.current_task_type == TASK_CHARGE
                else HIGH_MODE_ORDER
            )
            target = getattr(agent, "task_target", None)
            if target is None:
                actions.append(
                    np.array([mode_value, 0.0], dtype=np.float32)
                )
                continue
            target = np.asarray(target, dtype=np.float32)[: self.dim_actions]
            to_target = target - agent.pos
            target_dist = float(np.linalg.norm(to_target))
            max_dist = self._max_reachable_subgoal_distance(agent)
            line_length = min(target_dist, max_dist)
            if line_length <= eps:
                actions.append(
                    np.array([mode_value, 0.0], dtype=np.float32)
                )
                continue
            direction = to_target / (target_dist + eps)
            progress = float(np.dot(agent.subgoal - agent.pos, direction))
            progress = float(np.clip(progress, 0.0, line_length))
            min_progress = min(line_length, self.min_subgoal_progress)
            if line_length > min_progress + eps:
                fraction = (progress - min_progress) / (line_length - min_progress)
            else:
                fraction = 1.0
            progress_scalar = 2.0 * float(np.clip(fraction, 0.0, 1.0)) - 1.0
            if agent.current_task_type not in (TASK_CHARGE, TASK_ORDER):
                progress_scalar = 0.0
            actions.append(
                np.array([mode_value, progress_scalar], dtype=np.float32)
            )
        return np.stack(actions, axis=0)

    def apply_high_level_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_agents, self.high_level_n_actions):
            actions = actions.reshape(self.num_agents, self.high_level_n_actions)

        self.run_order_auction(record=True)
        self._refresh_charge_reservations()
        applied = np.zeros(self.num_agents, dtype=np.float32)
        mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)
        action_contexts = []
        charge_intent_agent_ids = []
        for agent_idx, action in enumerate(actions):
            agent = self.agents[agent_idx]
            if not agent.has_energy():
                action_contexts.append(None)
                continue

            mode_id, progress_scalar = self._parse_high_level_action(action)
            task_target = self._order_target_for_agent(agent)
            self._maybe_set_greedy_order_slot(agent)
            if task_target is None:
                task_target = self._order_target_for_agent(agent)
            order_locked = self._agent_order_locked(agent)
            charge_locked = self._agent_charge_locked(agent)
            energy_ratio = agent.energy / (agent.initial_energy + eps)
            if (
                self.fixed_charge_threshold_enabled
                and not order_locked
                and not charge_locked
            ):
                if energy_ratio <= self.fixed_charge_threshold:
                    mode_id = HIGH_MODE_CHARGE
                elif (
                    agent.current_task_type == TASK_CHARGE
                    and energy_ratio < self.fixed_charge_release_threshold
                ):
                    mode_id = HIGH_MODE_CHARGE
                else:
                    mode_id = HIGH_MODE_ORDER
            energy_blocked_order = (
                self.energy_shield_enabled
                and mode_id == HIGH_MODE_ORDER
                and not order_locked
                and not charge_locked
                and (
                    (
                        task_target is not None
                        and not self._selected_order_energy_feasible(agent)
                    )
                    or (
                        task_target is None
                        and self._has_available_order()
                        and not self._has_energy_feasible_available_order(agent)
                    )
                )
            )
            should_charge_fallback = (
                task_target is None
                and (
                    energy_ratio <= self.charge_energy_threshold
                    or (
                        agent.current_task_type == TASK_CHARGE
                        and energy_ratio < self.charge_release_threshold
                    )
                )
            )
            will_charge = (
                not order_locked
                and (
                    charge_locked
                    or energy_blocked_order
                    or mode_id == HIGH_MODE_CHARGE
                    or should_charge_fallback
                )
            )
            if will_charge:
                charge_intent_agent_ids.append(agent_idx)
            action_contexts.append(
                {
                    "mode_id": mode_id,
                    "progress_scalar": progress_scalar,
                    "task_target": task_target,
                    "order_locked": order_locked,
                    "charge_locked": charge_locked,
                    "energy_blocked_order": energy_blocked_order,
                    "should_charge_fallback": should_charge_fallback,
                }
            )
        for agent_idx in sorted(charge_intent_agent_ids):
            self._reserve_charge_slot(self.agents[agent_idx])
        charge_rank_by_agent = {
            agent_idx: rank for rank, agent_idx in enumerate(sorted(charge_intent_agent_ids))
        }

        for agent_idx, action in enumerate(actions):
            agent = self.agents[agent_idx]
            if not agent.has_energy():
                continue

            context = action_contexts[agent_idx]
            mode_id = context["mode_id"]
            progress_scalar = context["progress_scalar"]
            self.high_diag_decision_count += 1.0
            if mode_id == HIGH_MODE_CHARGE:
                self.high_diag_sampled_charge += 1.0
            else:
                self.high_diag_sampled_order += 1.0

            task_target = context["task_target"]
            task_type = TASK_ORDER
            order_locked = context["order_locked"]
            charge_locked = context["charge_locked"]
            energy_blocked_order = context["energy_blocked_order"]
            should_charge_fallback = context["should_charge_fallback"]

            if order_locked:
                self.high_diag_order_locked += 1.0
                task_target = self._order_target_for_agent(agent)
                task_type = TASK_ORDER
                applied[agent_idx] = 1.0
            elif charge_locked:
                self.high_diag_charge_locked += 1.0
                agent.auction_order_slot = None
                task_target = self._charge_slot_target(
                    agent, charge_rank_by_agent.get(agent_idx)
                )
                task_type = TASK_CHARGE
                applied[agent_idx] = 1.0
            elif energy_blocked_order:
                self.high_diag_charge_fallbacks += 1.0
                agent.auction_order_slot = None
                task_target = self._charge_slot_target(
                    agent, charge_rank_by_agent.get(agent_idx)
                )
                task_type = TASK_CHARGE
                applied[agent_idx] = 1.0
            elif mode_id == HIGH_MODE_CHARGE:
                agent.auction_order_slot = None
                task_target = self._charge_slot_target(
                    agent, charge_rank_by_agent.get(agent_idx)
                )
                task_type = TASK_CHARGE
                applied[agent_idx] = 1.0
                mode_train_mask[agent_idx, 0] = 1.0
            elif task_target is not None:
                if agent.assigned_order_slot is None:
                    self.high_diag_order_commit_attempts += 1.0
                    applied[agent_idx] = float(self._commit_auction_order_subgoal(agent))
                    self.high_diag_order_commit_successes += float(
                        applied[agent_idx] > 0.0
                    )
                    task_target = self._order_target_for_agent(agent)
                    mode_train_mask[agent_idx, 0] = float(applied[agent_idx] > 0.0)
                else:
                    applied[agent_idx] = 1.0
                    mode_train_mask[agent_idx, 0] = 1.0
            else:
                if should_charge_fallback:
                    self.high_diag_charge_fallbacks += 1.0
                    agent.auction_order_slot = None
                    task_target = self._charge_slot_target(
                        agent, charge_rank_by_agent.get(agent_idx)
                    )
                    task_type = TASK_CHARGE
                    applied[agent_idx] = 1.0
                else:
                    task_target = agent.pos.copy()
                    task_type = TASK_IDLE
                    applied[agent_idx] = 1.0

            if task_type == TASK_CHARGE:
                self.high_diag_executed_charge += 1.0
                progress_scalar = 1.0
            elif task_type == TASK_ORDER:
                self.high_diag_executed_order += 1.0
                if self.order_progress_override is not None:
                    progress_scalar = self.order_progress_override
            else:
                self.high_diag_executed_idle += 1.0
            self.high_diag_mode_train_mask_sum += float(
                mode_train_mask[agent_idx, 0]
            )
            self.high_diag_mode_train_mask_count += 1.0

            self._set_agent_subgoal_on_target_line(
                agent,
                np.array([progress_scalar], dtype=np.float32),
                task_type=task_type,
                task_target=task_target,
            )

        self._sync_order_lists()
        self._last_high_mode_train_mask = mode_train_mask
        if hasattr(self, "get_current_high_level_actions"):
            return self.get_current_high_level_actions()
        return applied

    def get_active_agent_mask(self):
        return np.asarray(
            [1.0 if agent.has_energy() else 0.0 for agent in self.agents],
            dtype=np.float32,
        )

    def _init_charging_station_positions(self, charging_station_pos):
        if charging_station_pos is not None:
            positions = np.asarray(charging_station_pos, dtype=np.float32)
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
            positions = positions[:, : self.dim_actions]
            if len(positions) >= self.charging_station_count:
                return positions[: self.charging_station_count].copy()
            extra = self._default_charging_station_positions(
                self.charging_station_count - len(positions)
            )
            return np.concatenate([positions, extra], axis=0).astype(np.float32)
        return self._default_charging_station_positions(self.charging_station_count)

    def _default_charging_station_positions(self, count):
        count = int(max(1, count))
        center = np.array([self.length * 0.5, self.width * 0.5], dtype=np.float32)
        radius = min(self.length, self.width) * 0.22
        positions = []
        for idx in range(count):
            angle = 2.0 * np.pi * float(idx) / float(count)
            pos2 = center + radius * np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32,
            )
            pos2[0] = np.clip(pos2[0], self.charging_radius, self.length - self.charging_radius)
            pos2[1] = np.clip(pos2[1], self.charging_radius, self.width - self.charging_radius)
            if self.dim_actions == 3:
                positions.append(np.array([pos2[0], pos2[1], self.height * 0.5], dtype=np.float32))
            else:
                positions.append(pos2.astype(np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def _nearest_charging_station_index(self, pos):
        distances = np.linalg.norm(
            self.charging_station_positions - np.asarray(pos, dtype=np.float32)[: self.dim_actions],
            axis=1,
        )
        return int(np.argmin(distances))

    def _nearest_charging_station_pos(self, pos):
        return self.charging_station_positions[
            self._nearest_charging_station_index(pos)
        ].copy()

    def _distance_to_nearest_charging_station_from_pos(self, pos):
        return float(
            np.min(
                np.linalg.norm(
                    self.charging_station_positions - np.asarray(pos, dtype=np.float32)[: self.dim_actions],
                    axis=1,
                )
            )
        )

    def _distance_to_charging_station(self, agent):
        return self._distance_to_nearest_charging_station_from_pos(agent.pos)

    def _empty_charge_reservations(self):
        return {
            station_idx: [None for _ in range(max(1, int(self.charging_capacity)))]
            for station_idx in range(self.charging_station_count)
        }

    def _release_charge_reservation(self, agent):
        station_idx = getattr(agent, "charge_station_idx", None)
        slot_idx = getattr(agent, "charge_slot_idx", None)
        if station_idx is not None and slot_idx is not None:
            station_idx = int(station_idx)
            slot_idx = int(slot_idx)
            slots = self.charging_slot_reservations.get(station_idx)
            if slots is not None and 0 <= slot_idx < len(slots):
                if slots[slot_idx] == agent.number:
                    slots[slot_idx] = None
        agent.charge_station_idx = None
        agent.charge_slot_idx = None

    def _agent_has_charge_reservation(self, agent):
        station_idx = getattr(agent, "charge_station_idx", None)
        slot_idx = getattr(agent, "charge_slot_idx", None)
        if station_idx is None or slot_idx is None:
            return False
        station_idx = int(station_idx)
        slot_idx = int(slot_idx)
        slots = self.charging_slot_reservations.get(station_idx)
        return bool(
            slots is not None
            and 0 <= slot_idx < len(slots)
            and slots[slot_idx] == agent.number
        )

    def _refresh_charge_reservations(self):
        active_charge_ids = {
            agent.number
            for agent in self.agents
            if agent.has_energy()
            and agent.current_task_type == TASK_CHARGE
            and not self._charge_option_complete(agent)
        }
        for station_idx, slots in self.charging_slot_reservations.items():
            for slot_idx, agent_id in enumerate(slots):
                if agent_id is None or agent_id in active_charge_ids:
                    continue
                slots[slot_idx] = None
        for agent in self.agents:
            if agent.number in active_charge_ids:
                continue
            agent.charge_station_idx = None
            agent.charge_slot_idx = None

    def _reserve_charge_slot(self, agent):
        if not agent.has_energy():
            self._release_charge_reservation(agent)
            return None
        if self._agent_has_charge_reservation(agent):
            return int(agent.charge_station_idx), int(agent.charge_slot_idx)

        self._release_charge_reservation(agent)
        free_slots = []
        for station_idx, station_pos in enumerate(self.charging_station_positions):
            slots = self.charging_slot_reservations.get(station_idx, [])
            for slot_idx, holder in enumerate(slots):
                if holder is None:
                    free_slots.append(
                        (
                            float(np.linalg.norm(agent.pos - station_pos)),
                            station_idx,
                            slot_idx,
                        )
                    )
        if not free_slots:
            return None
        _, station_idx, slot_idx = min(free_slots, key=lambda item: item)
        self.charging_slot_reservations[station_idx][slot_idx] = agent.number
        agent.charge_station_idx = int(station_idx)
        agent.charge_slot_idx = int(slot_idx)
        return int(station_idx), int(slot_idx)

    def _charge_dock_target(self, station_idx, slot_idx):
        capacity = max(1, int(self.charging_capacity))
        slot_count = capacity
        angle = 2.0 * np.pi * float(slot_idx) / float(max(1, slot_count))
        offset = np.zeros(self.dim_actions, dtype=np.float32)
        offset[0] = float(self.charge_dock_radius) * np.cos(angle)
        if self.dim_actions >= 2:
            offset[1] = float(self.charge_dock_radius) * np.sin(angle)
        target = self.charging_station_positions[int(station_idx)] + offset
        return self._clip_position_to_bounds(target)

    def _charge_wait_target(self, agent, charge_rank):
        if not self.charge_queue_enabled or charge_rank is None:
            return self._nearest_charging_station_pos(agent.pos)

        charge_rank = int(max(0, charge_rank))
        capacity = max(1, int(self.charging_capacity))
        dock_slots = max(1, self.charging_station_count * capacity)
        queue_rank = max(0, charge_rank - dock_slots)
        radius = float(self.charge_queue_radius)
        station_idx = queue_rank % self.charging_station_count
        slot_idx = queue_rank // self.charging_station_count
        slot_count = max(
            1,
            (self.num_agents + self.charging_station_count - 1)
            // self.charging_station_count,
        )

        angle = 2.0 * np.pi * float(slot_idx) / float(max(1, slot_count))
        offset = np.zeros(self.dim_actions, dtype=np.float32)
        offset[0] = radius * np.cos(angle)
        if self.dim_actions >= 2:
            offset[1] = radius * np.sin(angle)
        target = self.charging_station_positions[int(station_idx)] + offset
        return self._clip_position_to_bounds(target)

    def _charge_slot_target(self, agent, charge_rank):
        reservation = self._reserve_charge_slot(agent)
        if reservation is not None:
            station_idx, slot_idx = reservation
            return self._charge_dock_target(station_idx, slot_idx)
        return self._charge_wait_target(agent, charge_rank)

    def _consume_step_energy(self, powered_mask):
        for is_powered, agent in zip(powered_mask, self.agents):
            if is_powered:
                agent.consume_energy(self.energy_decay_per_step)

    def _charge_agents_at_station(self):
        self._refresh_charge_reservations()
        selected = []
        station_assignments = {
            station_idx: [] for station_idx in range(self.charging_station_count)
        }
        for station_idx, station_pos in enumerate(self.charging_station_positions):
            station_selected = []
            slots = self.charging_slot_reservations.get(station_idx, [])
            for slot_idx, agent_id in enumerate(slots):
                if agent_id is None:
                    continue
                agent = self.agents[int(agent_id)]
                if (
                    agent.current_task_type == TASK_CHARGE
                    and agent.reached
                    and self._agent_has_charge_reservation(agent)
                    and int(agent.charge_station_idx) == int(station_idx)
                    and int(agent.charge_slot_idx) == int(slot_idx)
                    and np.linalg.norm(agent.pos - station_pos) <= self.goal_tolerance
                    and agent.energy < agent.initial_energy
                ):
                    station_selected.append(agent)
            station_assignments[station_idx] = [agent.number for agent in station_selected]
            selected.extend(station_selected)
        self.charging_station_agent_ids = station_assignments
        self.charging_agent_ids = sorted({agent.number for agent in selected})
        for agent in selected:
            agent.charge_energy(self.charging_rate)

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(1, 100000)
        random.seed(seed)
        np.random.seed(seed)

        last_error = None
        for _ in range(self.reset_retry_limit):
            self.current_step = 0
            self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
            self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
            self.collision_count = 0.0
            self.obstacle_collision_count = 0.0
            self.agent_collision_count = 0.0
            self.agent_paths = [[] for _ in range(self.num_agents)]
            self.obstacles = []
            self.agents = []
            self.goals = []
            self.orders = []
            self.order_slots = [None for _ in range(self.max_active_orders)]
            self.available_order_slots = []
            self.active_order_ids = []
            self.next_order_id_to_activate = 0
            self.completed_order_count = 0
            self._last_step_order_status_before = [
                None for _ in range(self.num_agents)
            ]
            self.charging_agent_ids = []
            self.charging_station_agent_ids = {
                station_idx: [] for station_idx in range(self.charging_station_count)
            }
            self.charging_slot_reservations = self._empty_charge_reservations()
            self._last_high_mode_train_mask = np.ones(
                (self.num_agents, 1), dtype=np.float32
            )
            self._reset_high_level_diagnostics()
            self._reset_order_execution_diagnostics()

            try:
                self._initialize_obstacles()
                self._initialize_agents()
                self._initialize_orders()
                self._assign_orders()
                self.update_lasers()
                return self.get_obs()
            except RuntimeError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            'UAVEnv reset failed to sample a valid scenario '
            f'after {self.reset_retry_limit} retries. Last error: {last_error}'
        )

    def _apply_boundary_constraints(self, agent):
        collided = False
        boundary_risk = 0.0
        boundary_reward_risk = 0.0
        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        for dim, boundary in enumerate(boundaries):
            lower_clearance = float(agent.prev_pos[dim])
            upper_clearance = float(boundary - agent.prev_pos[dim])
            warning_clearance = agent.safe_radius + self.risk_warning_margin
            lower_overlap = max(0.0, warning_clearance - lower_clearance)
            upper_overlap = max(0.0, warning_clearance - upper_clearance)
            boundary_risk += self._nonlinear_risk_from_overlap(
                lower_overlap, warning_clearance
            )
            boundary_risk += self._nonlinear_risk_from_overlap(
                upper_overlap, warning_clearance
            )
            boundary_reward_risk += self._nonlinear_risk_from_overlap(
                max(0.0, agent.safe_radius - lower_clearance), agent.safe_radius
            )
            boundary_reward_risk += self._nonlinear_risk_from_overlap(
                max(0.0, agent.safe_radius - upper_clearance), agent.safe_radius
            )
            if agent.prev_pos[dim] - agent.safe_radius < 0:
                agent.prev_pos[dim] = agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
            elif agent.prev_pos[dim] + agent.safe_radius > boundary:
                agent.prev_pos[dim] = boundary - agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
        return collided, boundary_risk, boundary_reward_risk

    def _resolve_obstacle_collisions(self, agent):
        collided = False
        safe_penalty = 0.0
        reward_penalty = 0.0
        for obstacle in self.obstacles:
            delta_xy = agent.prev_pos[:2] - obstacle.pos
            dist_xy = np.linalg.norm(delta_xy)
            collision_min_dist = obstacle.radius + agent.safe_radius
            warning_min_dist = collision_min_dist + self.risk_warning_margin
            overlap = max(0.0, warning_min_dist - dist_xy)
            safe_penalty = max(
                safe_penalty,
                self._nonlinear_risk_from_overlap(overlap, warning_min_dist),
            )
            reward_penalty = max(
                reward_penalty,
                self._nonlinear_risk_from_overlap(
                    max(0.0, collision_min_dist - dist_xy), collision_min_dist
                ),
            )
            if dist_xy < collision_min_dist:
                collided = True
                if dist_xy < eps:
                    delta_xy = np.array([1.0, 0.0], dtype=np.float32)
                    dist_xy = 1.0
                direction_xy = delta_xy / dist_xy
                corrected_xy = obstacle.pos + direction_xy * collision_min_dist
                agent.prev_pos[:2] = corrected_xy
                agent.vel[:2] = 0.0
        return collided, safe_penalty, reward_penalty

    def _resolve_agent_collisions(self):
        collided = [False] * self.num_agents
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                delta = self.agents[i].prev_pos - self.agents[j].prev_pos
                dist = np.linalg.norm(delta)
                collision_min_dist = (
                    self.agents[i].safe_radius + self.agents[j].safe_radius
                )
                warning_min_dist = collision_min_dist + self.risk_warning_margin
                overlap = max(0.0, warning_min_dist - dist)
                pair_risk = self._nonlinear_risk_from_overlap(
                    overlap, warning_min_dist
                )
                pair_reward_risk = self._nonlinear_risk_from_overlap(
                    overlap, warning_min_dist
                )
                self.safe_value[i] += pair_risk
                self.safe_value[j] += pair_risk
                self.reward_safe_value[i] += pair_reward_risk
                self.reward_safe_value[j] += pair_reward_risk
                if dist < collision_min_dist:
                    collided[i] = True
                    collided[j] = True
                    if dist < eps:
                        delta = np.zeros(self.dim_actions, dtype=np.float32)
                        delta[0] = 1.0
                        dist = 1.0
                    direction = delta / dist
                    overlap = collision_min_dist - dist
                    i_powered = self.agents[i].has_energy()
                    j_powered = self.agents[j].has_energy()
                    if i_powered and j_powered:
                        self.agents[i].prev_pos += direction * (overlap / 2.0 + eps)
                        self.agents[j].prev_pos -= direction * (overlap / 2.0 + eps)
                    elif i_powered:
                        self.agents[i].prev_pos += direction * (overlap + eps)
                    elif j_powered:
                        self.agents[j].prev_pos -= direction * (overlap + eps)
                    self.agents[i].vel[:] = 0.0
                    self.agents[j].vel[:] = 0.0
        return collided

    def _predict_agent_kinematics(self, agent, action):
        accel = np.asarray(action, dtype=np.float32).copy()
        norm = np.linalg.norm(accel)
        if norm > agent.a_max:
            accel = accel / (norm + eps) * agent.a_max

        pred_vel = agent.vel + accel * self.time_step
        speed = np.linalg.norm(pred_vel)
        if speed > agent.v_max:
            pred_vel = pred_vel / (speed + eps) * agent.v_max

        pred_pos = agent.pos + agent.vel * self.time_step + 0.5 * accel * (
            self.time_step ** 2
        )
        return pred_pos.astype(np.float32), pred_vel.astype(np.float32)

    def _evaluate_joint_risk_batch_exact(self, actions_batch):
        actions_batch = np.asarray(actions_batch, dtype=np.float32)
        if actions_batch.ndim != 3:
            raise ValueError(
                "actions_batch must have shape (batch, n_agents, dim_actions)"
            )
        if actions_batch.shape[1] != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")
        if actions_batch.shape[2] != self.dim_actions:
            raise ValueError("Action dimension does not match UAV action dimension.")

        batch_size = actions_batch.shape[0]
        positions = np.asarray([agent.pos for agent in self.agents], dtype=np.float32)
        velocities = np.asarray([agent.vel for agent in self.agents], dtype=np.float32)
        safe_radii = np.asarray(
            [agent.safe_radius for agent in self.agents], dtype=np.float32
        )
        reached = np.asarray([agent.reached for agent in self.agents], dtype=bool)
        agent_a_max = np.asarray([agent.a_max for agent in self.agents], dtype=np.float32)
        agent_v_max = np.asarray([agent.v_max for agent in self.agents], dtype=np.float32)
        boundaries = np.asarray(
            [self.length, self.width, self.height][: self.dim_actions], dtype=np.float32
        )
        obstacle_pos = np.asarray(
            [obstacle.pos for obstacle in self.obstacles], dtype=np.float32
        )
        obstacle_radius = np.asarray(
            [obstacle.radius for obstacle in self.obstacles], dtype=np.float32
        )
        safe_values = np.zeros((batch_size, self.num_agents), dtype=np.float32)

        for batch_idx in range(batch_size):
            prev_pos = positions.copy()
            pred_vel = velocities.copy()
            batch_safe_value = np.zeros(self.num_agents, dtype=np.float32)

            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    pred_vel[agent_idx, :] = 0.0
                    prev_pos[agent_idx, :] = positions[agent_idx]
                    continue

                accel = actions_batch[batch_idx, agent_idx].copy()
                norm = np.linalg.norm(accel)
                if norm > agent_a_max[agent_idx]:
                    accel = accel / (norm + eps) * agent_a_max[agent_idx]

                pred_vel[agent_idx] = pred_vel[agent_idx] + accel * self.time_step
                speed = np.linalg.norm(pred_vel[agent_idx])
                if speed > agent_v_max[agent_idx]:
                    pred_vel[agent_idx] = (
                        pred_vel[agent_idx] / (speed + eps) * agent_v_max[agent_idx]
                    )

                prev_pos[agent_idx] = (
                    positions[agent_idx] + pred_vel[agent_idx] * self.time_step
                )

            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    continue

                boundary_risk = 0.0
                warning_clearance = safe_radii[agent_idx] + self.risk_warning_margin
                for dim_idx, boundary in enumerate(boundaries):
                    lower_clearance = float(prev_pos[agent_idx, dim_idx])
                    upper_clearance = float(boundary - prev_pos[agent_idx, dim_idx])
                    lower_overlap = max(0.0, warning_clearance - lower_clearance)
                    upper_overlap = max(0.0, warning_clearance - upper_clearance)
                    boundary_risk += self._nonlinear_risk_from_overlap(
                        lower_overlap, warning_clearance
                    )
                    boundary_risk += self._nonlinear_risk_from_overlap(
                        upper_overlap, warning_clearance
                    )
                    if prev_pos[agent_idx, dim_idx] - safe_radii[agent_idx] < 0:
                        prev_pos[agent_idx, dim_idx] = safe_radii[agent_idx]
                        pred_vel[agent_idx, dim_idx] = 0.0
                    elif (
                        prev_pos[agent_idx, dim_idx] + safe_radii[agent_idx] > boundary
                    ):
                        prev_pos[agent_idx, dim_idx] = boundary - safe_radii[agent_idx]
                        pred_vel[agent_idx, dim_idx] = 0.0
                batch_safe_value[agent_idx] += boundary_risk

                obstacle_risk = 0.0
                for obs_idx in range(len(self.obstacles)):
                    delta_xy = prev_pos[agent_idx, :2] - obstacle_pos[obs_idx]
                    dist_xy = np.linalg.norm(delta_xy)
                    collision_min_dist = obstacle_radius[obs_idx] + safe_radii[agent_idx]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    overlap = max(0.0, warning_min_dist - dist_xy)
                    obstacle_risk = max(
                        obstacle_risk,
                        self._nonlinear_risk_from_overlap(overlap, warning_min_dist),
                    )
                    if dist_xy < collision_min_dist:
                        if dist_xy < eps:
                            delta_xy = np.array([1.0, 0.0], dtype=np.float32)
                            dist_xy = 1.0
                        direction_xy = delta_xy / dist_xy
                        corrected_xy = (
                            obstacle_pos[obs_idx] + direction_xy * collision_min_dist
                        )
                        prev_pos[agent_idx, :2] = corrected_xy
                        pred_vel[agent_idx, :2] = 0.0
                batch_safe_value[agent_idx] += obstacle_risk

            for i in range(self.num_agents):
                if reached[i]:
                    continue
                for j in range(i + 1, self.num_agents):
                    if reached[j]:
                        continue
                    delta = prev_pos[i] - prev_pos[j]
                    dist = np.linalg.norm(delta)
                    collision_min_dist = safe_radii[i] + safe_radii[j]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    overlap = max(0.0, warning_min_dist - dist)
                    pair_risk = self._nonlinear_risk_from_overlap(
                        overlap, warning_min_dist
                    )
                    batch_safe_value[i] += pair_risk
                    batch_safe_value[j] += pair_risk
                    if dist < collision_min_dist:
                        if dist < eps:
                            delta = np.zeros(self.dim_actions, dtype=np.float32)
                            delta[0] = 1.0
                            dist = 1.0
                        direction = delta / dist
                        overlap = collision_min_dist - dist
                        prev_pos[i] += direction * (overlap / 2.0 + eps)
                        prev_pos[j] -= direction * (overlap / 2.0 + eps)
                        pred_vel[i, :] = 0.0
                        pred_vel[j, :] = 0.0

            safe_values[batch_idx] = batch_safe_value

        return safe_values

    def predict_joint_collision_flags(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        predicted_pos = []
        for agent, action in zip(self.agents, actions):
            if agent.reached or not agent.has_energy():
                predicted_pos.append(agent.pos.copy())
                continue
            pred_pos, _ = self._predict_agent_kinematics(agent, action)
            predicted_pos.append(pred_pos)

        collision_flags = np.zeros(self.num_agents, dtype=bool)

        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        for idx, (agent, pos) in enumerate(zip(self.agents, predicted_pos)):
            if agent.reached:
                continue
            guard_clearance = agent.safe_radius + self.guard_prediction_margin
            for dim, boundary in enumerate(boundaries):
                if (
                    pos[dim] - guard_clearance < 0.0
                    or pos[dim] + guard_clearance > boundary
                ):
                    collision_flags[idx] = True
                    break

        for idx, (agent, pos) in enumerate(zip(self.agents, predicted_pos)):
            if agent.reached:
                continue
            for obstacle in self.obstacles:
                delta_xy = pos[:2] - obstacle.pos
                dist_xy = np.linalg.norm(delta_xy)
                collision_min_dist = (
                    obstacle.radius + agent.safe_radius + self.guard_prediction_margin
                )
                if dist_xy < collision_min_dist:
                    collision_flags[idx] = True
                    break

        for i in range(self.num_agents):
            if self.agents[i].reached:
                continue
            for j in range(i + 1, self.num_agents):
                if self.agents[j].reached:
                    continue
                dist = np.linalg.norm(predicted_pos[i] - predicted_pos[j])
                collision_min_dist = (
                    self.agents[i].safe_radius
                    + self.agents[j].safe_radius
                    + self.guard_prediction_margin
                )
                if dist < collision_min_dist:
                    collision_flags[i] = True
                    collision_flags[j] = True

        return collision_flags.astype(np.float32)

    def _evaluate_joint_guard_horizon(self, actions_batch, horizon=None, guard_margin=None):
        actions_batch = np.asarray(actions_batch, dtype=np.float32)
        if actions_batch.ndim == 2:
            actions_batch = actions_batch.reshape(1, self.num_agents, self.dim_actions)
        if actions_batch.ndim != 3:
            raise ValueError(
                "actions_batch must have shape (batch, n_agents, dim_actions)"
            )
        if actions_batch.shape[1] != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")
        if actions_batch.shape[2] != self.dim_actions:
            raise ValueError("Action dimension does not match UAV action dimension.")

        horizon = self.guard_prediction_horizon if horizon is None else horizon
        horizon = max(1, int(horizon))
        margin = self.guard_prediction_margin if guard_margin is None else guard_margin
        margin = float(max(0.0, margin))

        batch_size = actions_batch.shape[0]
        positions = np.asarray([agent.pos for agent in self.agents], dtype=np.float32)
        velocities = np.asarray([agent.vel for agent in self.agents], dtype=np.float32)
        safe_radii = np.asarray(
            [agent.safe_radius for agent in self.agents], dtype=np.float32
        )
        moving = np.asarray(
            [
                (not agent.reached) and agent.has_energy()
                for agent in self.agents
            ],
            dtype=bool,
        )
        powered = np.asarray(
            [agent.has_energy() for agent in self.agents],
            dtype=bool,
        )
        agent_a_max = np.asarray([agent.a_max for agent in self.agents], dtype=np.float32)
        agent_v_max = np.asarray([agent.v_max for agent in self.agents], dtype=np.float32)
        boundaries = np.asarray(
            [self.length, self.width, self.height][: self.dim_actions],
            dtype=np.float32,
        )
        obstacle_pos = np.asarray(
            [obstacle.pos for obstacle in self.obstacles], dtype=np.float32
        )
        obstacle_radius = np.asarray(
            [obstacle.radius for obstacle in self.obstacles], dtype=np.float32
        )

        batch_flags = np.zeros((batch_size, self.num_agents), dtype=np.float32)
        batch_risk = np.zeros((batch_size, self.num_agents), dtype=np.float32)

        for batch_idx in range(batch_size):
            pos = positions.copy()
            vel = velocities.copy()
            actions = actions_batch[batch_idx].copy()

            for step_idx in range(horizon):
                for agent_idx in range(self.num_agents):
                    if not moving[agent_idx]:
                        continue

                    accel = actions[agent_idx]
                    norm = np.linalg.norm(accel)
                    if norm > agent_a_max[agent_idx]:
                        accel = accel / (norm + eps) * agent_a_max[agent_idx]

                    pos[agent_idx] = (
                        pos[agent_idx]
                        + vel[agent_idx] * self.time_step
                        + 0.5 * accel * (self.time_step ** 2)
                    )
                    vel[agent_idx] = vel[agent_idx] + accel * self.time_step
                    speed = np.linalg.norm(vel[agent_idx])
                    if speed > agent_v_max[agent_idx]:
                        vel[agent_idx] = (
                            vel[agent_idx] / (speed + eps) * agent_v_max[agent_idx]
                        )

                horizon_discount = 1.0 / float(step_idx + 1)

                for agent_idx in range(self.num_agents):
                    if not moving[agent_idx]:
                        continue

                    boundary_clearance = safe_radii[agent_idx] + margin
                    boundary_warning = boundary_clearance + self.risk_warning_margin
                    for dim_idx, boundary in enumerate(boundaries):
                        lower_clearance = float(pos[agent_idx, dim_idx])
                        upper_clearance = float(boundary - pos[agent_idx, dim_idx])
                        if (
                            lower_clearance < boundary_clearance
                            or upper_clearance < boundary_clearance
                        ):
                            batch_flags[batch_idx, agent_idx] = 1.0
                        lower_overlap = max(0.0, boundary_warning - lower_clearance)
                        upper_overlap = max(0.0, boundary_warning - upper_clearance)
                        batch_risk[batch_idx, agent_idx] += horizon_discount * (
                            self._nonlinear_risk_from_overlap(
                                lower_overlap, boundary_warning
                            )
                            + self._nonlinear_risk_from_overlap(
                                upper_overlap, boundary_warning
                            )
                        )

                    for obs_idx in range(len(self.obstacles)):
                        delta_xy = pos[agent_idx, :2] - obstacle_pos[obs_idx]
                        dist_xy = float(np.linalg.norm(delta_xy))
                        collision_min_dist = (
                            obstacle_radius[obs_idx] + safe_radii[agent_idx] + margin
                        )
                        warning_min_dist = (
                            collision_min_dist + self.risk_warning_margin
                        )
                        if dist_xy < collision_min_dist:
                            batch_flags[batch_idx, agent_idx] = 1.0
                        overlap = max(0.0, warning_min_dist - dist_xy)
                        batch_risk[batch_idx, agent_idx] += horizon_discount * (
                            self._nonlinear_risk_from_overlap(
                                overlap, warning_min_dist
                            )
                        )

                for i in range(self.num_agents):
                    if not powered[i]:
                        continue
                    for j in range(i + 1, self.num_agents):
                        if not powered[j]:
                            continue
                        dist = float(np.linalg.norm(pos[i] - pos[j]))
                        collision_min_dist = (
                            safe_radii[i] + safe_radii[j] + margin
                        )
                        warning_min_dist = (
                            collision_min_dist + self.risk_warning_margin
                        )
                        if dist < collision_min_dist:
                            batch_flags[batch_idx, i] = 1.0
                            batch_flags[batch_idx, j] = 1.0
                        overlap = max(0.0, warning_min_dist - dist)
                        pair_risk = horizon_discount * self._nonlinear_risk_from_overlap(
                            overlap, warning_min_dist
                        )
                        batch_risk[batch_idx, i] += pair_risk
                        batch_risk[batch_idx, j] += pair_risk

        return batch_flags, batch_risk

    def predict_joint_collision_flags_horizon(self, actions, horizon=None, guard_margin=None):
        flags, _ = self._evaluate_joint_guard_horizon(
            actions,
            horizon=horizon,
            guard_margin=guard_margin,
        )
        return flags[0].astype(np.float32)

    def estimate_joint_risk_horizon_batch(self, actions_batch, horizon=None, guard_margin=None):
        _, risk = self._evaluate_joint_guard_horizon(
            actions_batch,
            horizon=horizon,
            guard_margin=guard_margin,
        )
        return risk.astype(np.float32)

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        self.current_step += 1
        self._assign_orders()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        powered_mask = np.asarray(
            [agent.has_energy() for agent in self.agents],
            dtype=bool,
        )

        prev_dists = np.array([self._distance_to_goal(agent) for agent in self.agents], dtype=np.float32)
        task_types_before_step = [agent.current_task_type for agent in self.agents]
        order_status_before_step = []
        for agent in self.agents:
            if agent.assigned_order_id is None:
                order_status_before_step.append(None)
            else:
                order_status_before_step.append(
                    self.orders[agent.assigned_order_id].status
                )
        self._last_step_order_status_before = list(order_status_before_step)

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if powered_mask[idx] and agent.reached:
                self._advance_intermediate_subgoal_if_needed(agent)
            if (
                not powered_mask[idx]
                or agent.reached
                or not self._agent_has_motion_task(agent)
            ):
                agent.vel[:] = 0.0
                agent.prev_pos = agent.pos.copy()
                continue
            agent.update_velocity(action, self.time_step)
            agent.preview_position(self.time_step)

        obstacle_collisions = [False] * self.num_agents
        for idx, agent in enumerate(self.agents):
            if not powered_mask[idx]:
                agent.vel[:] = 0.0
                agent.prev_pos = agent.pos.copy()
                continue
            (
                boundary_collision,
                boundary_penalty,
                boundary_reward_penalty,
            ) = self._apply_boundary_constraints(agent)
            (
                obstacle_collision,
                obstacle_penalty,
                obstacle_reward_penalty,
            ) = self._resolve_obstacle_collisions(agent)
            obstacle_collisions[idx] = boundary_collision or obstacle_collision
            self.safe_value[idx] += boundary_penalty + obstacle_penalty
            self.reward_safe_value[idx] += (
                boundary_reward_penalty + obstacle_reward_penalty
            )

        agent_collisions = self._resolve_agent_collisions()

        for agent in self.agents:
            agent.pos = agent.prev_pos.copy()

        self._consume_step_energy(powered_mask)
        self._charge_agents_at_station()
        self.update_lasers()

        for idx, agent in enumerate(self.agents):
            if not powered_mask[idx]:
                agent.vel[:] = 0.0
                agent.prev_collided = agent.collided
                agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
                self.agent_paths[idx].append(agent.pos.copy())
                continue

            current_dist = self._distance_to_goal(agent)
            progress = prev_dists[idx] - current_dist
            goal_direction = agent.goal - agent.pos
            goal_direction_norm = np.linalg.norm(goal_direction)
            if goal_direction_norm > eps:
                goal_direction = goal_direction / goal_direction_norm
                velocity_toward_goal = float(np.dot(agent.vel, goal_direction)) / (
                    agent.v_max + eps
                )
            else:
                velocity_toward_goal = 0.0
            obstacle_penalty = self.obstacle_collision_penalty
            agent_penalty = self.agent_collision_penalty
            if agent.prev_collided:
                obstacle_penalty *= self.repeat_collision_scale
                agent_penalty *= self.repeat_collision_scale
            dense_reward_scale = (
                self.charge_dense_reward_scale
                if agent.current_task_type == TASK_CHARGE
                else 1.0
            )
            rewards[idx] += (
                dense_reward_scale
                * 2.5
                * progress
            )
            rewards[idx] += (
                dense_reward_scale
                * self.velocity_reward_weight
                * max(0.0, velocity_toward_goal)
            )
            rewards[idx] -= 0.01
            rewards[idx] -= obstacle_penalty * float(obstacle_collisions[idx])
            rewards[idx] -= agent_penalty * float(agent_collisions[idx])
            rewards[idx] -= 0.2 * self.reward_safe_value[idx]
            rewards[idx] -= dense_reward_scale * 0.3 * min(current_dist, 1.0)

            if (
                agent.current_task_type == TASK_ORDER
                and agent.assigned_order_id is not None
            ):
                self.order_diag_order_steps += 1.0
                self.order_diag_progress_sum += float(progress)
                self.order_diag_target_distance_sum += float(current_dist)
                if progress > 1e-4:
                    self.order_diag_positive_progress_steps += 1.0
                elif progress < -1e-4:
                    self.order_diag_regress_steps += 1.0
                order_status = order_status_before_step[idx]
                if order_status == DeliveryOrder.ASSIGNED:
                    self.order_diag_pickup_steps += 1.0
                elif order_status == DeliveryOrder.PICKED:
                    self.order_diag_delivery_steps += 1.0
            elif agent.current_task_type == TASK_CHARGE:
                self.order_diag_charge_steps += 1.0
            else:
                self.order_diag_idle_steps += 1.0

            if current_dist <= self.goal_tolerance:
                agent.reached = True
                if self._is_at_task_terminal_target(agent):
                    agent.vel[:] = 0.0

            prev_order_status = order_status_before_step[idx]
            order_reward = self._advance_order_if_reached(agent, current_dist)
            if order_reward > 0.0:
                agent.vel[:] = 0.0
                rewards[idx] += order_reward
                if prev_order_status == DeliveryOrder.ASSIGNED:
                    self.order_diag_pickup_successes += 1.0
                elif prev_order_status == DeliveryOrder.PICKED:
                    self.order_diag_delivery_successes += 1.0

            agent.prev_collided = agent.collided
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            self.agent_paths[idx].append(agent.pos.copy())

        for idx, task_type in enumerate(task_types_before_step):
            if task_type == TASK_ORDER:
                if obstacle_collisions[idx]:
                    self.order_diag_order_obstacle_collisions += 1.0
                if agent_collisions[idx]:
                    self.order_diag_order_agent_collisions += 1.0
            elif task_type == TASK_CHARGE:
                if obstacle_collisions[idx]:
                    self.order_diag_charge_obstacle_collisions += 1.0
                if agent_collisions[idx]:
                    self.order_diag_charge_agent_collisions += 1.0
            else:
                if obstacle_collisions[idx]:
                    self.order_diag_idle_obstacle_collisions += 1.0
                if agent_collisions[idx]:
                    self.order_diag_idle_agent_collisions += 1.0

        self._assign_orders()
        delivery_done = self._all_orders_completed()
        dones = [delivery_done for _ in self.agents]

        obstacle_collision_total = float(
            np.sum(np.asarray(obstacle_collisions, dtype=bool))
        )
        agent_collision_total = float(
            np.sum(np.asarray(agent_collisions, dtype=bool))
        )
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        if delivery_done:
            rewards += 5.0

        return self.get_obs(), rewards, dones, self.safe_value.copy()

    def _goal_features(self, agent):
        delta = agent.goal - agent.pos
        distance = np.linalg.norm(delta) / (np.linalg.norm(self._space_scale()) + eps)
        return np.concatenate([delta / (self._space_scale() + eps), np.array([distance], dtype=np.float32)])

    def _message_from_sender(self, receiver, sender):
        del receiver
        scale = self._space_scale() + eps
        goal_delta = (sender.goal - sender.pos) / scale
        velocity = sender.vel / (sender.v_max + eps)
        sender_energy = np.array(
            [sender.energy / (sender.initial_energy + eps)],
            dtype=np.float32,
        )
        sender_risk = np.array(
            [
                min(
                    1.0,
                    float(self.safe_value[sender.number]) / (sender.safe_radius + eps),
                )
            ],
            dtype=np.float32,
        )
        sender_reached = np.array([float(sender.reached)], dtype=np.float32)
        return np.concatenate(
            [
                goal_delta.astype(np.float32),
                velocity.astype(np.float32),
                sender_energy,
                sender_risk,
                sender_reached,
            ]
        )

    def get_msg(self):
        messages = []
        for receiver in self.agents:
            receiver_msgs = []
            for sender in self.agents:
                if sender is receiver:
                    continue
                receiver_msgs.append(self._message_from_sender(receiver, sender))
            if receiver_msgs:
                messages.append(np.stack(receiver_msgs, axis=0).astype(np.float32))
            else:
                messages.append(
                    np.zeros((0, self.msg_shape), dtype=np.float32)
                )
        return messages

    def get_agent_positions(self):
        return np.stack([agent.pos.copy() for agent in self.agents], axis=0).astype(
            np.float32
        )

    def get_current_subgoals(self):
        return np.stack([agent.subgoal.copy() for agent in self.agents], axis=0).astype(
            np.float32
        )

    def get_subgoal_distances(self, targets=None):
        if targets is None:
            targets = self.get_current_subgoals()
        targets = np.asarray(targets, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        positions = self.get_agent_positions()
        return np.linalg.norm(positions - targets, axis=1).astype(np.float32)

    def get_subgoal_success_mask(self, targets=None):
        distances = self.get_subgoal_distances(targets)
        return (distances <= self.goal_tolerance).astype(np.float32)

    def compute_intrinsic_rewards(self, prev_distances=None, targets=None):
        distances = self.get_subgoal_distances(targets)
        if prev_distances is None:
            progress = np.zeros_like(distances, dtype=np.float32)
        else:
            prev_distances = np.asarray(prev_distances, dtype=np.float32).reshape(-1)
            progress = prev_distances - distances
        space_norm = np.linalg.norm(self._space_scale()) + eps
        rewards = progress - self.intrinsic_distance_weight * (distances / space_norm)
        rewards += self.intrinsic_success_bonus * self.get_subgoal_success_mask(targets)
        delivery_mask = np.asarray(
            [
                float(
                    agent.current_task_type == TASK_ORDER
                    and agent.assigned_order_id is not None
                    and status == DeliveryOrder.PICKED
                )
                for agent, status in zip(
                    self.agents,
                    getattr(
                        self,
                        "_last_step_order_status_before",
                        [None for _ in range(self.num_agents)],
                    ),
                )
            ],
            dtype=np.float32,
        )
        rewards += (
            self.delivery_intrinsic_progress_bonus
            * np.maximum(progress, 0.0)
            * delivery_mask
        )
        if self.intrinsic_collision_penalty > 0.0:
            collision_mask = np.asarray(
                [float(agent.collided) for agent in self.agents], dtype=np.float32
            )
            rewards -= self.intrinsic_collision_penalty * collision_mask
        rewards -= 0.2 * np.asarray(self.reward_safe_value, dtype=np.float32)
        return (self.intrinsic_reward_scale * rewards).astype(np.float32)

    def relabel_observations_with_subgoals(self, observations, subgoals):
        obs = np.asarray(observations, dtype=np.float32).copy()
        original_shape = obs.shape
        obs = obs.reshape((-1, self.num_agents, original_shape[-1]))
        subgoals = np.asarray(subgoals, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        scale = self._space_scale() + eps
        goal_start = 2 * self.dim_actions + self.num_lasers
        goal_end = goal_start + self.dim_actions + 1
        for batch_idx in range(obs.shape[0]):
            for agent_idx in range(self.num_agents):
                pos = obs[batch_idx, agent_idx, : self.dim_actions] * scale
                delta = subgoals[agent_idx] - pos
                distance = np.linalg.norm(delta) / (np.linalg.norm(scale) + eps)
                goal_features = np.concatenate(
                    [
                        delta / scale,
                        np.array([distance], dtype=np.float32),
                    ]
                )
                obs[batch_idx, agent_idx, goal_start:goal_end] = goal_features
        return obs.reshape(original_shape).astype(np.float32)

    def get_obs(self):
        observations = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            own = np.concatenate([agent.pos / scale, agent.vel / (agent.v_max + eps)])
            energy = np.array(
                [agent.energy / (agent.initial_energy + eps)],
                dtype=np.float32,
            )
            obs = np.concatenate(
                [
                    own,
                    np.asarray(agent.lasers, dtype=np.float32),
                    self._goal_features(agent),
                    energy,
                ]
            )
            observations.append(obs.astype(np.float32))
        return np.stack(observations, axis=0)

    def get_state(self):
        parts = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            parts.append(agent.pos / scale)
            parts.append(agent.vel / (agent.v_max + eps))
            parts.append(agent.goal / scale)
            parts.append(np.array([float(agent.reached)], dtype=np.float32))
            parts.append(
                np.array(
                    [agent.energy / (agent.initial_energy + eps)],
                    dtype=np.float32,
                )
            )
        for obstacle in self.obstacles:
            parts.append(obstacle.pos / scale[:2])
            parts.append(np.array([obstacle.radius / max(self.length, self.width)], dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def update_lasers(self):
        for agent in self.agents:
            current_lasers = np.full(agent.num_lasers, agent.l_sensor, dtype=np.float32)
            for obstacle in self.obstacles:
                radius = obstacle.radius + agent.safe_radius
                obstacle_lasers = update_lasers_to_obstacle(
                    agent.pos[:2], obstacle.pos, radius, agent.l_sensor, agent.num_lasers
                )
                current_lasers = np.minimum(current_lasers, obstacle_lasers)
            for other in self.agents:
                if other is agent:
                    continue
                radius = agent.safe_radius + other.safe_radius
                agent_lasers = update_lasers_to_obstacle(
                    agent.pos[:2],
                    other.pos[:2],
                    radius,
                    agent.l_sensor,
                    agent.num_lasers,
                )
                current_lasers = np.minimum(current_lasers, agent_lasers)
            boundary_lasers = update_lasers_to_boundary(
                agent.pos[:2], agent.l_sensor, agent.num_lasers, self.length, self.width
            )
            agent.lasers = np.minimum(current_lasers, boundary_lasers)

    def summary(self):
        remaining = [
            self._distance_to_goal(agent)
            for agent in self.agents
            if agent.assigned_order_id is not None
        ]
        active_orders = self._active_order_count()
        available_orders = len(self.available_order_slots)
        picked_orders = sum(
            1 for order in self.orders if order.status == DeliveryOrder.PICKED
        )
        healthy_agents = float(np.sum([not agent.collided for agent in self.agents]))
        powered_agents = float(np.sum([agent.has_energy() for agent in self.agents]))
        mean_energy = (
            float(np.mean([agent.energy for agent in self.agents]))
            if self.agents
            else 0.0
        )
        reserved_charge_slots = float(
            sum(
                holder is not None
                for slots in self.charging_slot_reservations.values()
                for holder in slots
            )
        )
        charge_task_agents = float(
            np.sum([agent.current_task_type == TASK_CHARGE for agent in self.agents])
        )
        waiting_charge_agents = max(0.0, charge_task_agents - reserved_charge_slots)
        mean_goal_distance = float(np.mean(remaining)) if remaining else 0.0
        completed_orders = float(self.completed_order_count)
        summary = {
            "step": float(self.current_step),
            "agent_health": healthy_agents,
            "enemy_health": mean_goal_distance,
            "agent_alive": completed_orders,
            "collision_count": float(self.collision_count),
            "obstacle_collision_count": float(self.obstacle_collision_count),
            "agent_collision_count": float(self.agent_collision_count),
            "orders_completed": completed_orders,
            "total_orders": float(self.total_orders),
            "active_orders": float(active_orders),
            "available_orders": float(available_orders),
            "picked_orders": float(picked_orders),
            "idle_agents": float(
                np.sum([agent.current_task_type == TASK_IDLE for agent in self.agents])
            ),
            "available_order_slots": float(len(self.available_order_slots)),
            "healthy_agents": healthy_agents,
            "powered_agents": powered_agents,
            "depleted_agents": float(self.num_agents - powered_agents),
            "mean_energy": mean_energy,
            "energy_decay_per_step": float(self.energy_decay_per_step),
            "charging_station_count": float(self.charging_station_count),
            "charging_capacity": float(self.charging_capacity),
            "total_charging_capacity": float(
                self.charging_station_count * self.charging_capacity
            ),
            "charging_agents": float(len(self.charging_agent_ids)),
            "reserved_charge_slots": reserved_charge_slots,
            "charge_task_agents": charge_task_agents,
            "waiting_charge_agents": waiting_charge_agents,
            "mean_goal_distance": mean_goal_distance,
            "episode_reward": float(0.0),
            "win_tag": bool(self._all_orders_completed()),
        }
        summary.update(self._high_level_diagnostics_summary())
        summary.update(self._order_execution_diagnostics_summary())
        return summary

    def _capture_runtime_state(self):
        return {
            "current_step": int(self.current_step),
            "safe_value": self.safe_value.copy(),
            "reward_safe_value": self.reward_safe_value.copy(),
            "collision_count": float(self.collision_count),
            "obstacle_collision_count": float(self.obstacle_collision_count),
            "agent_collision_count": float(self.agent_collision_count),
            "order_execution_diagnostics": self._order_execution_diagnostics_state(),
            "agent_paths": [[pos.copy() for pos in path] for path in self.agent_paths],
            "order_slots": list(self.order_slots),
            "available_order_slots": list(self.available_order_slots),
            "active_order_ids": list(self.active_order_ids),
            "next_order_id_to_activate": int(self.next_order_id_to_activate),
            "completed_order_count": int(self.completed_order_count),
            "last_step_order_status_before": list(
                getattr(
                    self,
                    "_last_step_order_status_before",
                    [None for _ in range(self.num_agents)],
                )
            ),
            "charging_agent_ids": list(self.charging_agent_ids),
            "charging_station_agent_ids": {
                int(station_idx): list(agent_ids)
                for station_idx, agent_ids in self.charging_station_agent_ids.items()
            },
            "charging_slot_reservations": {
                int(station_idx): list(agent_ids)
                for station_idx, agent_ids in self.charging_slot_reservations.items()
            },
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "orders": [
                {
                    "order_id": int(order.order_id),
                    "pickup_pos": order.pickup_pos.copy(),
                    "dropoff_pos": order.dropoff_pos.copy(),
                    "status": order.status,
                    "assigned_agent": order.assigned_agent,
                }
                for order in self.orders
            ],
            "agents": [
                {
                    "pos": agent.pos.copy(),
                    "prev_pos": agent.prev_pos.copy(),
                    "last_pos": agent.last_pos.copy(),
                    "goal": agent.goal.copy(),
                    "vel": agent.vel.copy(),
                    "lasers": agent.lasers.copy(),
                    "reached": bool(agent.reached),
                    "collided": bool(agent.collided),
                    "prev_collided": bool(agent.prev_collided),
                    "assigned_order_id": agent.assigned_order_id,
                    "assigned_order_slot": agent.assigned_order_slot,
                    "auction_order_slot": agent.auction_order_slot,
                    "charge_station_idx": agent.charge_station_idx,
                    "charge_slot_idx": agent.charge_slot_idx,
                    "task_target": agent.task_target.copy(),
                    "subgoal": agent.subgoal.copy(),
                    "original_subgoal": agent.original_subgoal.copy(),
                    "subgoal_test": bool(agent.subgoal_test),
                    "carrying_order": bool(agent.carrying_order),
                    "current_task_type": int(agent.current_task_type),
                    "completed_orders": int(agent.completed_orders),
                    "energy": float(agent.energy),
                    "initial_energy": float(agent.initial_energy),
                }
                for agent in self.agents
            ],
        }

    def _restore_runtime_state(self, snapshot):
        self.current_step = int(snapshot["current_step"])
        self.safe_value = snapshot["safe_value"].copy()
        self.reward_safe_value = snapshot["reward_safe_value"].copy()
        self.collision_count = float(snapshot["collision_count"])
        self.obstacle_collision_count = float(
            snapshot.get("obstacle_collision_count", 0.0)
        )
        self.agent_collision_count = float(
            snapshot.get("agent_collision_count", 0.0)
        )
        for attr, value in snapshot.get("order_execution_diagnostics", {}).items():
            if attr in ORDER_EXEC_DIAGNOSTIC_ATTRS:
                setattr(self, attr, float(value))
        self.agent_paths = [[pos.copy() for pos in path] for path in snapshot["agent_paths"]]
        self.order_slots = list(
            snapshot.get("order_slots", [None for _ in range(self.max_active_orders)])
        )
        if len(self.order_slots) < self.max_active_orders:
            self.order_slots.extend(
                [None for _ in range(self.max_active_orders - len(self.order_slots))]
            )
        self.order_slots = self.order_slots[: self.max_active_orders]
        self.available_order_slots = list(snapshot.get("available_order_slots", []))
        self.active_order_ids = list(snapshot.get("active_order_ids", []))
        self.next_order_id_to_activate = int(
            snapshot.get("next_order_id_to_activate", 0)
        )
        self.completed_order_count = int(snapshot.get("completed_order_count", 0))
        self._last_step_order_status_before = list(
            snapshot.get(
                "last_step_order_status_before",
                [None for _ in range(self.num_agents)],
            )
        )
        self.charging_agent_ids = list(snapshot.get("charging_agent_ids", []))
        self.charging_station_agent_ids = {
            int(station_idx): list(agent_ids)
            for station_idx, agent_ids in snapshot.get(
                "charging_station_agent_ids",
                {idx: [] for idx in range(self.charging_station_count)},
            ).items()
        }
        self.charging_slot_reservations = {
            int(station_idx): list(agent_ids)
            for station_idx, agent_ids in snapshot.get(
                "charging_slot_reservations",
                self._empty_charge_reservations(),
            ).items()
        }
        if "python_random_state" in snapshot:
            random.setstate(snapshot["python_random_state"])
        if "numpy_random_state" in snapshot:
            np.random.set_state(snapshot["numpy_random_state"])
        self.orders = []
        for state in snapshot.get("orders", []):
            order = DeliveryOrder(
                state["order_id"],
                state["pickup_pos"].copy(),
                state["dropoff_pos"].copy(),
            )
            order.status = state["status"]
            order.assigned_agent = state["assigned_agent"]
            self.orders.append(order)
        if "order_slots" not in snapshot:
            self.order_slots = [None for _ in range(self.max_active_orders)]
            for slot_idx, order_id in enumerate(self.active_order_ids[: self.max_active_orders]):
                self.order_slots[slot_idx] = int(order_id)
        self._sync_goals_from_orders()
        for agent, state in zip(self.agents, snapshot["agents"]):
            agent.pos = state["pos"].copy()
            agent.prev_pos = state["prev_pos"].copy()
            agent.last_pos = state["last_pos"].copy()
            agent.goal = state["goal"].copy()
            agent.vel = state["vel"].copy()
            agent.lasers = state["lasers"].copy()
            agent.reached = bool(state["reached"])
            agent.collided = bool(state["collided"])
            agent.prev_collided = bool(state["prev_collided"])
            agent.assigned_order_id = state.get("assigned_order_id")
            agent.assigned_order_slot = state.get("assigned_order_slot")
            agent.auction_order_slot = state.get("auction_order_slot")
            agent.charge_station_idx = state.get("charge_station_idx")
            agent.charge_slot_idx = state.get("charge_slot_idx")
            agent.task_target = state.get("task_target", agent.goal).copy()
            agent.subgoal = state.get("subgoal", agent.goal).copy()
            agent.original_subgoal = state.get("original_subgoal", agent.subgoal).copy()
            agent.subgoal_test = bool(state.get("subgoal_test", False))
            agent.carrying_order = bool(state.get("carrying_order", False))
            default_task_type = (
                TASK_ORDER if agent.assigned_order_id is not None else TASK_IDLE
            )
            agent.current_task_type = int(
                state.get("current_task_type", default_task_type)
            )
            agent.completed_orders = int(state.get("completed_orders", 0))
            agent.energy = float(state.get("energy", self.initial_energy))
            agent.initial_energy = float(
                state.get("initial_energy", self.initial_energy)
            )

    def estimate_joint_risk(self, actions):
        snapshot = self._capture_runtime_state()
        try:
            _, _, _, safe_value = self.step(actions)
            return np.asarray(safe_value, dtype=np.float32).copy()
        finally:
            self._restore_runtime_state(snapshot)

    def estimate_joint_risk_batch(self, actions_batch):
        return self._evaluate_joint_risk_batch_exact(actions_batch)

    def _agent_color(self, idx):
        cmap = __import__("matplotlib").cm.get_cmap("tab10", max(self.num_agents, 1))
        return cmap(idx % max(self.num_agents, 1))

    def _visible_orders(self):
        orders = []
        for order_id in self.order_slots:
            if order_id is None or order_id >= len(self.orders):
                continue
            order = self.orders[order_id]
            if order.status != DeliveryOrder.COMPLETED:
                orders.append(order)
        return orders

    def _order_color(self, order):
        if order.assigned_agent is not None:
            return self._agent_color(order.assigned_agent)
        return "#2ca02c"

    def _assigned_order_render_target(self, agent):
        if agent.assigned_order_id is None:
            return None
        if agent.assigned_order_id >= len(self.orders):
            return None
        order = self.orders[agent.assigned_order_id]
        if order.status == DeliveryOrder.COMPLETED:
            return None
        if order.status == DeliveryOrder.PICKED or agent.carrying_order:
            return order.dropoff_pos
        return order.pickup_pos

    def _math_label_font(self):
        from matplotlib import font_manager

        cached = getattr(self, "_axis_math_font", None)
        if cached is not None:
            return cached

        candidates = [
            os.path.join(os.getcwd(), "fonts", "latinmodern-math.ttf"),
            os.path.join(os.getcwd(), "fonts", "latinmodern-math.otf"),
            "/mnt/c/Windows/Fonts/latinmodern-math.ttf",
            "/mnt/c/Windows/Fonts/latinmodern-math.otf",
            "/usr/share/fonts/truetype/lmodern/latinmodern-math.ttf",
            "/usr/share/fonts/opentype/lmodern/latinmodern-math.otf",
            "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm-math/latinmodern-math.otf",
            "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf",
        ]
        font_prop = None
        for font_path in candidates:
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                break
        if font_prop is None:
            try:
                font_path = font_manager.findfont(
                    "Latin Modern Math", fallback_to_default=False
                )
                font_prop = font_manager.FontProperties(fname=font_path)
            except ValueError:
                font_prop = font_manager.FontProperties(family="Times New Roman")
        self._axis_math_font = font_prop
        return font_prop

    def _set_axis_labels(self, ax, is_3d=False):
        label_font = self._math_label_font()
        ax.set_xlabel(r"$x$", fontproperties=label_font)
        ax.set_ylabel(r"$y$", fontproperties=label_font)
        if not is_3d:
            ax.yaxis.label.set_rotation(0)
        if is_3d:
            ax.set_zlabel(r"$z$", fontproperties=label_font)

    def _style_axes(self, ax, is_3d=False):
        font = "Times New Roman"
        label_font = self._math_label_font()
        label_size = 24
        tick_size = 26
        ax.set_facecolor("white")
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        tick_pad = -2 if is_3d else 14
        ax.tick_params(axis="x", labelsize=tick_size, pad=tick_pad)
        ax.tick_params(axis="y", labelsize=tick_size, pad=tick_pad)
        ax.xaxis.label.set_fontproperties(label_font)
        ax.yaxis.label.set_fontproperties(label_font)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.xaxis.labelpad = 7
        ax.yaxis.labelpad = 7
        if is_3d:
            ax.tick_params(axis="z", labelsize=tick_size, pad=1)
            ax.zaxis.label.set_fontproperties(label_font)
            ax.zaxis.label.set_fontsize(label_size)
            ax.zaxis.labelpad = 2
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font)
            label.set_fontsize(tick_size)
        if is_3d:
            for label in ax.get_zticklabels():
                label.set_fontname(font)
                label.set_fontsize(tick_size)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                    axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
                    axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.8)
                    axis._axinfo["grid"]["linewidth"] = 0.6
                except AttributeError:
                    pass
            try:
                ax.set_proj_type("ortho")
            except AttributeError:
                pass

    def _add_render_legend(self, ax, is_3d=False):
        font_size = legend_font_size_3d if is_3d else legend_font_size_xy

        legend_anchor = (0.98, 0.98)
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=legend_anchor,
            prop={"family": "Times New Roman", "size": font_size},
            markerscale=1.35,
            frameon=True,
            framealpha=0.88,
            borderpad=0.18,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.35,
            borderaxespad=0.25,
            scatterpoints=1,
        )
        if legend is not None:
            frame = legend.get_frame()
            frame.set_linewidth(0.6)
            frame.set_edgecolor("#666666")

    def _set_vertical_z_view(self, ax):
        try:
            ax.view_init(elev=25, azim=45, roll=0)
        except TypeError:
            ax.view_init(elev=25, azim=45)

    def _set_integer_ticks(self, ax, is_3d=False):
        ax.set_xticks(np.arange(0, int(round(self.length)) + 1, 1))
        ax.set_yticks(np.arange(0, int(round(self.width)) + 1, 1))
        if is_3d:
            ax.set_zticks(np.arange(0, int(round(self.height)) + 1, 1))

    def _agent_trajectory(self, idx, agent):
        path = [pos.copy() for pos in self.agent_paths[idx]]
        if not path or np.linalg.norm(path[-1] - agent.pos) > eps:
            path.append(agent.pos.copy())
        return np.asarray(path, dtype=np.float32)

    def _render_2d(self, ax):
        from matplotlib.patches import Circle

        pickup_labeled = False
        dropoff_labeled = False
        for order in self._visible_orders():
            color = self._order_color(order)
            if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
                ax.scatter(
                    order.pickup_pos[0],
                    order.pickup_pos[1],
                    c=[color],
                    marker="^",
                    s=115,
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.9,
                    label="pickup" if not pickup_labeled else None,
                )
                pickup_labeled = True
            ax.scatter(
                order.dropoff_pos[0],
                order.dropoff_pos[1],
                c=[color],
                marker="s",
                s=105,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="dropoff" if not dropoff_labeled else None,
            )
            dropoff_labeled = True

        assigned_labeled = False
        for idx, agent in enumerate(self.agents):
            color = self._agent_color(idx)
            trajectory = self._agent_trajectory(idx, agent)
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.9, linewidth=1.8)
            assigned_target = self._assigned_order_render_target(agent)
            if assigned_target is not None:
                ax.plot(
                    [agent.pos[0], assigned_target[0]],
                    [agent.pos[1], assigned_target[1]],
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.78,
                    label="assigned order" if not assigned_labeled else None,
                )
                assigned_labeled = True
            ax.scatter(
                agent.pos[0],
                agent.pos[1],
                c=[color],
                s=90,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="hunter" if idx == 0 else None,
            )
            ax.text(
                agent.pos[0] + 0.045,
                agent.pos[1] + 0.045,
                f"U{idx}",
                color=color,
                fontsize=10,
                fontweight="bold",
                ha="left",
                va="bottom",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "white",
                    "edgecolor": color,
                    "linewidth": 0.6,
                    "alpha": 0.82,
                },
            )

        for obstacle in self.obstacles:
            ax.add_patch(Circle(
                obstacle.pos,
                obstacle.radius,
                color="gray",
                alpha=0.5,
            ))

        for station_idx, station_pos in enumerate(self.charging_station_positions):
            ax.scatter(
                station_pos[0],
                station_pos[1],
                c=["#1f77b4"],
                marker="P",
                s=145,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="charging station" if station_idx == 0 else None,
            )

        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.width + 0.1)
        ax.set_aspect("equal", adjustable="box")
        self._set_integer_ticks(ax, is_3d=False)
        self._set_axis_labels(ax, is_3d=False)
        self._style_axes(ax, is_3d=False)
        ax.yaxis.set_label_coords(-0.14, 0.5)

    def _render_3d(self, ax):
        theta = np.linspace(0.0, 2.0 * math.pi, 24)
        z_values = np.linspace(0.0, self.height, 8)

        pickup_labeled = False
        dropoff_labeled = False
        for order in self._visible_orders():
            color = self._order_color(order)
            pickup_z = order.pickup_pos[2] if self.dim_actions == 3 else 0.0
            dropoff_z = order.dropoff_pos[2] if self.dim_actions == 3 else 0.0
            if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
                ax.scatter(
                    order.pickup_pos[0],
                    order.pickup_pos[1],
                    pickup_z,
                    c=[color],
                    marker="^",
                    s=115,
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.9,
                    label="pickup" if not pickup_labeled else None,
                )
                pickup_labeled = True
            ax.scatter(
                order.dropoff_pos[0],
                order.dropoff_pos[1],
                dropoff_z,
                c=[color],
                marker="s",
                s=105,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="dropoff" if not dropoff_labeled else None,
            )
            dropoff_labeled = True

        assigned_labeled = False
        for idx, agent in enumerate(self.agents):
            color = self._agent_color(idx)
            trajectory = self._agent_trajectory(idx, agent)
            if len(trajectory) > 1:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color=color,
                    alpha=0.9,
                    linewidth=1.8,
                )

            assigned_target = self._assigned_order_render_target(agent)
            if assigned_target is not None:
                target_z = assigned_target[2] if self.dim_actions == 3 else 0.0
                ax.plot(
                    [agent.pos[0], assigned_target[0]],
                    [agent.pos[1], assigned_target[1]],
                    [agent.pos[2], target_z],
                    color=color,
                    linestyle="--",
                    linewidth=1.4,
                    alpha=0.78,
                    label="assigned order" if not assigned_labeled else None,
                )
                assigned_labeled = True

            ax.scatter(
                agent.pos[0],
                agent.pos[1],
                agent.pos[2],
                c=[color],
                s=70,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="hunter" if idx == 0 else None,
            )
            ax.text(
                agent.pos[0],
                agent.pos[1],
                agent.pos[2] + 0.04,
                f"U{idx}",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

            speed = np.linalg.norm(agent.vel)
            if speed > eps:
                ax.quiver(
                    agent.pos[0],
                    agent.pos[1],
                    agent.pos[2],
                    agent.vel[0],
                    agent.vel[1],
                    agent.vel[2],
                    length=0.4,
                    normalize=True,
                    color=color,
                )

        theta_grid, z_grid = np.meshgrid(theta, z_values)
        for obstacle in self.obstacles:
            x_grid = obstacle.pos[0] + obstacle.radius * np.cos(theta_grid)
            y_grid = obstacle.pos[1] + obstacle.radius * np.sin(theta_grid)
            ax.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                color="#d8d8d8",
                alpha=0.22,
                linewidth=0,
                shade=True,
            )

        for station_idx, station_pos in enumerate(self.charging_station_positions):
            station_z = station_pos[2] if self.dim_actions == 3 else 0.0
            ax.scatter(
                station_pos[0],
                station_pos[1],
                station_z,
                c=["#1f77b4"],
                marker="P",
                s=130,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="charging station" if station_idx == 0 else None,
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

    def render(self, show=False, view=None):
        import matplotlib.backends.backend_agg as agg
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = "Times New Roman"
        render_xy = view == "xy" or self.dim_actions != 3

        if show:
            if self._render_fig is None:
                self._render_fig = plt.figure("UAVEnv")
                self._render_has_shown = False
            fig = self._render_fig
        else:
            fig = plt.figure("UAVEnv-rgb")
        fig.set_size_inches((5.2, 5.2), forward=True)
        fig.clf()
        fig.patch.set_facecolor("white")

        if not render_xy and self.dim_actions == 3:
            ax = fig.add_subplot(111, projection="3d")
            self._render_3d(ax)
        else:
            ax = fig.add_subplot(111)
            self._render_2d(ax)

        if render_xy:
            fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.985)
        else:
            fig.subplots_adjust(left=0.02, right=1.06, bottom=0.07, top=1.02)

        if show:
            plt.ion()
            if not self._render_has_shown:
                plt.show()
                self._render_has_shown = True
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except NotImplementedError:
                pass
            return None

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())

    def close(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        self._render_fig = None
        self._render_has_shown = False
        plt.close("all")


class UAVEnvDiscreteWrapper:
    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=8,
        num_targets=1,
        episode_limit=200,
        total_orders=8,
        max_active_orders=4,
        pickup_reward=3.0,
        delivery_reward=8.0,
        initial_energy=default_initial_energy,
        energy_decay_per_step=None,
        energy_depletion_fraction=default_energy_depletion_fraction,
        charging_capacity=default_charging_capacity,
        charging_station_count=default_charging_station_count,
        charging_radius=default_charging_radius,
        charging_rate=None,
        charging_station_pos=None,
        charge_mode_fraction=0.5,
        charge_dense_reward_scale=1.0,
        auction_enabled=True,
        fixed_charge_threshold_enabled=False,
        fixed_charge_threshold=0.35,
        fixed_charge_release_threshold=0.65,
    ):
        self.env = UAVEnv(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=episode_limit,
            total_orders=total_orders,
            max_active_orders=max_active_orders,
            pickup_reward=pickup_reward,
            delivery_reward=delivery_reward,
            initial_energy=initial_energy,
            energy_decay_per_step=energy_decay_per_step,
            energy_depletion_fraction=energy_depletion_fraction,
            charging_capacity=charging_capacity,
            charging_station_count=charging_station_count,
            charging_radius=charging_radius,
            charging_rate=charging_rate,
            charging_station_pos=charging_station_pos,
            charge_mode_fraction=charge_mode_fraction,
            charge_dense_reward_scale=charge_dense_reward_scale,
            auction_enabled=auction_enabled,
            fixed_charge_threshold_enabled=fixed_charge_threshold_enabled,
            fixed_charge_threshold=fixed_charge_threshold,
            fixed_charge_release_threshold=fixed_charge_release_threshold,
        )
        self.dim_actions = dim_actions
        self.episode_limit = episode_limit
        self.n_agents = self.env.num_agents
        self.discrete_actions = self._build_discrete_actions()
        self.n_actions = len(self.discrete_actions)
        self._episode_steps = 0
        self._last_obs = None
        self._last_reward = 0.0
        self._last_info = {}

    def _build_discrete_actions(self):
        action_values = [-1.0, 0.0, 1.0]
        discrete_actions = []
        for action in itertools.product(action_values, repeat=self.dim_actions):
            vec = np.array(action, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > eps:
                vec = vec / norm
            discrete_actions.append(vec * self.env.a_max)
        return discrete_actions

    def get_noop_action(self):
        for idx, action in enumerate(self.discrete_actions):
            if np.linalg.norm(action) <= eps:
                return idx
        return 0

    def reset(self, seed=None):
        self._episode_steps = 0
        self._last_obs = self.env.reset(seed=seed)
        self._last_reward = 0.0
        self._last_info = {
            "battle_won": False,
            "warning_signal": np.zeros((self.n_agents, 1), dtype=np.float32),
        }
        return self.get_obs()

    def step(self, actions):
        active_mask = self.get_active_agent_mask()
        noop_action = self.get_noop_action()
        actions = [
            int(action) if active_mask[idx] > 0.0 else noop_action
            for idx, action in enumerate(actions)
        ]
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        self._last_obs, rewards, dones, safe_value = self.env.step(agent_actions)
        self._episode_steps += 1
        active_count = float(np.sum(active_mask))
        reward = (
            float(np.sum(rewards * active_mask) / active_count)
            if active_count > 0.0
            else 0.0
        )
        battle_won = bool(self.env._all_orders_completed())
        post_step_active_mask = self.get_active_agent_mask()
        all_depleted = bool(np.sum(post_step_active_mask) <= 0.0)
        time_limit_reached = self._episode_steps >= self.episode_limit
        terminated = battle_won or all_depleted or time_limit_reached
        self._last_reward = reward
        self._last_info = {
            "battle_won": battle_won,
            "all_depleted": all_depleted,
            "time_limit_reached": time_limit_reached,
            "warning_signal": np.asarray(safe_value, dtype=np.float32).reshape(self.n_agents, 1),
            "per_agent_reward": np.asarray(rewards, dtype=np.float32).copy(),
            "agent_active_mask": np.asarray(post_step_active_mask, dtype=np.float32).reshape(self.n_agents, 1),
            "agent_energy": np.asarray(
                [agent.energy for agent in self.env.agents],
                dtype=np.float32,
            ).reshape(self.n_agents, 1),
        }
        return reward, terminated, self._last_info

    def estimate_joint_short_risk(self, actions):
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        return self.env.estimate_joint_risk(agent_actions)

    def estimate_joint_short_risk_batch(self, actions_batch):
        actions_batch = np.asarray(actions_batch)
        if actions_batch.ndim != 2:
            raise ValueError("actions_batch must have shape (batch, n_agents)")
        agent_actions_batch = np.asarray(
            [
                [self.discrete_actions[int(action)] for action in joint_actions]
                for joint_actions in actions_batch
            ],
            dtype=np.float32,
        )
        return self.env.estimate_joint_risk_batch(agent_actions_batch)

    def estimate_joint_collision_flags(self, actions):
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        return self.env.predict_joint_collision_flags(agent_actions)

    def estimate_joint_short_risk_horizon_batch(self, actions_batch, horizon=None, guard_margin=None):
        actions_batch = np.asarray(actions_batch)
        if actions_batch.ndim != 2:
            raise ValueError("actions_batch must have shape (batch, n_agents)")
        agent_actions_batch = np.asarray(
            [
                [self.discrete_actions[int(action)] for action in joint_actions]
                for joint_actions in actions_batch
            ],
            dtype=np.float32,
        )
        return self.env.estimate_joint_risk_horizon_batch(
            agent_actions_batch,
            horizon=horizon,
            guard_margin=guard_margin,
        )

    def estimate_joint_collision_flags_horizon(self, actions, horizon=None, guard_margin=None):
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        return self.env.predict_joint_collision_flags_horizon(
            agent_actions,
            horizon=horizon,
            guard_margin=guard_margin,
        )

    def _candidate_goal_progress(self, agent_id, action_id):
        agent = self.env.agents[int(agent_id)]
        if agent.reached or not agent.has_energy():
            return 0.0
        pred_pos, _ = self.env._predict_agent_kinematics(
            agent,
            self.discrete_actions[int(action_id)],
        )
        current_dist = float(np.linalg.norm(agent.goal - agent.pos))
        pred_dist = float(np.linalg.norm(agent.goal - pred_pos))
        return current_dist - pred_dist

    def revise_safe_actions(
        self,
        actions,
        avail_actions=None,
        guard_margin=None,
        guard_horizon=None,
    ):
        active_mask = self.get_active_agent_mask()
        for agent_id, agent in enumerate(self.env.agents):
            if active_mask[agent_id] > 0.0 and agent.reached:
                self.env._advance_intermediate_subgoal_if_needed(agent)
        noop_action = self.get_noop_action()
        revised = [
            int(action) if active_mask[idx] > 0.0 else noop_action
            for idx, action in enumerate(actions)
        ]
        guard_flags = np.zeros(self.n_agents, dtype=np.float32)

        if avail_actions is None:
            avail_actions = np.ones((self.n_agents, self.n_actions), dtype=np.float32)
        avail_actions = np.asarray(avail_actions, dtype=np.float32)

        old_margin = self.env.guard_prediction_margin
        old_horizon = self.env.guard_prediction_horizon
        if guard_margin is not None:
            self.env.guard_prediction_margin = float(max(0.0, guard_margin))
        horizon = (
            self.env.guard_prediction_horizon
            if guard_horizon is None
            else max(1, int(guard_horizon))
        )
        self.env.guard_prediction_horizon = horizon
        try:
            collision_flags = self.estimate_joint_collision_flags_horizon(
                revised,
                horizon=horizon,
                guard_margin=self.env.guard_prediction_margin,
            )
            if not np.any(collision_flags * active_mask > 0.0):
                return revised, guard_flags

            for agent_id in range(self.n_agents):
                if active_mask[agent_id] <= 0.0 or collision_flags[agent_id] <= 0.0:
                    continue

                candidate_ids = np.flatnonzero(avail_actions[agent_id] > 0.0)
                if candidate_ids.size == 0:
                    candidate_ids = np.array([noop_action], dtype=np.int64)

                joint_candidates = []
                for candidate_id in candidate_ids:
                    joint = list(revised)
                    joint[agent_id] = int(candidate_id)
                    joint_candidates.append(joint)

                risks = self.estimate_joint_short_risk_horizon_batch(
                    joint_candidates,
                    horizon=horizon,
                    guard_margin=self.env.guard_prediction_margin,
                )
                best_action = revised[agent_id]
                best_score = None
                for cand_idx, candidate_id in enumerate(candidate_ids):
                    joint = joint_candidates[cand_idx]
                    cand_flags = self.estimate_joint_collision_flags_horizon(
                        joint,
                        horizon=horizon,
                        guard_margin=self.env.guard_prediction_margin,
                    )
                    progress = self._candidate_goal_progress(agent_id, candidate_id)
                    candidate_changed = int(candidate_id != revised[agent_id])
                    score = (
                        int(cand_flags[agent_id] > 0.0),
                        int(np.sum(cand_flags * active_mask)),
                        float(risks[cand_idx, agent_id]),
                        -float(progress),
                        candidate_changed,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_action = int(candidate_id)

                if best_action != revised[agent_id]:
                    revised[agent_id] = best_action
                    guard_flags[agent_id] = 1.0
                    collision_flags = self.estimate_joint_collision_flags_horizon(
                        revised,
                        horizon=horizon,
                        guard_margin=self.env.guard_prediction_margin,
                    )

            return revised, guard_flags
        finally:
            self.env.guard_prediction_margin = old_margin
            self.env.guard_prediction_horizon = old_horizon

    def get_env_info(self):
        self.reset()
        obs = self.get_obs()
        state = self.get_state()
        return {
            "n_actions": self.n_actions,
            "n_agents": self.n_agents,
            "state_shape": int(state.shape[0]),
            "obs_shape": int(obs.shape[-1]),
            "episode_limit": self.episode_limit,
            "msg_shape": int(self.env.msg_shape),
            "high_level_n_actions": int(self.env.high_level_n_actions),
            "high_level_mode_n_actions": int(self.env.high_level_mode_n_actions),
            "high_level_obs_shape": int(self.env.get_high_level_obs().shape[-1]),
            "high_level_state_shape": int(self.env.get_high_level_state().shape[-1]),
            "low_task_shape": int(self.env.low_task_shape),
            "max_active_orders": int(self.env.max_active_orders),
            "charge_action_id": int(self.env.charge_action_id),
        }

    def get_obs(self):
        if self._last_obs is None:
            self.reset()
        return np.asarray(self._last_obs, dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_state(self):
        return self.env.get_state()

    def set_meta_period(self, meta_period):
        return self.env.set_meta_period(meta_period)

    def set_hrl_parameters(self, **kwargs):
        return self.env.set_hrl_parameters(**kwargs)

    def prepare_high_level_decision(self):
        return self.env.prepare_high_level_decision()

    def get_high_level_obs(self):
        return self.env.get_high_level_obs()

    def get_high_level_state(self):
        return self.env.get_high_level_state()

    def get_high_level_avail_actions(self):
        return self.env.get_high_level_avail_actions()

    def get_high_level_avail_agent_actions(self, agent_id):
        return self.env.get_high_level_avail_agent_actions(agent_id)

    def get_high_level_energy_margins(self):
        return self.env.get_high_level_energy_margins()

    def get_high_level_energy_order_masks(self):
        return self.env.get_high_level_energy_order_masks()

    def get_current_high_level_actions(self):
        return self.env.get_current_high_level_actions()

    def get_available_order_mask(self):
        return self.env.get_available_order_mask()

    def apply_high_level_actions(self, actions):
        applied = self.env.apply_high_level_actions(actions)
        self._last_obs = self.env.get_obs()
        return applied

    def get_agent_positions(self):
        return self.env.get_agent_positions()

    def get_current_subgoals(self):
        return self.env.get_current_subgoals()

    def get_subgoal_distances(self, targets=None):
        return self.env.get_subgoal_distances(targets)

    def get_subgoal_success_mask(self, targets=None):
        return self.env.get_subgoal_success_mask(targets)

    def compute_intrinsic_rewards(self, prev_distances=None, targets=None):
        return self.env.compute_intrinsic_rewards(prev_distances, targets)

    def relabel_observations_with_subgoals(self, observations, subgoals):
        return self.env.relabel_observations_with_subgoals(observations, subgoals)

    def get_msg(self):
        return self.env.get_msg()

    def get_avail_agent_actions(self, agent_id):
        if self.get_active_agent_mask()[agent_id] > 0.0:
            return np.ones(self.n_actions, dtype=np.float32)
        avail = np.zeros(self.n_actions, dtype=np.float32)
        avail[self.get_noop_action()] = 1.0
        return avail

    def get_active_agent_mask(self):
        return self.env.get_active_agent_mask()

    def summary(self):
        summary = self.env.summary()
        summary["episode_reward"] = self._last_reward
        return summary

    def close(self):
        self.env.close()


class UAVParallelEnv:
    metadata = {"name": "uav_delivery_env"}

    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=4,
        num_targets=1,
        max_cycles=200,
        continuous_actions=True,
        render_mode=None,
        reset_retry_limit=20,
        sample_retry_limit=100,
        obstacle_crash_penalty=10.0,
        total_orders=8,
        max_active_orders=4,
        pickup_reward=3.0,
        delivery_reward=8.0,
        initial_energy=default_initial_energy,
        energy_decay_per_step=None,
        energy_depletion_fraction=default_energy_depletion_fraction,
        charging_capacity=default_charging_capacity,
        charging_station_count=default_charging_station_count,
        charging_radius=default_charging_radius,
        charging_rate=None,
        charging_station_pos=None,
        auction_enabled=True,
        fixed_charge_threshold_enabled=False,
        fixed_charge_threshold=0.35,
        fixed_charge_release_threshold=0.65,
    ):
        self.render_mode = render_mode
        self.continuous_actions = continuous_actions
        self.max_cycles = int(max_cycles)
        self.possible_agents = [f"agent_{i}" for i in range(num_hunters)]
        self.max_num_agents = len(self.possible_agents)
        self.agents = list(self.possible_agents)
        self._agent_index = {
            agent_name: idx for idx, agent_name in enumerate(self.possible_agents)
        }
        self.unwrapped = self

        self.base_env = UAVEnv(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=self.max_cycles,
            reset_retry_limit=reset_retry_limit,
            sample_retry_limit=sample_retry_limit,
            obstacle_crash_penalty=obstacle_crash_penalty,
            total_orders=total_orders,
            max_active_orders=max_active_orders,
            pickup_reward=pickup_reward,
            delivery_reward=delivery_reward,
            initial_energy=initial_energy,
            energy_decay_per_step=energy_decay_per_step,
            energy_depletion_fraction=energy_depletion_fraction,
            charging_capacity=charging_capacity,
            charging_station_count=charging_station_count,
            charging_radius=charging_radius,
            charging_rate=charging_rate,
            charging_station_pos=charging_station_pos,
            auction_enabled=auction_enabled,
            fixed_charge_threshold_enabled=fixed_charge_threshold_enabled,
            fixed_charge_threshold=fixed_charge_threshold,
            fixed_charge_release_threshold=fixed_charge_release_threshold,
        )
        self.msg_shape = getattr(self.base_env, "msg_shape", 0)

        if continuous_actions:
            self.action_spaces = {
                agent_name: self.base_env.action_space[agent_name]
                for agent_name in self.possible_agents
            }
        else:
            discrete_actions = self._build_discrete_actions()
            self._discrete_actions = discrete_actions
            discrete_space = spaces.Discrete(len(discrete_actions))
            self.action_spaces = {
                agent_name: discrete_space
                for agent_name in self.possible_agents
            }
        self.observation_spaces = {
            agent_name: self.base_env.observation_space[agent_name]
            for agent_name in self.possible_agents
        }

    def _build_discrete_actions(self):
        action_values = [-1.0, 0.0, 1.0]
        discrete_actions = []
        for action in itertools.product(action_values, repeat=self.base_env.dim_actions):
            vec = np.array(action, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > eps:
                vec = vec / norm
            discrete_actions.append(vec * self.base_env.a_max)
        return discrete_actions

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _obs_array_to_dict(self, obs_array):
        return {
            agent_name: np.asarray(obs_array[idx], dtype=np.float32)
            for idx, agent_name in enumerate(self.possible_agents)
        }

    def _info_dict(self):
        summary = self.base_env.summary()
        return {
            agent_name: dict(summary)
            for agent_name in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        del options
        obs_array = self.base_env.reset(seed=seed)
        self.agents = list(self.possible_agents)
        return self._obs_array_to_dict(obs_array), self._info_dict()

    def _convert_action(self, agent_name, action):
        if self.continuous_actions:
            return np.asarray(action, dtype=np.float32)
        return np.asarray(
            self._discrete_actions[int(action)],
            dtype=np.float32,
        )

    def step(self, actions):
        active_mask = self.base_env.get_active_agent_mask()
        action_list = [
            (
                self._convert_action(agent_name, actions[agent_name])
                if active_mask[idx] > 0.0
                else np.zeros(self.base_env.dim_actions, dtype=np.float32)
            )
            for idx, agent_name in enumerate(self.possible_agents)
        ]
        obs_array, rewards, reached_flags, safe_value = self.base_env.step(action_list)
        collision_flags = [bool(agent.collided) for agent in self.base_env.agents]

        success_terminated = bool(all(reached_flags))
        collision_terminated = bool(any(collision_flags))
        post_step_active_mask = self.base_env.get_active_agent_mask()
        all_depleted = bool(np.sum(post_step_active_mask) <= 0.0)
        terminated = success_terminated or all_depleted
        truncated = bool(
            self.base_env.current_step >= self.max_cycles and not terminated
        )
        obs_dict = self._obs_array_to_dict(obs_array)
        reward_dict = {
            agent_name: float(rewards[idx])
            for idx, agent_name in enumerate(self.possible_agents)
        }
        done_dict = {
            agent_name: terminated
            for agent_name in self.possible_agents
        }
        trunc_dict = {
            agent_name: truncated
            for agent_name in self.possible_agents
        }
        info_dict = {
            agent_name: {
                "safe_value": float(safe_value[idx]),
                "reached": bool(reached_flags[idx]),
                "collided": bool(collision_flags[idx]),
                "active": bool(post_step_active_mask[idx]),
                "energy": float(self.base_env.agents[idx].energy),
                "collision_terminated": collision_terminated,
                "success_terminated": success_terminated,
                "all_depleted": all_depleted,
                **self.base_env.summary(),
            }
            for idx, agent_name in enumerate(self.possible_agents)
        }

        self.agents = [] if terminated or truncated else list(self.possible_agents)
        return obs_dict, reward_dict, done_dict, trunc_dict, info_dict

    def render(self, view=None):
        if self.render_mode == "human":
            self.base_env.render(show=True, view=view)
            return None
        return self.base_env.render(show=False, view=view)

    def get_msg(self):
        return self.base_env.get_msg()

    def get_high_level_obs(self):
        return self.base_env.get_high_level_obs()

    def get_high_level_state(self):
        return self.base_env.get_high_level_state()

    def set_meta_period(self, meta_period):
        return self.base_env.set_meta_period(meta_period)

    def set_hrl_parameters(self, **kwargs):
        return self.base_env.set_hrl_parameters(**kwargs)

    def prepare_high_level_decision(self):
        return self.base_env.prepare_high_level_decision()

    def get_high_level_avail_actions(self):
        return self.base_env.get_high_level_avail_actions()

    def get_high_level_avail_agent_actions(self, agent):
        agent_idx = self._agent_index[agent] if isinstance(agent, str) else int(agent)
        return self.base_env.get_high_level_avail_agent_actions(agent_idx)

    def get_high_level_energy_margins(self):
        return self.base_env.get_high_level_energy_margins()

    def get_high_level_energy_order_masks(self):
        return self.base_env.get_high_level_energy_order_masks()

    def get_current_high_level_actions(self):
        return self.base_env.get_current_high_level_actions()

    def get_available_order_mask(self):
        return self.base_env.get_available_order_mask()

    def apply_high_level_actions(self, actions):
        if isinstance(actions, dict):
            actions = [
                actions[agent_name]
                for agent_name in self.possible_agents
            ]
        return self.base_env.apply_high_level_actions(actions)

    def get_agent_positions(self):
        return self.base_env.get_agent_positions()

    def get_current_subgoals(self):
        return self.base_env.get_current_subgoals()

    def get_subgoal_distances(self, targets=None):
        return self.base_env.get_subgoal_distances(targets)

    def get_subgoal_success_mask(self, targets=None):
        return self.base_env.get_subgoal_success_mask(targets)

    def compute_intrinsic_rewards(self, prev_distances=None, targets=None):
        return self.base_env.compute_intrinsic_rewards(prev_distances, targets)

    def relabel_observations_with_subgoals(self, observations, subgoals):
        return self.base_env.relabel_observations_with_subgoals(observations, subgoals)

    def close(self):
        self.base_env.close()


def parallel_env(
    dim_actions=3,
    length=boundary_length,
    width=boundary_width,
    height=boundary_height,
    num_obstacle=default_num_obstacles,
    num_hunters=4,
    num_targets=1,
    max_cycles=200,
    continuous_actions=True,
    render_mode=None,
    reset_retry_limit=20,
    sample_retry_limit=100,
    obstacle_crash_penalty=10.0,
    total_orders=8,
    max_active_orders=4,
    pickup_reward=3.0,
    delivery_reward=8.0,
    initial_energy=default_initial_energy,
    energy_decay_per_step=None,
    energy_depletion_fraction=default_energy_depletion_fraction,
    charging_capacity=default_charging_capacity,
    charging_station_count=default_charging_station_count,
    charging_radius=default_charging_radius,
    charging_rate=None,
    charging_station_pos=None,
    auction_enabled=True,
    fixed_charge_threshold_enabled=False,
    fixed_charge_threshold=0.35,
    fixed_charge_release_threshold=0.65,
):
    return UAVParallelEnv(
        dim_actions=dim_actions,
        length=length,
        width=width,
        height=height,
        num_obstacle=num_obstacle,
        num_hunters=num_hunters,
        num_targets=num_targets,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        render_mode=render_mode,
        reset_retry_limit=reset_retry_limit,
        sample_retry_limit=sample_retry_limit,
        obstacle_crash_penalty=obstacle_crash_penalty,
        total_orders=total_orders,
        max_active_orders=max_active_orders,
        pickup_reward=pickup_reward,
        delivery_reward=delivery_reward,
        initial_energy=initial_energy,
        energy_decay_per_step=energy_decay_per_step,
        energy_depletion_fraction=energy_depletion_fraction,
        charging_capacity=charging_capacity,
        charging_station_count=charging_station_count,
        charging_radius=charging_radius,
        charging_rate=charging_rate,
        charging_station_pos=charging_station_pos,
        auction_enabled=auction_enabled,
        fixed_charge_threshold_enabled=fixed_charge_threshold_enabled,
        fixed_charge_threshold=fixed_charge_threshold,
        fixed_charge_release_threshold=fixed_charge_release_threshold,
    )
