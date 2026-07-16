import numpy as np

from envs.UAVEnergyDelivery import (
    UAVEnvDiscreteWrapper as BaseUAVEnvDiscreteWrapper,
    DeliveryOrder,
    UAVEnv,
    boundary_height,
    boundary_length,
    boundary_width,
    default_charging_capacity,
    default_charging_radius,
    default_charging_station_count,
    default_energy_depletion_fraction,
    default_initial_energy,
    default_num_obstacles,
    eps,
)


TASK_IDLE = 0
TASK_ORDER = 1
TASK_CHARGE = 2
HIGH_MODE_CHARGE = 0
HIGH_MODE_ORDER = 1


class HierarchicalUAVEnv(UAVEnv):
    """H-MAPPO adapter over the flat UAVEnergyDelivery environment.

    The base geometry, order sampling, charging, collision handling, energy
    accounting, and environment reward terms come from UAVEnergyDelivery.py.
    This class only adds high-level subgoal control and low-level intrinsic
    rewards needed by hierarchical training.
    """

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
        total_orders=16,
        max_active_orders=8,
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
        high_goal_style="line",
        high_mode_policy="hybrid",
        high_lateral_scale=0.35,
    ):
        super().__init__(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=episode_limit,
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
        )
        self.meta_period = 5
        self.high_goal_style = str(high_goal_style)
        self.high_mode_policy = str(high_mode_policy)
        self.high_lateral_scale = float(high_lateral_scale)
        self.charge_mode_fraction = float(np.clip(charge_mode_fraction, 0.01, 0.99))
        self.charge_mode_threshold = -1.0 + 2.0 * self.charge_mode_fraction
        self.charge_mode_center = -1.0 + self.charge_mode_fraction
        self.order_mode_center = self.charge_mode_fraction
        self.high_level_mode_n_actions = 2 if self.high_mode_policy == "hybrid" else 1
        self.high_level_n_actions = 3
        self.low_task_shape = 0

        self.reachable_subgoal_scale = 1.0
        self.intrinsic_reward_scale = 1.0
        self.intrinsic_success_bonus = 1.0
        self.intrinsic_collision_penalty = 0.0
        self.low_energy_budget_enabled = False
        self.low_energy_budget_min_ratio = 0.0
        self.low_energy_budget_max_ratio = 0.08
        self.low_energy_budget_overuse_coef = 2.0
        self.order_progress_override = None
        self.energy_margin_reserve_ratio = 0.05
        self.charge_energy_threshold = 0.35
        self.charge_release_threshold = 0.65
        self.charge_queue_enabled = False
        self.charge_queue_radius = 0.24
        self.min_subgoal_progress = 1.25 * self.goal_tolerance

        self._last_high_mode_train_mask = np.ones(
            (self.num_agents, 1), dtype=np.float32
        )
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_budget_overuse_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_reward_terms = {}
        self._init_hierarchical_agent_state()

    def _init_hierarchical_agent_state(self):
        for agent in self.agents:
            agent.assigned_order_slot = None
            agent.current_task_type = TASK_IDLE
            agent.task_target = agent.pos.copy()
            agent.subgoal = agent.pos.copy()
            agent.original_subgoal = agent.pos.copy()
            agent.high_energy_budget_ratio = 0.0
            agent.high_energy_budget_remaining = 0.0
            agent.high_energy_budget_steps_remaining = 0

    def set_meta_period(self, meta_period):
        self.meta_period = max(1, int(meta_period))

    def set_hrl_parameters(
        self,
        reachable_subgoal_scale=None,
        intrinsic_reward_scale=None,
        intrinsic_success_bonus=None,
        intrinsic_collision_penalty=None,
        low_energy_budget_enabled=None,
        low_energy_budget_min_ratio=None,
        low_energy_budget_max_ratio=None,
        low_energy_budget_overuse_coef=None,
        high_goal_style=None,
        high_lateral_scale=None,
        order_progress_override=None,
        energy_margin_reserve_ratio=None,
        charge_energy_threshold=None,
        charge_release_threshold=None,
        charge_queue_enabled=None,
        charge_queue_radius=None,
    ):
        if reachable_subgoal_scale is not None:
            self.reachable_subgoal_scale = float(max(0.0, reachable_subgoal_scale))
        if intrinsic_reward_scale is not None:
            self.intrinsic_reward_scale = float(intrinsic_reward_scale)
        if intrinsic_success_bonus is not None:
            self.intrinsic_success_bonus = float(intrinsic_success_bonus)
        if intrinsic_collision_penalty is not None:
            self.intrinsic_collision_penalty = float(intrinsic_collision_penalty)
        if low_energy_budget_enabled is not None:
            self.low_energy_budget_enabled = bool(low_energy_budget_enabled)
        if low_energy_budget_min_ratio is not None:
            self.low_energy_budget_min_ratio = float(
                np.clip(low_energy_budget_min_ratio, 0.0, 1.0)
            )
        if low_energy_budget_max_ratio is not None:
            self.low_energy_budget_max_ratio = float(
                np.clip(low_energy_budget_max_ratio, 0.0, 1.0)
            )
        if self.low_energy_budget_max_ratio < self.low_energy_budget_min_ratio:
            self.low_energy_budget_max_ratio = self.low_energy_budget_min_ratio
        if low_energy_budget_overuse_coef is not None:
            self.low_energy_budget_overuse_coef = float(
                max(0.0, low_energy_budget_overuse_coef)
            )
        if high_goal_style is not None:
            self.high_goal_style = str(high_goal_style)
        if high_lateral_scale is not None:
            self.high_lateral_scale = float(max(0.0, high_lateral_scale))
        if order_progress_override is not None:
            self.order_progress_override = float(
                np.clip(order_progress_override, -1.0, 1.0)
            )
        if energy_margin_reserve_ratio is not None:
            self.energy_margin_reserve_ratio = float(
                np.clip(energy_margin_reserve_ratio, 0.0, 1.0)
            )
        if charge_energy_threshold is not None:
            self.charge_energy_threshold = float(
                np.clip(charge_energy_threshold, 0.0, 1.0)
            )
        if charge_release_threshold is not None:
            self.charge_release_threshold = float(
                np.clip(charge_release_threshold, 0.0, 1.0)
            )
        if charge_queue_enabled is not None:
            self.charge_queue_enabled = bool(charge_queue_enabled)
        if charge_queue_radius is not None:
            self.charge_queue_radius = float(max(self.goal_tolerance, charge_queue_radius))

    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        self._init_hierarchical_agent_state()
        self._last_high_mode_train_mask = np.ones(
            (self.num_agents, 1), dtype=np.float32
        )
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_budget_overuse_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_reward_terms = {}
        return obs

    def _assign_orders(self):
        self._activate_orders()

    @property
    def order_slots(self):
        slots = list(self.active_order_ids[: self.max_active_orders])
        if len(slots) < self.max_active_orders:
            slots.extend([None for _ in range(self.max_active_orders - len(slots))])
        return slots

    @property
    def available_order_slots(self):
        slots = []
        for slot_idx, order_id in enumerate(self.order_slots):
            if order_id is None or order_id >= len(self.orders):
                continue
            order = self.orders[int(order_id)]
            if order.status == DeliveryOrder.ACTIVE and order.assigned_agent is None:
                slots.append(slot_idx)
        return slots

    def _slot_order(self, slot_idx):
        slots = self.order_slots
        if slot_idx is None or int(slot_idx) < 0 or int(slot_idx) >= len(slots):
            return None
        order_id = slots[int(slot_idx)]
        if order_id is None or int(order_id) >= len(self.orders):
            return None
        return self.orders[int(order_id)]

    def _max_reachable_subgoal_distance(self, agent):
        return (
            float(agent.v_max)
            * float(self.time_step)
            * float(max(1, self.meta_period))
            * float(self.reachable_subgoal_scale)
        )

    def _clip_position_to_bounds(self, pos):
        pos = np.asarray(pos, dtype=np.float32)[: self.dim_actions].copy()
        lower = np.full(self.dim_actions, self.safe_radius, dtype=np.float32)
        upper = self._space_scale() - self.safe_radius
        return np.clip(pos, lower, upper).astype(np.float32)

    def _set_agent_idle(self, agent):
        super()._set_agent_idle(agent)
        agent.assigned_order_slot = None
        agent.current_task_type = TASK_IDLE
        agent.task_target = agent.pos.copy()
        agent.subgoal = agent.pos.copy()
        agent.original_subgoal = agent.pos.copy()
        return agent.subgoal.copy()

    def _set_agent_subgoal(self, agent, target, task_type):
        target = self._clip_position_to_bounds(target)
        agent.goal = target.copy()
        agent.subgoal = target.copy()
        agent.original_subgoal = target.copy()
        agent.current_task_type = int(task_type)
        agent.reached = self._distance_to_goal(agent) <= self.goal_tolerance
        return target.copy()

    def _task_target_for_agent(self, agent):
        if agent.current_task_type == TASK_CHARGE:
            return self._nearest_charging_station_pos(agent.pos)
        if agent.assigned_order_id is None:
            return None
        return self._order_target(self.orders[agent.assigned_order_id])

    def _select_order_for_agent(self, agent):
        if agent.assigned_order_id is not None:
            return self.orders[agent.assigned_order_id]
        return self._nearest_available_order(agent)

    def _assign_order_slot_to_agent(self, agent, slot_idx):
        order = self._slot_order(slot_idx)
        if order is None or order.status != DeliveryOrder.ACTIVE:
            return False
        self._assign_order_to_agent(agent, order)
        agent.assigned_order_slot = int(slot_idx)
        agent.current_task_type = TASK_ORDER
        agent.task_target = order.pickup_pos.copy()
        agent.subgoal = agent.goal.copy()
        agent.original_subgoal = agent.goal.copy()
        return True

    def _charge_option_complete(self, agent):
        energy_ratio = agent.energy / (agent.initial_energy + eps)
        return (
            agent.current_task_type == TASK_CHARGE
            and np.linalg.norm(agent.pos - self._nearest_charging_station_pos(agent.pos))
            <= self.goal_tolerance
            and energy_ratio >= self.charge_release_threshold
        )

    def _set_agent_charging(self, agent):
        target = self._nearest_charging_station_pos(agent.pos)
        agent.assigned_order_slot = None
        if agent.assigned_order_id is not None and not agent.carrying_order:
            order = self.orders[agent.assigned_order_id]
            if order.status == DeliveryOrder.ASSIGNED:
                order.status = DeliveryOrder.ACTIVE
                order.assigned_agent = None
                agent.assigned_order_id = None
        agent.task_target = target.copy()
        self._set_agent_subgoal(
            agent,
            self._subgoal_on_line(agent, target, 1.0),
            TASK_CHARGE,
        )
        return True

    def _set_agent_energy_budget(self, agent, scalar):
        fraction = 0.5 * (float(np.clip(scalar, -1.0, 1.0)) + 1.0)
        budget_ratio = self.low_energy_budget_min_ratio + fraction * (
            self.low_energy_budget_max_ratio - self.low_energy_budget_min_ratio
        )
        agent.high_energy_budget_ratio = float(budget_ratio)
        agent.high_energy_budget_remaining = float(budget_ratio * agent.initial_energy)
        agent.high_energy_budget_steps_remaining = int(max(1, self.meta_period))

    def _subgoal_on_line(self, agent, target, progress_scalar):
        target = np.asarray(target, dtype=np.float32)[: self.dim_actions]
        to_target = target - agent.pos
        target_dist = float(np.linalg.norm(to_target))
        if target_dist <= eps:
            return agent.pos.copy()
        max_dist = self._max_reachable_subgoal_distance(agent)
        line_length = min(target_dist, max_dist)
        min_progress = min(line_length, self.min_subgoal_progress)
        fraction = 0.5 * (float(np.clip(progress_scalar, -1.0, 1.0)) + 1.0)
        if line_length > min_progress + eps:
            progress = min_progress + fraction * (line_length - min_progress)
        else:
            progress = line_length
        direction = to_target / (target_dist + eps)
        return self._clip_position_to_bounds(agent.pos + direction * progress)

    def _parse_high_level_action(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.high_level_mode_n_actions > 1:
            mode_id = int(np.clip(np.rint(action[0]), 0, self.high_level_mode_n_actions - 1))
            progress = float(action[1]) if action.size > 1 else 0.0
            budget = float(action[2]) if action.size > 2 else 0.0
        else:
            mode_scalar = float(action[0]) if action.size > 0 else 0.0
            mode_id = HIGH_MODE_CHARGE if mode_scalar < self.charge_mode_threshold else HIGH_MODE_ORDER
            progress = float(action[1]) if action.size > 1 else 0.0
            budget = float(action[2]) if action.size > 2 else 0.0
        return mode_id, progress, budget

    def prepare_high_level_decision(self):
        self._activate_orders()
        return {}

    def apply_high_level_actions(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, -1)
        if actions.shape[-1] < self.high_level_n_actions:
            pad_width = self.high_level_n_actions - actions.shape[-1]
            actions = np.pad(actions, ((0, 0), (0, pad_width))).astype(np.float32)
        actions = actions[:, : self.high_level_n_actions].reshape(
            self.num_agents, self.high_level_n_actions
        )
        self._activate_orders()
        mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)

        for agent_idx, action in enumerate(actions):
            agent = self.agents[agent_idx]
            if not agent.has_energy():
                self._set_agent_idle(agent)
                continue
            mode_id, progress_scalar, budget_scalar = self._parse_high_level_action(action)
            self._set_agent_energy_budget(agent, budget_scalar)

            if mode_id == HIGH_MODE_CHARGE:
                target = self._nearest_charging_station_pos(agent.pos)
                self._set_agent_charging(agent)
                self._set_agent_subgoal(agent, self._subgoal_on_line(agent, target, progress_scalar), TASK_CHARGE)
                mode_train_mask[agent_idx, 0] = 1.0
                continue

            order = self._select_order_for_agent(agent)
            if order is None:
                self._set_agent_idle(agent)
                mode_train_mask[agent_idx, 0] = 1.0
                continue
            if agent.assigned_order_id is None:
                self._assign_order_to_agent(agent, order)
            target = self._task_target_for_agent(agent)
            if target is None:
                self._set_agent_idle(agent)
                mode_train_mask[agent_idx, 0] = 1.0
                continue
            if (
                self.order_progress_override is not None
                and order.status in (DeliveryOrder.ASSIGNED, DeliveryOrder.PICKED)
            ):
                progress_scalar = self.order_progress_override
            agent.task_target = target.copy()
            self._set_agent_subgoal(
                agent,
                self._subgoal_on_line(agent, target, progress_scalar),
                TASK_ORDER,
            )
            mode_train_mask[agent_idx, 0] = 1.0

        self._last_high_mode_train_mask = mode_train_mask
        return actions.copy()

    def _agent_has_motion_task(self, agent):
        return agent.current_task_type in (TASK_ORDER, TASK_CHARGE)

    def _advance_order_if_reached(self, agent, current_dist=None):
        if (
            agent.current_task_type != TASK_ORDER
            or agent.assigned_order_id is None
        ):
            return 0.0

        order = self.orders[agent.assigned_order_id]
        if order.status == DeliveryOrder.ASSIGNED:
            if np.linalg.norm(agent.pos - order.pickup_pos) > self.goal_tolerance:
                return 0.0
            reward = self._mark_order_picked(agent, order, update_goal=False)
            agent.task_target = order.dropoff_pos.copy()
            agent.reached = True
            return reward

        if order.status == DeliveryOrder.PICKED:
            if np.linalg.norm(agent.pos - order.dropoff_pos) > self.goal_tolerance:
                return 0.0
            return self._mark_order_completed(agent, order)

        return 0.0

    def _consume_step_energy(self, powered_mask, actions=None):
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_budget_overuse_ratio = np.zeros(self.num_agents, dtype=np.float32)
        for agent_idx, (is_powered, agent) in enumerate(zip(powered_mask, self.agents)):
            if not is_powered:
                continue
            step_energy = float(self.energy_decay_per_step)
            self._last_step_energy_ratio[agent_idx] = step_energy / (
                agent.initial_energy + eps
            )
            if self.low_energy_budget_enabled:
                remaining = float(
                    max(0.0, getattr(agent, "high_energy_budget_remaining", 0.0))
                )
                steps_remaining = int(
                    max(1, getattr(agent, "high_energy_budget_steps_remaining", 1))
                )
                allowance = remaining / float(steps_remaining)
                self._last_budget_overuse_ratio[agent_idx] = max(
                    0.0, step_energy - allowance
                ) / (agent.initial_energy + eps)
                agent.high_energy_budget_remaining = max(0.0, remaining - step_energy)
                agent.high_energy_budget_steps_remaining = max(0, steps_remaining - 1)
            agent.consume_energy(step_energy)

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        self.current_step += 1
        self._activate_orders()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        reward_terms = {
            "progress": np.zeros(self.num_agents, dtype=np.float32),
            "velocity_toward_goal": np.zeros(self.num_agents, dtype=np.float32),
            "time": np.zeros(self.num_agents, dtype=np.float32),
            "obstacle_collision": np.zeros(self.num_agents, dtype=np.float32),
            "agent_collision": np.zeros(self.num_agents, dtype=np.float32),
            "pickup": np.zeros(self.num_agents, dtype=np.float32),
            "delivery": np.zeros(self.num_agents, dtype=np.float32),
            "all_orders_completed": np.zeros(self.num_agents, dtype=np.float32),
        }
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        powered_mask = np.asarray([agent.has_energy() for agent in self.agents], dtype=bool)
        prev_dists = np.array(
            [self._distance_to_goal(agent) for agent in self.agents], dtype=np.float32
        )
        order_status_before = []
        for agent in self.agents:
            if agent.assigned_order_id is None:
                order_status_before.append(None)
            else:
                order_status_before.append(self.orders[agent.assigned_order_id].status)

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
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
            self.reward_safe_value[idx] += boundary_reward_penalty + obstacle_reward_penalty

        agent_collisions = self._resolve_agent_collisions()

        for agent in self.agents:
            agent.pos = agent.prev_pos.copy()

        self._consume_step_energy(powered_mask, actions)
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

            progress_reward = 2.5 * progress
            velocity_reward = self.velocity_reward_weight * max(0.0, velocity_toward_goal)
            time_penalty = -0.01
            obstacle_collision_penalty = -obstacle_penalty * float(obstacle_collisions[idx])
            agent_collision_penalty = -agent_penalty * float(agent_collisions[idx])
            reward_terms["progress"][idx] = progress_reward
            reward_terms["velocity_toward_goal"][idx] = velocity_reward
            reward_terms["time"][idx] = time_penalty
            reward_terms["obstacle_collision"][idx] = obstacle_collision_penalty
            reward_terms["agent_collision"][idx] = agent_collision_penalty
            rewards[idx] += (
                progress_reward
                + velocity_reward
                + time_penalty
                + obstacle_collision_penalty
                + agent_collision_penalty
            )

            if current_dist <= self.goal_tolerance:
                agent.reached = True

            prev_status = order_status_before[idx]
            order_reward = self._advance_order_if_reached(agent, current_dist)
            if order_reward > 0.0:
                agent.vel[:] = 0.0
                if prev_status == DeliveryOrder.ASSIGNED:
                    reward_terms["pickup"][idx] = order_reward
                elif prev_status == DeliveryOrder.PICKED:
                    reward_terms["delivery"][idx] = order_reward
                rewards[idx] += order_reward

            agent.prev_collided = agent.collided
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            self.agent_paths[idx].append(agent.pos.copy())

        self._activate_orders()
        delivery_done = self._all_orders_completed()
        dones = [delivery_done for _ in self.agents]

        obstacle_collision_total = float(np.sum(np.asarray(obstacle_collisions, dtype=bool)))
        agent_collision_total = float(np.sum(np.asarray(agent_collisions, dtype=bool)))
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        if delivery_done:
            reward_terms["all_orders_completed"] += 5.0
            rewards += 5.0

        self._last_reward_terms = {
            name: values.astype(np.float32).copy() for name, values in reward_terms.items()
        }
        return self.get_obs(), rewards, dones, self.safe_value.copy()

    def get_obs(self):
        observations = super().get_obs()
        budget_features = []
        for agent in self.agents:
            budget_remaining = float(
                getattr(agent, "high_energy_budget_remaining", 0.0)
                / (agent.initial_energy + eps)
            )
            budget_steps = float(
                getattr(agent, "high_energy_budget_steps_remaining", 0)
                / float(max(1, self.meta_period))
            )
            budget_features.append(
                [
                    np.clip(budget_remaining, 0.0, 1.0),
                    np.clip(budget_steps, 0.0, 1.0),
                ]
            )
        return np.concatenate(
            [observations, np.asarray(budget_features, dtype=np.float32)], axis=-1
        ).astype(np.float32)

    def _high_level_agent_obs(self, agent):
        scale = self._space_scale() + eps
        base = np.concatenate([agent.pos / scale, agent.vel / (agent.v_max + eps)])
        energy = np.array([agent.energy / (agent.initial_energy + eps)], dtype=np.float32)
        task_onehot = np.zeros(3, dtype=np.float32)
        task_onehot[int(getattr(agent, "current_task_type", TASK_IDLE))] = 1.0
        station = self._nearest_charging_station_pos(agent.pos)
        station_delta = (station - agent.pos) / scale
        station_dist = np.array(
            [np.linalg.norm(station - agent.pos) / (np.linalg.norm(scale) + eps)],
            dtype=np.float32,
        )
        target = self._task_target_for_agent(agent)
        if target is None:
            order_delta = np.zeros(self.dim_actions, dtype=np.float32)
            order_dist = np.array([1.0], dtype=np.float32)
            has_order = np.array([0.0], dtype=np.float32)
        else:
            order_delta = (target - agent.pos) / scale
            order_dist = np.array(
                [np.linalg.norm(target - agent.pos) / (np.linalg.norm(scale) + eps)],
                dtype=np.float32,
            )
            has_order = np.array([1.0], dtype=np.float32)
        return np.concatenate(
            [
                base.astype(np.float32),
                energy,
                task_onehot,
                station_delta.astype(np.float32),
                station_dist,
                order_delta.astype(np.float32),
                order_dist,
                has_order,
            ]
        ).astype(np.float32)

    def get_high_level_obs(self):
        return np.stack(
            [self._high_level_agent_obs(agent) for agent in self.agents], axis=0
        ).astype(np.float32)

    def get_high_level_state(self):
        return np.concatenate(
            [
                self.get_state(),
                self.get_high_level_obs().reshape(-1),
            ]
        ).astype(np.float32)

    def get_high_level_avail_agent_actions(self, agent_id):
        agent = self.agents[int(agent_id)]
        avail = np.zeros(self.high_level_mode_n_actions, dtype=np.float32)
        if not agent.has_energy():
            return avail
        avail[:] = 1.0
        return avail

    def get_high_level_avail_actions(self):
        return np.stack(
            [self.get_high_level_avail_agent_actions(i) for i in range(self.num_agents)],
            axis=0,
        )

    def get_high_level_energy_margins(self):
        margins = []
        for agent in self.agents:
            order = self._select_order_for_agent(agent)
            if order is None:
                margins.append([0.0])
                continue
            target = order.dropoff_pos if order.status == DeliveryOrder.PICKED else order.pickup_pos
            dist = np.linalg.norm(agent.pos - target)
            dist += self._distance_to_nearest_charging_station_from_pos(target)
            est_steps = dist / (agent.v_max * self.time_step + eps)
            required = est_steps * self.energy_decay_per_step
            reserve = self.energy_margin_reserve_ratio * agent.initial_energy
            margins.append([(agent.energy - required - reserve) / (agent.initial_energy + eps)])
        return np.asarray(margins, dtype=np.float32)

    def get_high_level_energy_order_masks(self):
        return np.asarray(
            [[1.0 if self._select_order_for_agent(agent) is not None else 0.0] for agent in self.agents],
            dtype=np.float32,
        )

    def get_high_level_mode_training_mask(self):
        return np.asarray(self._last_high_mode_train_mask, dtype=np.float32).copy()

    def get_current_subgoals(self):
        return np.stack([agent.subgoal.copy() for agent in self.agents], axis=0).astype(
            np.float32
        )

    def get_current_task_targets(self):
        targets = []
        for agent in self.agents:
            target = getattr(agent, "task_target", None)
            if target is None:
                target = getattr(agent, "subgoal", agent.pos)
            targets.append(np.asarray(target, dtype=np.float32)[: self.dim_actions])
        return np.stack(targets, axis=0).astype(np.float32)

    def relabel_high_level_actions_with_achieved(
        self,
        start_positions,
        end_positions,
        actions,
        task_targets=None,
        active_mask=None,
    ):
        """Map achieved low-level endpoints back to equivalent high-level progress.

        This is the on-policy analogue of HAC hindsight action relabeling: keep
        the executed high-level mode/budget fixed, but train the continuous
        progress component using the position the low level actually reached.
        """
        actions = np.asarray(actions, dtype=np.float32).copy()
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, -1)
        if actions.shape[-1] < 2:
            return actions.astype(np.float32)

        start_positions = np.asarray(start_positions, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        end_positions = np.asarray(end_positions, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        if task_targets is None:
            task_targets = self.get_current_task_targets()
        task_targets = np.asarray(task_targets, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        if active_mask is None:
            active_values = np.ones(self.num_agents, dtype=np.float32)
        else:
            active_values = np.asarray(active_mask, dtype=np.float32).reshape(-1)

        for agent_idx, agent in enumerate(self.agents):
            if active_values[agent_idx] <= 0.0:
                continue
            start = start_positions[agent_idx]
            end = end_positions[agent_idx]
            target = task_targets[agent_idx]
            to_target = target - start
            target_dist = float(np.linalg.norm(to_target))
            if target_dist <= eps:
                continue

            max_dist = self._max_reachable_subgoal_distance(agent)
            line_length = min(target_dist, max_dist)
            if line_length <= eps:
                continue
            min_progress = min(line_length, self.min_subgoal_progress)
            direction = to_target / (target_dist + eps)
            achieved_progress = float(np.dot(end - start, direction))
            achieved_progress = float(np.clip(achieved_progress, 0.0, line_length))

            if line_length > min_progress + eps:
                fraction = (achieved_progress - min_progress) / (
                    line_length - min_progress
                )
                fraction = float(np.clip(fraction, 0.0, 1.0))
            else:
                fraction = 1.0
            actions[agent_idx, 1] = 2.0 * fraction - 1.0
        return actions.astype(np.float32)

    def get_subgoal_distances(self, targets=None):
        if targets is None:
            targets = self.get_current_subgoals()
        targets = np.asarray(targets, dtype=np.float32).reshape(
            self.num_agents, self.dim_actions
        )
        positions = np.stack([agent.pos.copy() for agent in self.agents], axis=0)
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
        rewards = progress
        rewards += self.intrinsic_success_bonus * self.get_subgoal_success_mask(targets)
        if self.intrinsic_collision_penalty > 0.0:
            collision_mask = np.asarray(
                [float(agent.collided) for agent in self.agents], dtype=np.float32
            )
            rewards -= self.intrinsic_collision_penalty * collision_mask
        if self.low_energy_budget_enabled and self.low_energy_budget_overuse_coef > 0.0:
            rewards -= (
                self.low_energy_budget_overuse_coef
                * np.asarray(self._last_budget_overuse_ratio, dtype=np.float32)
            )
        return (self.intrinsic_reward_scale * rewards).astype(np.float32)

    def relabel_observations_with_subgoals(self, observations, subgoals):
        return np.asarray(observations, dtype=np.float32).copy()


class UAVEnvDiscreteWrapper(BaseUAVEnvDiscreteWrapper):
    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=4,
        num_targets=1,
        episode_limit=200,
        total_orders=16,
        max_active_orders=8,
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
        high_goal_style="line",
        high_mode_policy="hybrid",
        high_lateral_scale=0.35,
        **unused_kwargs,
    ):
        self.env = HierarchicalUAVEnv(
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
            high_goal_style=high_goal_style,
            high_mode_policy=high_mode_policy,
            high_lateral_scale=high_lateral_scale,
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

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        info["per_agent_reward_terms"] = {
            name: np.asarray(values, dtype=np.float32).copy()
            for name, values in getattr(self.env, "_last_reward_terms", {}).items()
        }
        return reward, terminated, info

    def get_env_info(self):
        info = super().get_env_info()
        info.update(
            {
                "high_level_n_actions": int(self.env.high_level_n_actions),
                "high_level_mode_n_actions": int(self.env.high_level_mode_n_actions),
                "high_level_obs_shape": int(self.env.get_high_level_obs().shape[-1]),
                "high_level_state_shape": int(
                    self.env.get_high_level_state().shape[-1]
                ),
                "low_task_shape": int(self.env.low_task_shape),
                "max_active_orders": int(self.env.max_active_orders),
                "charge_action_id": -1,
            }
        )
        return info

    def apply_high_level_actions(self, actions):
        applied = self.env.apply_high_level_actions(actions)
        self._last_obs = self.env.get_obs()
        return applied

    def revise_safe_actions(
        self,
        actions,
        avail_actions=None,
        guard_margin=None,
        guard_horizon=None,
    ):
        del avail_actions, guard_margin, guard_horizon
        return list(actions), np.zeros(self.n_agents, dtype=np.float32)


def parallel_env(**kwargs):
    return UAVEnvDiscreteWrapper(**kwargs)
