"""Fresh-world paired C/R branches for the decision-history (Gate H) study.

This module uses the user-locked nonterminal collision recovery plant.  It does
not expose simulator geometry to the navigation actor or to a deployable
history representation.  Geometry is returned only by ``physical_world_record``
for blind allocation and ground-truth identity checks.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from envs.UAVEnergyDeliverySAC import SACTrainingPhase
from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.correction_supervision import CorrectionRecovery
from review_bundle.safety.switching.commitment import SortieMode


class DVOIHBranch(CorrectionRecovery):
    """One deterministic world with task-history generation and paired branches."""

    MAX_PREBRANCH_PREFIX_STEPS = 768

    def __init__(self, *, obstacles: int = 48, guard: int = 12_000):
        super().__init__(seed_start=1, seed_stride=1, horizon=4000, obstacles=obstacles, hocbf=True)
        self.guard = int(guard)
        self.base.set_phase(SACTrainingPhase.ENERGY_MANAGED)
        self.base.configure_calibrated_battery(
            CAPACITY, reserve_fraction=0.0, source="historical_synthetic_capacity"
        )
        self.base.reset_at_charger = True
        self.base.mission_switching_enabled = False
        self.base.energy_learning_enabled = False
        # Branch termination is registered relative to its anchor. Keep the
        # plant's task and absolute-episode administrative clocks strictly past
        # the latest anchor prefix plus the registered branch guard.
        administrative_limit = self.MAX_PREBRANCH_PREFIX_STEPS + self.guard + 1
        self.base.max_steps_per_task = administrative_limit
        self.base.phase2_episode_limit = administrative_limit
        original = self.observation_space
        self.observation_space = gym.spaces.Dict(
            {
                "nav": original,
                "return": original,
                "battery": gym.spaces.Box(0.0, np.inf, (1,), np.float32),
                "at_home": gym.spaces.Box(0.0, 1.0, (1,), np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self.world_seed = 0
        self.total_steps = 0
        self.return_steps = 0
        self.collision_count = 0
        self.branch_anchor_tasks = 0
        self.branch_anchor_contacts = 0
        self.branch_anchor_energy = 0.0
        self.branch_anchor_step = 0
        self.branch_mode = "HISTORY"

    def observation(self) -> dict[str, np.ndarray]:
        base = self.base
        returning = base.mode is SortieMode.CHARGER_COMMITTED
        nav_clock = self.return_steps if returning else base.steps_in_current_task
        nav = np.append(
            base.sac_observation_for_goal(base.active_goal),
            max(0.0, 1.0 - nav_clock / 4000.0),
        ).astype(np.float32)
        ret = np.append(base.charger_relative_observation(), 1.0).astype(np.float32)
        at_home = float(np.linalg.norm(base.agent.pos - base.charger_position) <= base.goal_tolerance)
        return {
            "nav": nav,
            "return": ret,
            "battery": np.array([base.agent.energy], np.float32),
            "at_home": np.array([at_home], np.float32),
        }

    def reset_world(self, world_seed: int, *, soc: float = 1.0) -> dict[str, np.ndarray]:
        if not 0.0 < soc <= 1.0:
            raise ValueError("soc must lie in (0,1]")
        self.world_seed = int(world_seed)
        self.base.bind_keyed_task_schedule(self.world_seed)
        self.base.reset(seed=self.world_seed)
        self.base.agent.energy = CAPACITY * float(soc)
        self.base.cycle_start_energy = self.base.agent.energy
        self.total_steps = 0
        self.return_steps = 0
        self.collision_count = 0
        self.branch_anchor_tasks = int(self.base.tasks_completed)
        self.branch_anchor_contacts = 0
        self.branch_anchor_energy = float(self.base.agent.energy)
        self.branch_anchor_step = 0
        self.branch_mode = "HISTORY"
        return self.observation()

    def begin_branch(self, mode: str) -> None:
        if mode not in {"C", "R"}:
            raise ValueError("branch mode must be C or R")
        self.branch_mode = mode
        self.branch_anchor_tasks = int(self.base.tasks_completed)
        self.branch_anchor_contacts = int(self.collision_count)
        self.branch_anchor_energy = float(self.base.agent.energy)
        self.branch_anchor_step = int(self.total_steps)
        self.return_steps = 0
        if mode == "R":
            self.commit_return()

    def commit_return(self) -> None:
        base = self.base
        if base.mode is SortieMode.CHARGER_COMMITTED:
            return
        base._finalize_goal_trajectory(success=False, censored_reason="dvoi_registered_return")
        base.mode = SortieMode.CHARGER_COMMITTED
        base.agent.goal = base.charger_position
        base.steps_in_current_task = 0
        self.return_steps = 0
        base._start_goal_trajectory(base.charger_position)

    def step_policy(self, action: np.ndarray) -> tuple[dict[str, np.ndarray], dict, bool]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("navigation action must be finite shape (3,)")
        returning_before = self.base.mode is SortieMode.CHARGER_COMMITTED
        _, _, terminated, truncated, info = self.base.step(action)
        self.total_steps += 1
        self.return_steps += int(returning_before)
        contact = bool(info["obstacle_collision"] or info["boundary_contact"])
        self.collision_count += int(contact)

        if (
            self.branch_mode == "C"
            and self.base.mode is SortieMode.TASK
            and int(self.base.tasks_completed) > self.branch_anchor_tasks
        ):
            self.commit_return()

        cycle = info.get("battery_cycle_record")
        returned = bool(cycle is not None and cycle.get("return_success", False))
        return_deadline = (
            self.base.mode is SortieMode.CHARGER_COMMITTED
            and self.return_steps >= 4000
            and not returned
        )
        branch_steps = self.total_steps - self.branch_anchor_step
        branch_guard = self.branch_mode in {"C", "R"} and branch_steps >= self.guard
        done = bool(terminated or truncated or returned or return_deadline or branch_guard)
        if returned:
            termination = "returned"
        elif return_deadline:
            termination = "return_deadline"
        elif branch_guard:
            termination = "branch_guard"
        elif terminated:
            termination = str(info.get("end_reason") or "terminated")
        elif truncated:
            termination = str(info.get("end_reason") or "truncated")
        else:
            termination = "running"
        light = {
            "contact": contact,
            "collision_count": int(self.collision_count),
            "nominal_action": np.asarray(info["nominal_action"], np.float32),
            "executed_action": np.asarray(info["executed_action"], np.float32),
            "realized_acceleration": np.asarray(info["realized_acceleration"], np.float32),
            "hocbf_intervened": bool(info["hocbf_intervened"]),
            "termination": termination,
            "returned": returned,
        }
        return self.observation(), light, done

    def branch_outcome(self, terminal: dict) -> dict[str, object]:
        collisions = int(self.collision_count - self.branch_anchor_contacts)
        task_increment = int(self.base.tasks_completed) - self.branch_anchor_tasks
        returned = bool(terminal.get("returned", False))
        collision_free_arrival = bool(returned and collisions == 0)
        operational_failure = not returned
        utility = float(task_increment - 2.0 * operational_failure - 0.25 * (collisions > 0))
        return {
            "termination": str(terminal["termination"]),
            "returned": returned,
            "collision_count": collisions,
            "collision_free_arrival": collision_free_arrival,
            "task_increment": task_increment,
            "initial_energy": float(self.branch_anchor_energy),
            "remaining_energy": float(self.base.agent.energy),
            "operational_failure": operational_failure,
            "utility": utility,
        }

    def physical_world_record(self) -> dict[str, object]:
        """Privileged audit identity; never an actor/estimator input."""
        base = self.base
        return {
            "schema_version": "dvoi-uav-physical-world-v1",
            "world_seed": self.world_seed,
            "num_obstacles": int(base.num_obstacles),
            "static_obstacles": base.static_obstacle_layout(),
            "charger_position": np.asarray(base.charger_position).tolist(),
            "initial_task_point": np.asarray(base.current_task_point).tolist(),
            "workspace": {
                "length": float(base.length),
                "width": float(base.width),
                "height": float(base.height),
            },
        }
