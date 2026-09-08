from __future__ import annotations

import multiprocessing as mp
import os
import traceback
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Iterable

import numpy as np

from envs.UAVEnergyDeliverySAC import (
    GoalConditionedQuantileTDEnergyEstimator,
    UAVEnergyDeliverySACEnv,
)
from experiments.forked_action_safety.core import (
    sanitize_snapshot_position,
)


_LIGHT_INFO_KEYS = (
    "progress",
    "realized_energy_cost",
    "transition_dt",
    "physics_substeps",
    "is_success",
    "end_reason",
    "boundary_contact",
    "obstacle_collision",
    "hocbf_intervened",
    "hocbf_emergency_brake",
    "hocbf_intervention_norm",
    "nominal_action",
    "executed_action",
    "distance_to_goal_before",
    "projection_geometry",
    "active_set_changed",
    "completed_goal_evaluation",
    "energy_exhausted",
    "continuous_battery_validation",
    "battery_validation_task_rollover",
    "battery_validation_task_rollover_count",
    "battery_validation_episode_guard_exceeded",
)


@dataclass(frozen=True)
class WorkerReset:
    worker_id: int
    seed: int
    options: dict[str, object] | None = None


@dataclass(frozen=True)
class WorkerRetarget:
    """Reset one worker at a new goal while preserving its obstacle layout."""

    worker_id: int
    seed: int
    start_position: np.ndarray
    start_velocity: np.ndarray
    goal_position: np.ndarray


@dataclass(frozen=True)
class WorkerInPlaceRetarget:
    """Change the goal/horizon without reconstructing the physical state."""

    worker_id: int
    goal_position: np.ndarray


@dataclass(frozen=True)
class WorkerForkReset:
    """Restore a pre-action anchor in a locked static scene.

    ``validation_start_position`` is a known legal reset position used only to
    validate/reconstruct the obstacle layout.  The worker then moves the agent
    to the already observed anchor without resampling the scene.
    """

    worker_id: int
    seed: int
    validation_start_position: np.ndarray
    anchor_position: np.ndarray
    anchor_velocity: np.ndarray
    goal_position: np.ndarray
    static_obstacles: list[dict[str, object]]
    elapsed_policy_steps: int


@dataclass(frozen=True)
class WorkerStep:
    worker_id: int
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]
    position: np.ndarray
    velocity: np.ndarray
    current_step: int
    simulation_time: float
    tasks_completed: int
    current_goal_path_length: float


def _payload(
    environment: UAVEnergyDeliverySACEnv,
    observation: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, object],
) -> dict[str, object]:
    return {
        "observation": np.asarray(observation, dtype=np.float32),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "info": {key: info.get(key) for key in _LIGHT_INFO_KEYS},
        "position": np.asarray(environment.agent.pos, dtype=np.float32),
        "velocity": np.asarray(environment.agent.vel, dtype=np.float32),
        "current_step": int(environment.current_step),
        "simulation_time": float(environment.simulation_time),
        "tasks_completed": int(environment.tasks_completed),
        "current_goal_path_length": float(environment._current_goal_path_length),
    }


def _worker_main(
    connection: Connection,
    environment_kwargs: dict[str, object],
    battery_capacity: float | None,
    battery_validation: bool,
    energy_estimator_checkpoint: str | None = None,
) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    try:
        environment = UAVEnergyDeliverySACEnv(**environment_kwargs)
        if energy_estimator_checkpoint is not None:
            estimator = GoalConditionedQuantileTDEnergyEstimator.load(
                energy_estimator_checkpoint,
                device="cpu",
            )
            environment.bind_energy_learning(
                energy_estimator=estimator,
                goal_action_provider=lambda _observation: np.zeros(
                    3,
                    dtype=np.float32,
                ),
                training_enabled=False,
            )
        if battery_capacity is not None:
            environment.configure_calibrated_battery(
                battery_capacity,
                reserve_fraction=float(environment_kwargs["energy_reserve_fraction"]),
                source="phase1_frozen_policy_calibration",
            )
        if battery_validation:
            environment.enable_battery_validation()
        while True:
            command, request = connection.recv()
            if command == "reset":
                observation, info = environment.reset(
                    seed=int(request["seed"]),
                    options=request.get("options"),
                )
                connection.send(
                    (
                        "ok",
                        _payload(environment, observation, 0.0, False, False, info),
                    )
                )
            elif command == "retarget":
                obstacle_layout = environment.static_obstacle_layout()
                observation, info = environment.reset(
                    seed=int(request["seed"]),
                    options={
                        "start_position": request["start_position"],
                        "start_velocity": request["start_velocity"],
                        "task_point": request["goal_position"],
                        "static_obstacles": obstacle_layout,
                    },
                )
                connection.send(
                    (
                        "ok",
                        _payload(environment, observation, 0.0, False, False, info),
                    )
                )
            elif command == "retarget_in_place":
                goal = environment._validate_task_point(
                    np.asarray(request["goal_position"], dtype=np.float32),
                    environment.agent.pos,
                )
                # This is a new control leg, not a new physical episode.  Keep
                # position and velocity bitwise unchanged while resetting only
                # the goal-relative horizon and trajectory bookkeeping.
                environment.current_task_point = goal.copy()
                environment.current_step = 0
                environment.steps_in_current_task = 0
                environment.simulation_time = 0.0
                environment.agent.goal = goal.copy()
                environment.agent.reached = False
                environment.agent_paths = [[environment.agent.pos.copy()]]
                environment._start_goal_trajectory(goal)
                environment._update_lidar()
                observation = environment._active_goal_sac_observation()
                connection.send(
                    (
                        "ok",
                        _payload(
                            environment,
                            observation,
                            0.0,
                            False,
                            False,
                            environment._info(),
                        ),
                    )
                )
            elif command == "fork_reset":
                observation, info = environment.reset(
                    seed=int(request["seed"]),
                    options={
                        "start_position": request["validation_start_position"],
                        "start_velocity": np.zeros(3, dtype=np.float32),
                        "task_point": request["goal_position"],
                        "static_obstacles": request["static_obstacles"],
                    },
                )
                del observation
                anchor_position = environment._validate_position(
                    sanitize_snapshot_position(
                        request["anchor_position"],
                        world_extent=np.asarray(
                            [environment.length, environment.width, environment.height],
                            dtype=np.float32,
                        ),
                        safe_radius=environment.safe_radius,
                    ),
                    "fork_anchor_position",
                    check_obstacles=False,
                )
                # Anchor preparation (or the unstarted-anchor migration path)
                # is the sole canonicalization point.  A fork must restore the
                # registered, hash-bound float32 snapshot without transforming
                # it again; validation still rejects a genuinely invalid state.
                anchor_velocity = environment._validate_velocity(
                    np.asarray(request["anchor_velocity"], dtype=np.float32)
                )
                environment.agent.pos = anchor_position.copy()
                environment.agent.prev_pos = anchor_position.copy()
                environment.agent.last_pos = anchor_position.copy()
                environment.agent.spawn_pos = anchor_position.copy()
                environment.agent.vel = anchor_velocity.copy()
                environment.agent.goal = environment.current_task_point.copy()
                environment.agent_paths = [[anchor_position.copy()]]
                elapsed = int(request["elapsed_policy_steps"])
                if not 0 <= elapsed < environment.max_steps_per_task:
                    raise ValueError("fork elapsed steps lie outside the episode")
                environment.current_step = elapsed
                environment.steps_in_current_task = elapsed
                environment.simulation_time = elapsed * environment.policy_dt
                environment._current_goal_initial_distance = float(
                    np.linalg.norm(environment.active_goal - anchor_position)
                )
                environment._current_goal_path_length = 0.0
                environment._update_lidar()
                observation = environment._active_goal_sac_observation()
                connection.send(
                    (
                        "ok",
                        _payload(environment, observation, 0.0, False, False, info),
                    )
                )
            elif command == "step":
                previous_cbf_enabled = bool(environment.cbf_enabled)
                if bool(request.get("disable_cbf", False)):
                    environment.cbf_enabled = False
                try:
                    observation, reward, terminated, truncated, info = environment.step(
                        np.asarray(request["action"], dtype=np.float32)
                    )
                finally:
                    environment.cbf_enabled = previous_cbf_enabled
                connection.send(
                    (
                        "ok",
                        _payload(
                            environment,
                            observation,
                            reward,
                            terminated,
                            truncated,
                            info,
                        ),
                    )
                )
            elif command == "close":
                environment.close()
                connection.send(("ok", None))
                break
            else:
                raise ValueError(f"unsupported worker command: {command}")
    except BaseException as error:
        connection.send(
            (
                "error",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
        )
    finally:
        connection.close()


class ParallelUAVEnvPool:
    def __init__(
        self,
        environment_kwargs: dict[str, object],
        *,
        num_workers: int,
        battery_capacity: float | None = None,
        battery_validation: bool = False,
        energy_estimator_checkpoint: str | None = None,
    ) -> None:
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")
        context = mp.get_context("spawn")
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self.closed = False
        for worker_id in range(int(num_workers)):
            parent, child = context.Pipe()
            process = context.Process(
                target=_worker_main,
                args=(
                    child,
                    environment_kwargs,
                    battery_capacity,
                    battery_validation,
                    energy_estimator_checkpoint,
                ),
                name=f"uav-eval-worker-{worker_id}",
                daemon=True,
            )
            process.start()
            child.close()
            self._connections.append(parent)
            self._processes.append(process)

    @property
    def num_workers(self) -> int:
        return len(self._connections)

    @staticmethod
    def _decode(worker_id: int, response) -> dict[str, object]:
        status, payload = response
        if status != "ok":
            raise RuntimeError(
                f"parallel UAV worker {worker_id} failed: "
                f"{payload['error_type']}: {payload['error']}\n{payload['traceback']}"
            )
        return payload

    def reset_many(self, requests: Iterable[WorkerReset]) -> dict[int, WorkerStep]:
        selected = list(requests)
        for request in selected:
            self._connections[request.worker_id].send(
                (
                    "reset",
                    {"seed": request.seed, "options": request.options},
                )
            )
        return {
            request.worker_id: self._step_from_payload(
                request.worker_id,
                self._decode(
                    request.worker_id,
                    self._connections[request.worker_id].recv(),
                ),
            )
            for request in selected
        }

    def step_many(
        self,
        worker_ids: list[int],
        actions: np.ndarray,
        *,
        disable_cbf: bool | np.ndarray = False,
    ) -> dict[int, WorkerStep]:
        action_batch = np.asarray(actions, dtype=np.float32)
        if action_batch.shape != (len(worker_ids), 3):
            raise ValueError(
                f"actions must have shape ({len(worker_ids)}, 3), got {action_batch.shape}"
            )
        raw_flags = np.asarray(disable_cbf, dtype=np.bool_)
        if raw_flags.ndim == 0:
            raw_flags = np.full(len(worker_ids), bool(raw_flags), dtype=np.bool_)
        if raw_flags.shape != (len(worker_ids),):
            raise ValueError("disable_cbf must be scalar or one flag per worker")
        for worker_id, action, raw in zip(
            worker_ids, action_batch, raw_flags, strict=True
        ):
            self._connections[worker_id].send(
                ("step", {"action": action, "disable_cbf": bool(raw)})
            )
        return {
            worker_id: self._step_from_payload(
                worker_id,
                self._decode(worker_id, self._connections[worker_id].recv()),
            )
            for worker_id in worker_ids
        }

    def fork_reset_many(
        self,
        requests: Iterable[WorkerForkReset],
    ) -> dict[int, WorkerStep]:
        selected = list(requests)
        for request in selected:
            self._connections[request.worker_id].send(
                (
                    "fork_reset",
                    {
                        "seed": int(request.seed),
                        "validation_start_position": np.asarray(
                            request.validation_start_position, dtype=np.float32
                        ),
                        "anchor_position": np.asarray(
                            request.anchor_position, dtype=np.float32
                        ),
                        "anchor_velocity": np.asarray(
                            request.anchor_velocity, dtype=np.float32
                        ),
                        "goal_position": np.asarray(
                            request.goal_position, dtype=np.float32
                        ),
                        "static_obstacles": list(request.static_obstacles),
                        "elapsed_policy_steps": int(request.elapsed_policy_steps),
                    },
                )
            )
        return {
            request.worker_id: self._step_from_payload(
                request.worker_id,
                self._decode(
                    request.worker_id,
                    self._connections[request.worker_id].recv(),
                ),
            )
            for request in selected
        }

    def retarget_many(
        self,
        requests: Iterable[WorkerRetarget],
    ) -> dict[int, WorkerStep]:
        """Start a second leg in the identical physical scene.

        This is intentionally distinct from ``reset_many``: the worker captures
        the already sampled obstacle layout before resetting, so task and return
        legs form one physically matched counterfactual mission.
        """

        selected = list(requests)
        for request in selected:
            self._connections[request.worker_id].send(
                (
                    "retarget",
                    {
                        "seed": int(request.seed),
                        "start_position": np.asarray(
                            request.start_position, dtype=np.float32
                        ),
                        "start_velocity": np.asarray(
                            request.start_velocity, dtype=np.float32
                        ),
                        "goal_position": np.asarray(
                            request.goal_position, dtype=np.float32
                        ),
                    },
                )
            )
        return {
            request.worker_id: self._step_from_payload(
                request.worker_id,
                self._decode(
                    request.worker_id,
                    self._connections[request.worker_id].recv(),
                ),
            )
            for request in selected
        }

    def retarget_in_place_many(
        self,
        requests: Iterable[WorkerInPlaceRetarget],
    ) -> dict[int, WorkerStep]:
        """Retarget workers while preserving their current physical states."""

        selected = list(requests)
        for request in selected:
            self._connections[request.worker_id].send(
                (
                    "retarget_in_place",
                    {
                        "goal_position": np.asarray(
                            request.goal_position, dtype=np.float32
                        ),
                    },
                )
            )
        return {
            request.worker_id: self._step_from_payload(
                request.worker_id,
                self._decode(
                    request.worker_id,
                    self._connections[request.worker_id].recv(),
                ),
            )
            for request in selected
        }

    @staticmethod
    def _step_from_payload(worker_id: int, payload: dict[str, object]) -> WorkerStep:
        return WorkerStep(
            worker_id=worker_id,
            observation=np.asarray(payload["observation"], dtype=np.float32),
            reward=float(payload["reward"]),
            terminated=bool(payload["terminated"]),
            truncated=bool(payload["truncated"]),
            info=dict(payload["info"]),
            position=np.asarray(payload["position"], dtype=np.float32),
            velocity=np.asarray(payload["velocity"], dtype=np.float32),
            current_step=int(payload["current_step"]),
            simulation_time=float(payload["simulation_time"]),
            tasks_completed=int(payload["tasks_completed"]),
            current_goal_path_length=float(payload["current_goal_path_length"]),
        )

    def close(self) -> None:
        if self.closed:
            return
        for connection, process in zip(
            self._connections,
            self._processes,
            strict=True,
        ):
            if process.is_alive():
                try:
                    connection.send(("close", {}))
                except (BrokenPipeError, EOFError):
                    pass
        for worker_id, (connection, process) in enumerate(
            zip(self._connections, self._processes, strict=True)
        ):
            if process.is_alive():
                try:
                    self._decode(worker_id, connection.recv())
                except (EOFError, BrokenPipeError, ConnectionResetError):
                    pass
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            connection.close()
        self.closed = True

    def __enter__(self) -> "ParallelUAVEnvPool":
        return self

    def __exit__(self, *_args) -> None:
        self.close()
