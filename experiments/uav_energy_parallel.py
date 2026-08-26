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


_LIGHT_INFO_KEYS = (
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
    "projection_geometry",
    "active_set_changed",
    "completed_goal_evaluation",
    "energy_exhausted",
)


@dataclass(frozen=True)
class WorkerReset:
    worker_id: int
    seed: int
    options: dict[str, object] | None = None


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
            elif command == "step":
                observation, reward, terminated, truncated, info = environment.step(
                    np.asarray(request["action"], dtype=np.float32)
                )
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
    ) -> dict[int, WorkerStep]:
        action_batch = np.asarray(actions, dtype=np.float32)
        if action_batch.shape != (len(worker_ids), 3):
            raise ValueError(
                f"actions must have shape ({len(worker_ids)}, 3), got {action_batch.shape}"
            )
        for worker_id, action in zip(worker_ids, action_batch, strict=True):
            self._connections[worker_id].send(("step", {"action": action}))
        return {
            worker_id: self._step_from_payload(
                worker_id,
                self._decode(worker_id, self._connections[worker_id].recv()),
            )
            for worker_id in worker_ids
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
