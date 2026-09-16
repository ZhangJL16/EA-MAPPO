"""Frozen high-level interventions for the productive-stuckness kill test.

This module does not alter the navigation controller, collision handling, plant
physics, recharge mechanics, or keyed world generator.  M3 is deliberately an
experiment-only action outside the original Continue/Return action space.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import random
from typing import Any, Callable

import joblib
import numpy as np

from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.threshold_stress import ThresholdStress
from review_bundle.safety.switching.commitment import SortieMode


METHODS = {
    0: "M0_soc40",
    1: "M1_timeout4000_return",
    2: "M2_no_progress256_return",
    3: "M3_no_progress256_replan",
    4: "M4_completion_within_4000_oracle_return",
}
SOC_THRESHOLD = 0.40
TIMEOUT_STEPS = 4000
NO_PROGRESS_STEPS = 256
ORACLE_PROBE_STEPS = 4000


def snapshot_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_rng_fingerprint(env: "OracleStucknessEnv") -> str:
    """Canonical byte-level digest of complete wrapper state and global RNGs.

    Cloudpickle preserves the object graph but does not promise that serializing
    a deserialized graph emits the identical pickle stream.  joblib.hash is used
    as the deterministic content fingerprint; continuation identity is tested
    independently.
    """
    canonical = joblib.hash((env.__dict__, random.getstate(), np.random.get_state()),
                            hash_name="sha1")
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


class OracleStucknessEnv(ThresholdStress):
    """Continuing plant plus isolated probe and experiment-only replan."""

    def physical_world_record(self) -> dict[str, Any]:
        b = self.base
        return {
            "schema_version": "oracle-stuckness-uav-world-v1",
            "world_seed": int(self.job["world_seed"]),
            "num_obstacles": int(b.num_obstacles),
            "static_obstacles": b.static_obstacle_layout(),
            "charger_position": np.asarray(b.charger_position).tolist(),
            "initial_task_point": np.asarray(b.current_task_point).tolist(),
            "workspace": {
                "length": float(b.length),
                "width": float(b.width),
                "height": float(b.height),
            },
        }

    def task_key(self) -> tuple[int, int, tuple[float, float, float]]:
        b = self.base
        return (
            int(b.battery_cycle_id),
            int(b.tasks_in_current_battery_cycle),
            tuple(float(x) for x in np.asarray(b.current_task_point)),
        )

    def replan_current_task(self) -> dict[str, Any]:
        """Consume the next keyed task slot without teleporting or recharging."""
        b = self.base
        if b.mode is not SortieMode.TASK:
            raise RuntimeError("M3 replan is only legal in TASK mode")
        position = b.agent.pos.copy()
        energy = float(b.agent.energy)
        old_key = self.task_key()
        b._finalize_goal_trajectory(
            success=False, censored_reason="oracle_stuckness_experiment_replan"
        )
        b.task_stuck_count += 1
        # This is a schedule-slot counter inside the frozen keyed generator.
        # Global tasks_completed remains unchanged and is the throughput source.
        b.tasks_in_current_battery_cycle += 1
        b.agent.vel[:] = 0.0
        b.current_task_point = b._sample_task_point(b.agent.pos)
        b.steps_in_current_task = 0
        b.mode = SortieMode.TASK
        b.agent.goal = b.current_task_point.copy()
        b._start_goal_trajectory(b.current_task_point)
        if not np.array_equal(position, b.agent.pos) or float(b.agent.energy) != energy:
            raise RuntimeError("M3 illegally changed position or energy")
        return {"old_task_key": old_key, "new_task_key": self.task_key()}

    def completion_within_probe(
        self,
        action_fn: Callable[[dict[str, np.ndarray], bool], np.ndarray],
        *,
        horizon: int = ORACLE_PROBE_STEPS,
    ) -> dict[str, Any]:
        """Privileged snapshot--probe--restore diagnostic; never a perfect oracle."""
        if self.base.mode is not SortieMode.TASK:
            raise RuntimeError("oracle probe must start in TASK mode")
        before = self.snapshot()
        before_hash = snapshot_sha256(before)
        before_state_hash = state_rng_fingerprint(self)
        before_clock = float(self.base.simulation_time)
        before_energy = float(self.base.agent.energy)
        before_tasks = int(self.base.tasks_completed)
        start_tasks = before_tasks
        completed = False
        terminal_reason = "probe_horizon"
        steps = 0
        try:
            observation = self.observation()
            for _ in range(int(horizon)):
                nav = action_fn(observation, False)
                action = np.concatenate((np.asarray(nav, np.float32), [0.0])).astype(np.float32)
                observation, _, _, _, info = self.step(action)
                steps += 1
                if int(self.base.tasks_completed) > start_tasks:
                    completed = True
                    terminal_reason = "task_completed"
                    break
                if bool(info.get("exhausted", False)):
                    terminal_reason = "energy_exhausted"
                    break
        finally:
            self.restore(before)
        after = self.snapshot()
        after_hash = snapshot_sha256(after)
        after_state_hash = state_rng_fingerprint(self)
        if after_state_hash != before_state_hash:
            raise RuntimeError(
                "snapshot--probe--restore state/RNG identity failed: "
                f"{before_state_hash} != {after_state_hash}"
            )
        if (
            float(self.base.simulation_time) != before_clock
            or float(self.base.agent.energy) != before_energy
            or int(self.base.tasks_completed) != before_tasks
        ):
            raise RuntimeError("oracle probe leaked into formal simulator state")
        return {
            "completed_within_4000": bool(completed),
            "doomed": not completed,
            "probe_steps": int(steps),
            "probe_terminal": terminal_reason,
            "snapshot_sha256": before_hash,
            "restored_snapshot_sha256": after_hash,
            "state_rng_sha256": before_state_hash,
            "restored_state_rng_sha256": after_state_hash,
        }


@dataclass
class AttemptLedger:
    seconds: float = 0.0
    steps: int = 0


class FrozenIntervention:
    """State machine implementing exactly one of the frozen M0--M4 methods."""

    def __init__(self, method: int):
        if method not in METHODS:
            raise ValueError(f"unknown method: {method}")
        self.method = int(method)
        self.window: deque[tuple[tuple, float, int]] = deque(maxlen=NO_PROGRESS_STEPS + 1)
        self.last_task_key: tuple | None = None
        self.oracle_labels: dict[tuple, dict[str, Any]] = {}
        self.counts = {"timeout": 0, "no_progress_return": 0, "replan": 0,
                       "oracle_probe": 0, "oracle_doomed": 0, "oracle_return": 0}

    @staticmethod
    def distance_to_task(env: OracleStucknessEnv) -> float:
        return float(np.linalg.norm(env.base.current_task_point - env.base.agent.pos))

    def _refresh_window(self, env: OracleStucknessEnv) -> None:
        if env.base.mode is not SortieMode.TASK:
            self.window.clear()
            self.last_task_key = None
            return
        key = env.task_key()
        if key != self.last_task_key:
            self.window.clear()
            self.last_task_key = key
        self.window.append((key, self.distance_to_task(env), int(env.base.tasks_completed)))

    def no_progress_trigger(self, env: OracleStucknessEnv) -> bool:
        self._refresh_window(env)
        if len(self.window) != NO_PROGRESS_STEPS + 1:
            return False
        first, last = self.window[0], self.window[-1]
        same_task = all(row[0] == first[0] for row in self.window)
        no_completion = first[2] == last[2]
        net_progress = first[1] - last[1]
        proactive = float(env.base.agent.energy) > SOC_THRESHOLD * CAPACITY
        return bool(same_task and no_completion and net_progress <= 0.0 and proactive)

    def decide(
        self,
        env: OracleStucknessEnv,
        action_fn: Callable[[dict[str, np.ndarray], bool], np.ndarray],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return (commit_return, event). Replan occurs directly when registered."""
        b = env.base
        if b.mode is SortieMode.CHARGER_COMMITTED:
            self._refresh_window(env)
            return False, None
        key = env.task_key()
        if (
            self.method == 4
            and b.steps_in_current_task == 0
            and key not in self.oracle_labels
            and float(b.agent.energy) > SOC_THRESHOLD * CAPACITY
        ):
            label = env.completion_within_probe(action_fn)
            self.oracle_labels[key] = label
            self.counts["oracle_probe"] += 1
            self.counts["oracle_doomed"] += int(label["doomed"])
            if label["doomed"] and float(b.agent.energy) > SOC_THRESHOLD * CAPACITY:
                self.counts["oracle_return"] += 1
                return True, {"kind": "oracle_return", **label}
        if self.method == 1 and b.steps_in_current_task >= TIMEOUT_STEPS:
            self.counts["timeout"] += 1
            return True, {"kind": "timeout", "task_clock": int(b.steps_in_current_task)}
        if self.method in {2, 3} and self.no_progress_trigger(env):
            if self.method == 2:
                self.counts["no_progress_return"] += 1
                self.window.clear()
                return True, {"kind": "no_progress_return"}
            detail = env.replan_current_task()
            self.counts["replan"] += 1
            self.window.clear()
            self.last_task_key = env.task_key()
            return False, {"kind": "replan", **detail}
        if self.method not in {2, 3}:
            self._refresh_window(env)
        soc_return = float(b.agent.energy) <= SOC_THRESHOLD * CAPACITY
        return bool(soc_return), ({"kind": "soc40_return"} if soc_return else None)
