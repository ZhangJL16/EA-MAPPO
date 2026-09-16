"""Finite-time prospective evaluation of a legal nominal-stall return signal."""
from __future__ import annotations

import numpy as np

from experiments.directional_navigation.persistent_recharge import PersistentRecharge
from review_bundle.safety.switching.commitment import SortieMode

STALL_WINDOW = 16
ACTION_EPSILON = 1e-6


def prospective_stall_jobs(worlds: int = 10) -> list[dict]:
    if not 1 <= worlds <= 10:
        raise ValueError("use 1..10 preregistered independent worlds")
    return [dict(job_id=2 * index + method, world_seed=1120000101 + index,
                 head_seed=0, method=method,
                 method_name=("soc40", "soc40_stall16")[method], soc=1.0)
            for index in range(worlds) for method in range(2)]


class StallValidation(PersistentRecharge):
    """Persistent run with a counter from legal policy/filter outputs only."""

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        self.stall_streak = 0
        self.stall_return_count = 0
        self.stall_trigger_steps: list[int] = []
        return observation, info

    def step(self, action):
        was_task = self.base.mode is SortieMode.TASK
        committed_by_stall = bool(action[3] > 0 and was_task and self.stall_streak >= STALL_WINDOW)
        observation, reward, terminated, truncated, info = super().step(action)
        telemetry = self.base.trajectory_log[-1]
        nominal = np.asarray(action[:3], dtype=np.float32)
        executed = np.asarray(telemetry["executed_action"], dtype=np.float32)
        is_stall = bool(was_task and action[3] <= 0
                        and np.linalg.norm(nominal) <= ACTION_EPSILON
                        and np.linalg.norm(executed) <= ACTION_EPSILON)
        self.stall_streak = self.stall_streak + 1 if is_stall else 0
        if committed_by_stall:
            self.stall_return_count += 1
            self.stall_trigger_steps.append(self.total_steps)
        info.update(executed_action=executed.copy(), nominal_stall=is_stall,
                    stall_streak=self.stall_streak, stall_return=committed_by_stall)
        if "persistent_run" in info:
            info["persistent_run"].update(stall_return_count=self.stall_return_count,
                                          stall_trigger_steps=self.stall_trigger_steps)
        return observation, reward, terminated, truncated, info
