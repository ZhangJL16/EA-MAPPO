"""Fixed-horizon stall-rule evaluation on unlimited task/return clocks."""
from __future__ import annotations

import math
import numpy as np

from experiments.directional_navigation.threshold_stress import ThresholdStress, CAPACITY

STALL_WINDOW = 16
ACTION_EPSILON = 1e-6


def continuing_stall_jobs(worlds: int = 10) -> list[dict]:
    if not 1 <= worlds <= 10:
        raise ValueError("expected 1..10 registered worlds")
    return [dict(job_id=4*i+2*q+enabled, world_seed=1120000301+i,
                 head_seed=0, method=2*q+enabled,
                 method_name=f"soc{int(threshold*100)}" + ("_stall16" if enabled else ""),
                 threshold=threshold, stall_enabled=bool(enabled), obstacles=48, soc=1.)
            for i in range(worlds) for q, threshold in enumerate((.2,.4))
            for enabled in range(2)]


class ContinuingStallValidation(ThresholdStress):
    def __init__(self, jobs: list[dict], duration_seconds: float = 7200.):
        if len(jobs) != 1 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("one job and positive finite horizon required")
        self.duration_seconds = float(duration_seconds)
        super().__init__(jobs[0])

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.stall_streak = 0
        self.stall_return_count = 0
        self.proactive_stall_return_count = 0
        self.stall_episode_count = 0
        self.stall_steps = 0
        self.stall_trigger_steps = []
        self.cycles = []
        self.first_contact_step = None
        self.run_finished = False
        return obs, info

    def step(self, action):
        if self.run_finished:
            raise RuntimeError("evaluation already finished")
        was_task = self.base.mode.value == "TASK"
        eligible = was_task and self.stall_streak >= STALL_WINDOW
        stall_commit = bool(action[3] > 0 and eligible and self.job["stall_enabled"])
        proactive = stall_commit and self.observation()["battery"][0] > self.job["threshold"] * CAPACITY
        obs, reward, _, _, info = super().step(action)
        hit = bool(was_task and action[3] <= 0 and not info["task_gain"]
                   and not info["recharge_event"]
                   and np.linalg.norm(action[:3]) <= ACTION_EPSILON
                   and np.linalg.norm(info["executed_action"]) <= ACTION_EPSILON)
        self.stall_streak = self.stall_streak + 1 if hit else 0
        self.stall_steps += int(hit)
        self.stall_episode_count += int(self.stall_streak == STALL_WINDOW)
        if stall_commit:
            self.stall_return_count += 1
            self.proactive_stall_return_count += int(proactive)
            self.stall_trigger_steps.append(self.total_steps)
        if info["cost"] and self.first_contact_step is None:
            self.first_contact_step = self.total_steps
        if info["cycle"]:
            self.cycles.append(dict(info["cycle"], end_step=self.total_steps,
                                    end_seconds=self.base.simulation_time))
        horizon = self.base.simulation_time + 1e-9 >= self.duration_seconds
        self.run_finished = bool(self.exhausted or horizon)
        info.update(nominal_stall=hit, stall_streak=self.stall_streak, stall_return=stall_commit,
                    proactive_stall_return=proactive, simulation_seconds=self.base.simulation_time)
        if self.run_finished:
            info["persistent_run"] = dict(
                **self.job, termination="energy_exhausted" if self.exhausted else "evaluation_time",
                time_complete=horizon, evaluation_seconds=self.duration_seconds,
                simulation_seconds=self.base.simulation_time, steps=self.total_steps,
                tasks_completed=int(self.base.tasks_completed),
                tasks_per_simulated_hour=self.base.tasks_completed*3600/self.duration_seconds,
                collision_count=self.contacts, first_contact_step=self.first_contact_step,
                energy_exhausted=self.exhausted, censored=False,
                alive_at_time_limit=horizon and not self.exhausted,
                initial_energy=CAPACITY*self.job["soc"], remaining_energy=float(self.base.agent.energy),
                replenished_energy=self.replenished, energy_used=self.base.cumulative_virtual_energy,
                energy_balance_residual=self.diagnostic()["energy_balance_residual"],
                recharge_count=self.recharges, cycles=self.cycles, mode_at_end=self.base.mode.value,
                return_steps=self.return_steps, stall_steps=self.stall_steps,
                stall_episode_count=self.stall_episode_count, stall_return_count=self.stall_return_count,
                proactive_stall_return_count=self.proactive_stall_return_count,
                stall_trigger_steps=self.stall_trigger_steps)
        return obs, reward, self.exhausted, horizon and not self.exhausted, info
