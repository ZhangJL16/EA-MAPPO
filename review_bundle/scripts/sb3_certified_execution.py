from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from scripts.sb3_recovery_teacher import CertifiedRecoveryOracle


class CertifiedExecutionTrainingEnv(gym.Wrapper):
    """Clean-observation energy task with online verifier/κ execution authority."""

    def __init__(
        self,
        environment,
        *,
        scenario: str,
        flight_energy_multiplier: float,
        certificate_artifact: Path,
        training_seed: int,
    ) -> None:
        super().__init__(environment)
        self.oracle = CertifiedRecoveryOracle(
            scenario,
            flight_energy_multiplier,
            certificate_artifact,
        )
        self.training_seed = int(training_seed)
        self._episode_index = 0
        self._recovery_active = False
        self._kappa_takeover_count = 0
        self._kappa_execution_steps = 0
        self._charger_hold_steps = 0
        self._verified_departure_steps = 0
        self._policy_execution_steps = 0

    @property
    def actor_environment(self):
        return self.env.unwrapped

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        self.oracle.reset()
        effective_seed = self.training_seed + self._episode_index if seed is None else int(seed)
        certified_start = self.oracle.atlas.sample_initial_state(
            effective_seed,
            self.actor_environment.energy_navigation_config.battery_capacity,
        )
        reset_options = {
            "start_position": certified_start.position,
            "start_velocity": certified_start.velocity,
            "initial_energy_fraction": certified_start.energy
            / self.actor_environment.energy_navigation_config.battery_capacity,
        }
        if options is not None:
            reset_options.update(options)
        observation, info = self.env.reset(seed=effective_seed, options=reset_options)
        _, context = self.oracle.context(self.actor_environment, recovery=False)
        if not context.recovery.certified:
            raise RuntimeError("CERTIFIED_EXECUTION_RESET_OUTSIDE_RECOVERABLE_DOMAIN")
        self._episode_index += 1
        self._recovery_active = False
        self._kappa_takeover_count = 0
        self._kappa_execution_steps = 0
        self._charger_hold_steps = 0
        self._verified_departure_steps = 0
        self._policy_execution_steps = 0
        return observation, info | {
            "certified_execution": True,
            "certified_initial_state": True,
            "certificate_energy_version": self.oracle.certificate["energy_version"],
            "certificate_atlas_hash": self.oracle.certificate["recovery_atlas_hash"],
        }

    def _select_action(self, proposed_action: np.ndarray):
        actor_environment = self.actor_environment
        at_charger = actor_environment._charging_admissible(
            actor_environment.state.position,
            actor_environment.state.velocity,
        )
        recovery_context = None
        takeover_started = False
        if at_charger:
            departure_allowed = self.oracle.departure_allowed(actor_environment)
            mapped = self.oracle.certified_support_action(
                actor_environment,
                proposed_action,
                charging_state=True,
                charging_support_required=not departure_allowed,
            )
            if mapped is not None:
                executed_action, recovery_context = mapped
                authority = (
                    "POLICY_CERTIFIED_DEPARTURE_SUPPORT"
                    if departure_allowed
                    else "POLICY_CERTIFIED_CHARGING_SUPPORT"
                )
                return executed_action, authority, recovery_context, takeover_started
            return self.oracle.hold_action(actor_environment), "CERTIFIED_CHARGER_HOLD", recovery_context, takeover_started
        if self._recovery_active:
            action, recovery_context = self.oracle.recovery_action(actor_environment)
            return action, "FROZEN_KAPPA", recovery_context, takeover_started
        mapped = self.oracle.certified_support_action(
            actor_environment,
            proposed_action,
            charging_state=False,
            charging_support_required=False,
        )
        if mapped is not None:
            executed_action, recovery_context = mapped
            return executed_action, "POLICY_CERTIFIED_RUN_SUPPORT", recovery_context, takeover_started
        self.oracle.begin_recovery()
        self._recovery_active = True
        self._kappa_takeover_count += 1
        takeover_started = True
        action, recovery_context = self.oracle.recovery_action(actor_environment)
        return action, "FROZEN_KAPPA", recovery_context, takeover_started

    def _apply_takeover_cost(self, reward: float, info: dict[str, Any]) -> float:
        cost = float(self.actor_environment.reward_config.backup_intervention_cost)
        components = dict(info["reward_components"])
        components["backup_intervention_event_cost"] = -cost
        info["reward_components"] = components
        records = info["goal_attempt_records"]
        if records and info["task_completed_now"]:
            records[0]["reward_component_totals"]["backup_intervention_event_cost"] -= cost
        else:
            self.actor_environment._goal_attempt["reward_component_totals"][
                "backup_intervention_event_cost"
            ] -= cost
        return float(reward - cost)

    def step(self, action: np.ndarray):
        proposed_action = np.asarray(action, dtype=np.float32)
        if proposed_action.shape != self.action_space.shape:
            raise ValueError("certified execution action has the wrong shape")
        if not np.all(np.isfinite(proposed_action)):
            raise FloatingPointError("certified execution received a nonfinite action")
        proposed_action = np.clip(proposed_action, -1.0, 1.0)
        executed_action, authority, recovery_context, takeover_started = self._select_action(proposed_action)
        observation, reward, terminated, truncated, info = self.env.step(executed_action)
        if recovery_context is not None:
            if authority == "FROZEN_KAPPA":
                self.oracle.commit_recovery(recovery_context)
            else:
                self.oracle.commit_policy(recovery_context)
        if terminated:
            raise RuntimeError("CERTIFIED_EXECUTION_UNEXPECTED_TERMINATION")
        if info["energy_stranded"]:
            raise RuntimeError("CERTIFIED_EXECUTION_ENERGY_STRANDED")
        if info["collision"]:
            raise RuntimeError("CERTIFIED_EXECUTION_COLLISION_VIOLATION")

        if authority == "FROZEN_KAPPA":
            self._kappa_execution_steps += 1
        elif authority == "CERTIFIED_CHARGER_HOLD":
            self._charger_hold_steps += 1
        elif authority == "POLICY_CERTIFIED_DEPARTURE_SUPPORT":
            self._verified_departure_steps += 1
            if not self.actor_environment._charging_admissible(
                self.actor_environment.state.position,
                self.actor_environment.state.velocity,
            ):
                self.oracle.end_recovery()
                self._recovery_active = False
        else:
            self._policy_execution_steps += 1

        if takeover_started:
            reward = self._apply_takeover_cost(reward, info)
        executed_action = np.asarray(executed_action, dtype=np.float32)
        info |= {
            "proposed_normalized_action": proposed_action.copy(),
            "executed_normalized_action": executed_action.copy(),
            "execution_authority": authority,
            "policy_action_accepted": authority.startswith("POLICY_CERTIFIED_"),
            "kappa_takeover_started": takeover_started,
            "kappa_takeover_count": self._kappa_takeover_count,
            "fallback_count": self._kappa_takeover_count,
            "kappa_execution_steps": self._kappa_execution_steps,
            "charger_hold_steps": self._charger_hold_steps,
            "verified_departure_steps": self._verified_departure_steps,
            "policy_execution_steps": self._policy_execution_steps,
            "recovery_active": self._recovery_active,
            "online_teacher_override": False,
            "replay_action_semantics": "PROPOSED_NORMALIZED_CERTIFIED_SUPPORT_COORDINATE",
            "executed_action_semantics": "NORMALIZED_PHYSICAL_ACCELERATION",
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.oracle.environment.close()
        super().close()
