from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from envs.navigation import OperationalEnergyWrapper
from safety.switching import EnergySwitchController, EnergySwitchDecision, SortieMode


class GoalPolicy(Protocol):
    def action(self, observation: np.ndarray, *, deterministic: bool = True) -> np.ndarray: ...


class ReturnEnergyPredictor(Protocol):
    def predict(self, observation: np.ndarray, action: np.ndarray) -> float: ...


@dataclass(frozen=True)
class ManagedGoalDecision:
    switch: EnergySwitchDecision
    charger_observation: np.ndarray
    charger_action: np.ndarray


def decide_managed_goal(
    environment: OperationalEnergyWrapper,
    policy: GoalPolicy,
    predictor: ReturnEnergyPredictor,
    controller: EnergySwitchController,
    *,
    task_goal: np.ndarray,
) -> ManagedGoalDecision:
    navigation = environment.navigation_env
    charger_observation = navigation.observation_for_goal(navigation.station_position)
    charger_action = policy.action(charger_observation, deterministic=True)
    predicted_return = predictor.predict(charger_observation, charger_action)
    decision = controller.decide(
        remaining_energy=environment.operational_energy,
        predicted_return_energy=predicted_return,
        task_goal=task_goal,
        charger_goal=navigation.station_position,
    )
    if decision.mode is SortieMode.CHARGER_COMMITTED and not np.allclose(
        navigation.goal, navigation.station_position
    ):
        navigation.set_external_goal(navigation.station_position)
    elif decision.mode is SortieMode.TASK and not np.allclose(navigation.goal, task_goal):
        raise RuntimeError("TASK mode goal changed outside the energy-management interface")
    return ManagedGoalDecision(decision, charger_observation, charger_action)
