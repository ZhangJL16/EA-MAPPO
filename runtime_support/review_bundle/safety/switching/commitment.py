from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SortieMode(str, Enum):
    TASK = "TASK"
    CHARGER_COMMITTED = "CHARGER_COMMITTED"


@dataclass(frozen=True)
class CommitmentDecision:
    mode: SortieMode
    switched_now: bool
    active_goal_kind: str
    reason: str


class SortieCommitment:
    def __init__(self) -> None:
        self.mode = SortieMode.TASK
        self.task_to_charger_count = 0

    def decide(
        self,
        *,
        task_continuation_feasible: bool,
        charger_action_feasible: bool,
    ) -> CommitmentDecision:
        if self.mode is SortieMode.CHARGER_COMMITTED:
            return CommitmentDecision(self.mode, False, "CHARGER", "commitment_is_absorbing")
        if task_continuation_feasible:
            return CommitmentDecision(self.mode, False, "TASK", "task_action_feasible")
        if charger_action_feasible:
            self.mode = SortieMode.CHARGER_COMMITTED
            self.task_to_charger_count += 1
            return CommitmentDecision(self.mode, True, "CHARGER", "energy_stopping_condition")
        return CommitmentDecision(self.mode, False, "NONE", "no_admissible_action")

    def complete_charge_and_start_sortie(self) -> None:
        self.mode = SortieMode.TASK
        self.task_to_charger_count = 0
