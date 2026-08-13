from __future__ import annotations

import numpy as np

from safety import JointSafetyFilter
from safety.switching import SortieCommitment, SortieMode


def test_joint_filter_uses_intersection_not_weighted_sum() -> None:
    decision = JointSafetyFilter(reserve=1.0).evaluate(
        collision_upper=np.array([0.01, 0.50, 0.01]),
        energy_upper=np.array([9.0, 1.0, 11.0]),
        task_scores=np.array([1.0, 100.0, 50.0]),
        collision_budget=0.05,
        conservative_energy=10.5,
    )
    np.testing.assert_array_equal(decision.admissible, np.array([True, False, False]))
    assert decision.selected_index == 0


def test_commitment_is_irreversible_within_sortie() -> None:
    machine = SortieCommitment()
    first = machine.decide(task_continuation_feasible=False, charger_action_feasible=True)
    assert first.switched_now and first.mode is SortieMode.CHARGER_COMMITTED
    for _ in range(20):
        decision = machine.decide(task_continuation_feasible=True, charger_action_feasible=True)
        assert decision.mode is SortieMode.CHARGER_COMMITTED
        assert not decision.switched_now
    assert machine.task_to_charger_count == 1


def test_new_sortie_only_after_charge_completion() -> None:
    machine = SortieCommitment()
    machine.decide(task_continuation_feasible=False, charger_action_feasible=True)
    machine.complete_charge_and_start_sortie()
    assert machine.mode is SortieMode.TASK
    assert machine.task_to_charger_count == 0
