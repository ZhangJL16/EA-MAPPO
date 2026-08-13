from __future__ import annotations

import numpy as np

from safety import JointSafetyFilter
from safety.switching import EnergySwitchController, SortieCommitment, SortieMode


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


def test_energy_switch_controller_uses_boundary_and_is_one_way() -> None:
    controller = EnergySwitchController(reserve=0.2)
    task_goal = np.array([3.0, 3.0, 1.0])
    charger_goal = np.array([0.4, 0.5, 1.0])
    continuing = controller.decide(
        remaining_energy=1.1,
        predicted_return_energy=0.8,
        task_goal=task_goal,
        charger_goal=charger_goal,
    )
    assert continuing.mode is SortieMode.TASK
    np.testing.assert_array_equal(continuing.active_goal, task_goal)
    switching = controller.decide(
        remaining_energy=1.0,
        predicted_return_energy=0.8,
        task_goal=task_goal,
        charger_goal=charger_goal,
    )
    assert switching.mode is SortieMode.CHARGER_COMMITTED
    assert switching.switched_now
    np.testing.assert_array_equal(switching.active_goal, charger_goal)
    after = controller.decide(
        remaining_energy=2.0,
        predicted_return_energy=0.1,
        task_goal=task_goal,
        charger_goal=charger_goal,
    )
    assert after.mode is SortieMode.CHARGER_COMMITTED
    assert not after.switched_now
    assert controller.task_to_charger_count == 1
