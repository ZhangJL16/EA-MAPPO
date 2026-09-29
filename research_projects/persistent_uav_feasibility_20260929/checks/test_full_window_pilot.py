import unittest

from run_full_window_pilot import choose_full_recharge


class FullWindowPolicyTests(unittest.TestCase):
    def setUp(self):
        self.observation = dict(
            decision_required=True, can_recharge=False,
            position=[0.0, 0.0, 2.0], charger_position=[0.0, 0.0, 2.0],
            queue=[dict(id=3, arrival=0.0, position=[100.0, 0.0, 2.0]),
                   dict(id=4, arrival=10.0, position=[10.0, 0.0, 2.0])])

    def test_nearest_and_fifo_make_distinct_legal_choices(self):
        near = choose_full_recharge(self.observation, "full_recharge_nearest", None)
        fifo = choose_full_recharge(self.observation, "full_recharge_fifo", None)
        self.assertEqual((near.kind, near.task_id), ("serve", 4))
        self.assertEqual((fifo.kind, fifo.task_id), ("serve", 3))

    def test_recharge_when_legal_even_with_backlog(self):
        self.observation["can_recharge"] = True
        for method in ("full_recharge_nearest", "full_recharge_fifo"):
            action = choose_full_recharge(self.observation, method, None)
            self.assertEqual((action.kind, action.task_id), ("recharge", None))

    def test_no_action_outside_decision_rejected(self):
        self.observation["decision_required"] = False
        with self.assertRaises(ValueError):
            choose_full_recharge(self.observation, "full_recharge_nearest", None)


if __name__ == "__main__":
    unittest.main()
