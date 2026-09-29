import unittest

from run import choose, margin


class SimpleModel:
    def predict(self, start, goal):
        distance = abs(goal[0] - start[0])
        return distance, distance


class MarginPolicyTests(unittest.TestCase):
    def setUp(self):
        self.observation = dict(
            decision_required=True, can_recharge=False,
            position=[0, 0, 100], charger_position=[0, 0, 100],
            battery=30.0,
            queue=[dict(id=1, position=[10, 0, 100]),
                   dict(id=2, position=[4, 0, 100])],
        )

    def test_conditional_margin_uses_both_legs(self):
        self.assertAlmostEqual(margin("C", [0, 0, 0], [10, 0, 0], [0, 0, 0]), 8.2)
        self.assertAlmostEqual(margin("C", [0, 0, 0], [3000, 0, 0], [0, 0, 0]), 13.0)

    def test_no_safe_task_at_full_station_keeps_legal_fallback(self):
        self.observation["battery"] = 3.0
        action, category = choose(self.observation, "U15", SimpleModel())
        self.assertEqual((action.kind, action.task_id, category),
                         ("serve", 2, "forced_station_fallback"))

    def test_empty_queue_return_and_margin_screen(self):
        self.observation["queue"] = []
        self.observation["can_recharge"] = True
        action, category = choose(self.observation, "C", SimpleModel())
        self.assertEqual((action.kind, category), ("recharge", "empty_queue_return"))
        self.observation["queue"] = [dict(id=1, position=[10, 0, 100])]
        self.observation["battery"] = 15.0
        action, category = choose(self.observation, "U5", SimpleModel())
        self.assertEqual((action.kind, category), ("recharge", "empty_filter_return"))


if __name__ == "__main__":
    unittest.main()
