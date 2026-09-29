import unittest

from collect_initial_choice_lb import binomial_upper_tail, one_sided_lower


class InitialChoiceStatisticsTest(unittest.TestCase):
    def test_three_capacity_group_threshold_at_100(self):
        self.assertGreater(binomial_upper_tail(10, 100, .05), .05 / 3)
        self.assertLessEqual(binomial_upper_tail(11, 100, .05), .05 / 3)

    def test_lower_bound_is_inverse_tail(self):
        alpha = .05 / 3
        lower = one_sided_lower(11, 100, alpha)
        self.assertAlmostEqual(binomial_upper_tail(11, 100, lower), alpha, places=10)
        self.assertGreater(lower, .05)

    def test_zero_events_never_excludes(self):
        self.assertEqual(one_sided_lower(0, 100, .05 / 3), 0.)


if __name__ == "__main__":
    unittest.main()
