"""Exact structural counterexamples; no Monte Carlo or research outcomes."""
from fractions import Fraction as F
from itertools import product
import math
import unittest


def kl_bernoulli(p, q):
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


class TheoryObstructions(unittest.TestCase):
    def test_positive_information_does_not_make_finite_evidence_exclusive(self):
        for k in (F(2,100), F(4,100), F(8,100), F(16,100)):
            ps = (F(1,2)-k, F(1,2)+k)
            self.assertGreater(kl_bernoulli(*map(float, ps)), 0)
            for n in range(9):
                for ones in range(n+1):
                    self.assertTrue(all(p**ones * (1-p)**(n-ones) > 0 for p in ps))

    def test_recursive_root_contains_indistinguishable_decision_pair(self):
        models = list(product((0,1), repeat=2))
        k = F(8,100)
        p0 = {m: F(1,2)+(2*m[0]-1)*k for m in models}
        self.assertEqual(p0[(0,0)], p0[(0,1)])
        self.assertEqual(kl_bernoulli(float(p0[(0,0)]), float(p0[(0,1)])), 0)
        blocks = {tuple(m for m in models if p0[m] == p) for p in set(p0.values())}
        self.assertEqual(blocks, {((0,0),(0,1)), ((1,0),(1,1))})
        # Stage-two experiment is safe only within one of these blocks.
        self.assertFalse(all(m[0] == 0 for m in models))
        self.assertTrue(all(m[0] == 0 for m in ((0,0),(0,1))))

    def test_pairwise_overlap_does_not_imply_common_answer(self):
        acceptable = [set('ab'), set('bc'), set('ac')]
        self.assertTrue(all(acceptable[i] & acceptable[j] for i in range(3) for j in range(i)))
        self.assertEqual(set.intersection(*acceptable), set())

    def test_exact_cycle_clock_reveals_uav_parameter(self):
        energy, flight_time, rate = F(40), F(200), F(3,10)
        for theta in map(F, ('0.85','0.925','1.0','1.075','1.15')):
            observed_time = flight_time + theta * energy / rate
            self.assertEqual(rate * (observed_time-flight_time) / energy, theta)

    def test_nuisance_enumeration_size(self):
        self.assertEqual(2**(2+32), 17179869184)


if __name__ == '__main__':
    unittest.main()
