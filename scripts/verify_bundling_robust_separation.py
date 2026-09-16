"""Rational inequality checks for the analytic open-box theorem; NO simulation.

These checks verify constants, not the coupling theorem or an external proof audit.
The pre-existing exact anchor receipt is an explicit theorem dependency.
"""
from fractions import Fraction as F
import json
from pathlib import Path


def verify():
    eps = F(1, 10000)
    r, u, v, w = F(1, 20), F(1, 20), F(3, 10), F(19, 20)
    assert min(r, u, v, w) > eps
    assert max(r, u, v, w) < 1-eps
    assert 3*r-u > 4*eps
    assert v-3*r > 4*eps
    assert w-3*v > 4*eps
    # KL <= chi-square and Pinsker: algebraic bounds on the entire open box.
    q_lower = (3*r-u-4*eps)*(v-eps)*(1-v+eps)/(v-u+2*eps)**2
    a_upper = (v-r+2*eps)/(2*(w-r-2*eps)**2)
    assert q_lower > F(33, 100)
    assert a_upper < F(16, 100)
    lower, upper = F(2173, 8000), F(8604621, 40000000)
    penalty = 12*(12+3)*eps
    robust_gap = lower-upper-penalty
    assert robust_gap == F('0.038509475')
    static_gap = F(32, 115)-upper-penalty
    assert static_gap > F('0.045145344')
    receipt = json.loads((Path(__file__).resolve().parents[1]/
        'artifacts/bundling_finite_budget_theory_20260916/strict_benefit_certificate.json').read_text())
    # Schema-independent check of the analytic corollary's constants;
    # full receipt recomputation is a separate pre-existing verification command.
    assert receipt
    assert F(1, 20)-24*(24+3)*F(1, 100000) == F('0.04352')
    return {'open_box_radius': str(eps), 'T12_gap_lower': str(robust_gap),
            'q_coefficient_lower': str(q_lower), 'A_coefficient_upper': str(a_upper),
            'static_gap_lower': str(static_gap), 'sampled_trajectories': 0,
            'proof_status': 'rational constants checked; analytic proof and anchor dependency explicit'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
