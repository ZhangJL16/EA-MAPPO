"""Mathematical verification tests, not simulation/performance experiments."""
from fractions import Fraction as F

from scripts.verify_bundling_finite_budget_theory import (
    exact_certificate, exact_response, primitive_low_bayes, fixed_point_relaxation,
)


def test_exact_twelve_step_lower_and_upper():
    certificate = exact_certificate(12)
    assert F(certificate["lower"]) == F(2173, 8000)
    assert F(certificate["upper"]) == F(8604621, 40000000)
    assert F(certificate["gap"]) == F(2260379, 40000000)
    assert certificate["gap_gt_one_twentieth"]


def test_primitive_adaptive_low_agrees_with_route_bellman():
    priors = ((F(1, 2), F(9, 20), F(1, 20)), (F(1, 3),)*3)
    for prior in priors:
        for T in (0, 1, 2, 3, 4, 6, 12):
            assert primitive_low_bayes(T, prior) == exact_response(T, prior, False)[1]


def test_known_model_charger_terminal_execution_is_unchanged():
    for hypothesis in range(3):
        prior = tuple(F(int(i==hypothesis)) for i in range(3))
        for bundled in (False, True):
            risks, value, _, _ = exact_response(12, prior, bundled, True)
            assert value == 0
            assert risks[hypothesis] == 0


def test_high_witness_never_requires_an_incomplete_departure():
    prior = (F(71, 125), F(421, 1000), F(11, 1000))
    _, _, _, policy = exact_response(12, prior, True, True)
    length = {"dock": 1, "A": 3, "B": 3, "AB": 4}
    assert all(length[action]<=state[0] for state, action in policy.items())


def test_static_high_capacity_lower_bound_is_exact():
    prior = (F(21, 23), F(0), F(2, 23))
    theta = ((F(1, 20), F(1, 20)), (F(3, 10), F(1, 20)), (F(3, 10), F(19, 20)))
    gain = (F(1, 20), F(1, 10), F(19, 60))
    paths = ((1, None), (2, ()), (3, (0,)), (3, (1,)), (4, (0, 1)))
    for duration, channels in paths:
        reward = [F(1, 20) if channels is None else sum((th[c] for c in channels), F(0))
                  for th in theta]
        weighted_cost = sum(p*(g*duration-r) for p, g, r in zip(prior, gain, reward))
        assert weighted_cost >= F(8, 345)*duration
    # Static mixture 5/46*(one AB + eight dock) +41/46*(three AB).
    risks = (F(32, 115), F(21, 115), F(32, 115))
    assert max(risks) == 12*F(8, 345)


def test_original_fixed_point_bound_is_vacuous_at_twelve():
    assert fixed_point_relaxation(12, False) == 0
    assert fixed_point_relaxation(12, True) == 0


def test_low_bayes_policy_risk_has_a_direct_reward_formula():
    # Dock once, inspect A twice. If both rewards are zero, use three dock
    # services; otherwise take one more full A path. Finally take partial A.
    # This directly evaluates the chosen policy, not its global optimality.
    gain = (F(1, 20), F(1, 10), F(19, 60))
    means = (F(1, 20), F(3, 10), F(3, 10))
    risks = tuple(12*g-(F(1, 20)+4*m+(F(3, 20)-m)*(1-m)**2)
                  for g, m in zip(gain, means))
    assert risks == (F(1039, 4000), F(47, 2000), F(5247, 2000))
