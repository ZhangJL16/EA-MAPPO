"""Exact constants only; not empirical parameter sampling."""
from fractions import Fraction as F
from scripts.verify_bundling_robust_separation import verify


def test_open_box_constants():
    result = verify()
    assert result['sampled_trajectories'] == 0
    assert F(result['T12_gap_lower']) > F(3, 100)
    assert F(result['q_coefficient_lower']) > F(result['A_coefficient_upper'])


def test_existing_horizon_interval_perturbation_bound():
    # Reuse the old interval, do not enumerate new decision games.
    penalty = lambda T: T*(T+3)*F(1, 100000)
    assert all(F(1, 20)-penalty(T) >= F('0.04352') for T in range(12, 25))
