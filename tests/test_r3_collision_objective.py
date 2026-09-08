import numpy as np
import pytest

from experiments.r3_collision.core import correction_distance, shaped_transition, survival_weight


def test_radial_action_alias_does_not_create_safety_correction():
    # Raw [1,1,0] and [1/sqrt(2),1/sqrt(2),0] command identical physical acceleration.
    physical = np.array([5/np.sqrt(2), 5/np.sqrt(2), 0.0])
    assert correction_distance(physical, physical, 5, 3) == 0
    assert correction_distance(np.array([5.,0,3]), np.array([-5.,0,-3]), 5, 3) == 1


def test_hazard_is_product_over_substeps_and_contact_dominates_goal():
    distances = np.array([0.0, .5, .1, .7])
    q = survival_weight(contact=False, correction_integral=float((distances**2).sum()*.05), kappa=3)
    assert q == pytest.approx(np.prod(np.exp(-3*distances**2*.05)))
    reward, beta = shaped_transition(goal=True, contact=True, correction_integral=0,
                                    kappa=0, gamma=.99, potential=-.2, next_potential=0)
    assert beta == 0
    assert reward == pytest.approx(.2)  # Base Q=0; shifted Q=-Phi, not a goal bonus.


def test_potential_cancellation_preserves_action_order():
    phi, next_phi, base_next = -.7, -.2, .8
    for goal, contact in [(False, False), (True, False), (False, True)]:
        for kappa in [0, .5, 4]:
            r, beta = shaped_transition(goal=goal, contact=contact, correction_integral=.13,
                                       kappa=kappa, gamma=.99, potential=phi, next_potential=next_phi)
            q = survival_weight(contact=contact, correction_integral=.13, kappa=kappa)
            base = q*(goal and not contact) + beta*base_next
            assert r + beta*(base_next-next_phi) == pytest.approx(base-phi)


def test_invalid_inputs_fail():
    with pytest.raises(ValueError):
        survival_weight(contact=False, correction_integral=-1, kappa=1)
    with pytest.raises(ValueError):
        correction_distance(np.array([6.,0,0]), np.zeros(3), 5, 3)
