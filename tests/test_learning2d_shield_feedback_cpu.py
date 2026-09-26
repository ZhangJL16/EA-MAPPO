"""The paired conditions differ only on the specified safety events."""

from learning2d.shield_feedback_gym import intervention_penalty


def test_feedback_penalizes_only_forced_return_and_rejected_departure():
    trace = {"plant_trace": [
        {"event": "flight", "collision_intervention_steps": 8},
        {"event": "return_takeover"},
        {"event": "returning"},
        {"event": "departure_rejected"},
    ]}
    assert intervention_penalty(trace, feedback=False) == (0.0, 1, 1)
    assert intervention_penalty(trace, feedback=True) == (-3.8, 1, 1)


def test_voluntary_return_and_ordinary_qp_correction_get_no_extra_penalty():
    trace = {"plant_trace": [
        {"event": "returning"},
        {"event": "docked"},
        {"event": "flight", "collision_intervention_steps": 12},
    ]}
    assert intervention_penalty(trace, feedback=True) == (0.0, 0, 0)
