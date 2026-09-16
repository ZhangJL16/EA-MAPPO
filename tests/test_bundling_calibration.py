import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"experiments"))
from bundling_calibration_core import (HYPOTHESES, METHODS, Learner, Plant, audits,
                                      bias_audit, catalogue, kl, solve_models)

spec = importlib.util.spec_from_file_location("bundling_runner", ROOT/"scripts/run_bundling_calibration.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_predictions_and_fixed_optima():
    rows = audits()
    truth = [r for r in rows if r["theta"] == (.30, .05)]
    assert [r["C"] for r in truth] == pytest.approx([.25/kl(.05, .95), .05/kl(.05, .95),
                                                    .25/kl(.05, .95), .25/kl(.05, .95)])
    assert all(r["span"] == pytest.approx(.2) and r["gain"] == pytest.approx(.1) for r in truth)
    assert all(r["confusing"] == [(.30, .95)] for r in truth)
    assert len(rows) == 12


@pytest.mark.parametrize("capacity,bundling", [(3, True), (4, True), (3, False), (4, False)])
def test_physical_debit_clock_pose_and_catalogue(capacity, bundling):
    for route in catalogue(capacity, bundling):
        plant = Plant(capacity, bundling)
        channels = []
        for action in route.actions:
            before = plant.state[2]
            c = plant.step(action)
            if c:
                channels.append(c)
            assert before >= 1
            assert plant.state[2] >= 0
            if action not in ("return", "dock"):
                assert plant.state[2] == before-1
        assert plant.time == route.length and plant.at_dock
        assert channels == (["dock"] if route.name == "dock" else list(route.channels))
    plant = Plant(capacity, bundling)
    plant.step("undock")
    assert not plant.at_dock and plant.state[2] == capacity-1
    plant.step("A")
    if capacity == 3 or not bundling:
        with pytest.raises(ValueError):
            plant.step("B")


def test_cost_aware_reference_matches_without_truth():
    routes = catalogue(4, True)
    a, b = Learner(routes, "resource_path"), Learner(routes, "cost_aware_reference")
    time = 0
    for j in range(150):
        choice_a, choice_b = a.select(time), b.select(time)
        assert choice_a == choice_b
        i, explore, _ = choice_a
        feedback = [(c, int((j % 9) == 0)) for c in routes[i].channels]
        a.observe(i, feedback, explore)
        b.observe(i, feedback, explore)
        time += routes[i].length
    with pytest.raises(ValueError):
        Learner(routes, "oracle_allocation")
    import inspect
    actor = inspect.getsource(Learner)
    assert all(token not in actor for token in ("TRUTH", "sealed_predictions", "open(", "read_text", "Path("))


def test_prediction_seal_and_source_change_denial(tmp_path, monkeypatch):
    output = tmp_path/"sealed"
    runner.prepare(output)
    runner.validate(output)
    assert not list(output.glob("*.checkpoint.json"))
    monkeypatch.setattr(runner, "sources", lambda: {})
    with pytest.raises(ValueError, match="source"):
        runner.validate(output)


def test_resume_identity_with_incomplete_path(tmp_path):
    straight, resumed = tmp_path/"straight", tmp_path/"resumed"
    straight.mkdir(); resumed.mkdir()
    runner.run_one(straight, 4, True, "resource_path", 64)
    runner.run_one(resumed, 4, True, "resource_path", 17)
    runner.run_one(resumed, 4, True, "resource_path", 64)
    name = "B4_on_resource_path.checkpoint.json"
    a, b = [json.loads((p/name).read_text()) for p in (straight, resumed)]
    a.pop("checkpoints"); b.pop("checkpoints")
    assert a == b


def test_capacity_only_control_identical_except_battery(tmp_path):
    for method in METHODS:
        runner.run_one(tmp_path, 3, False, method, 64)
        runner.run_one(tmp_path, 4, False, method, 64)
        states = [json.loads((tmp_path/f"B{b}_off_{method}.checkpoint.json").read_text()) for b in (3, 4)]
        assert states[0]["counts"] == states[1]["counts"]
        assert states[0]["pseudo_regret"] == states[1]["pseudo_regret"]
        for a, b in zip(states[0]["trace"], states[1]["trace"]):
            for item in ("before", "after"):
                a[item].pop(2); b[item].pop(2)
            assert a == b
