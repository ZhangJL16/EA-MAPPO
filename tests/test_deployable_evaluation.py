"""V2 evaluation cursor semantics and parked workers, not performance gates."""

from argparse import Namespace
import json

import numpy as np
import pytest

from scripts import evaluate_deployable_observation_v2 as evaluator


class ZeroPolicy:
    def predict(self, observations, deterministic=True):
        assert deterministic and observations.shape[1] == 1039
        return np.zeros((len(observations), 3), np.float32), None


def test_completed_worker_parks_without_changing_episode_cursor():
    env = evaluator.ParkedDeployableEvaluation(battery_capacity=100., horizon=1, obstacles=0)
    try:
        env.reset(seed=31)
        assert env.step(np.zeros(3))[2]
        cursor = env.next_episode_index
        obs, _ = env.reset()
        assert env.parked and obs.shape == (1039,)
        assert env.step(np.zeros(3))[4] == {"parked": True}
        assert env.next_episode_index == cursor
        env.reset(seed=32)
        assert not env.parked and env.steps == 0
    finally:
        env.close()


def test_committed_batch_resume_and_reject_wrong_task(tmp_path, monkeypatch):
    tasks, seed = evaluator.load_tasks()
    tasks = tasks[:3]
    args = Namespace(num_envs=2, horizon=1, obstacles=0, battery_capacity=100., initial_soc=1.)
    original_write = evaluator.atomic_json
    def save_then_pause(path, value):
        original_write(path, value)
        if path.name == "evaluation_stratified.json":
            evaluator.STOP = True
    monkeypatch.setattr(evaluator, "STOP", False)
    monkeypatch.setattr(evaluator, "atomic_json", save_then_pause)
    assert not evaluator.evaluate(ZeroPolicy(), tmp_path, args, tasks, seed)
    path = tmp_path / "evaluation_stratified.json"
    before = json.loads(path.read_text())
    assert len(before) == 2
    monkeypatch.setattr(evaluator, "STOP", False)
    monkeypatch.setattr(evaluator, "atomic_json", original_write)
    assert evaluator.evaluate(ZeroPolicy(), tmp_path, args, tasks, seed)
    after = json.loads(path.read_text())
    assert len(after) == 3 and after[:2] == before
    assert all(r["collision_count"] == 0 for r in after)
    after[0]["source_task_index"] += 1
    with pytest.raises(ValueError, match="immutable"):
        evaluator.validate_prefix(after, tasks, seed)


def test_summary_counts_safe_arrivals_not_just_arrivals():
    rows = [{"goal_reached": True, "safe_goal": True, "termination": "goal",
             "collision_count": 0, "success_path_ratio": 1., "energy": 2.},
            {"goal_reached": True, "safe_goal": False, "termination": "goal",
             "collision_count": 10, "success_path_ratio": 1.5, "energy": 3.}]
    result = evaluator.summarize(rows)
    assert result["goal_reached"] == 2 and result["safe_goal"] == 1
    assert result["mean_collision_count"] == 5
