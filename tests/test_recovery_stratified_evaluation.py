"""The corrected evaluation uses the exact legacy balanced task set."""

from argparse import Namespace

import numpy as np

from scripts.resume_recovery_sac_ppo_stratified import (
    BUCKETS,
    load_tasks,
    stratified_evaluate,
)


class ZeroPolicy:
    def predict(self, observations, deterministic=True):
        assert deterministic
        return np.zeros((len(observations), 3), np.float32), None


def test_locked_tasks_are_balanced_and_interleaved():
    tasks, seed = load_tasks()
    assert seed == 170001 and len(tasks) == 500
    assert [task["distance_bucket"] for task in tasks[:5]] == list(BUCKETS)
    for bucket in BUCKETS:
        assert sum(task["distance_bucket"] == bucket for task in tasks) == 100
    assert len({task["source_task_index"] for task in tasks}) == 500


def test_tiny_stratified_execution_covers_all_buckets(tmp_path, monkeypatch):
    tasks, _ = load_tasks()
    monkeypatch.setattr(
        "scripts.resume_recovery_sac_ppo_stratified.load_tasks",
        lambda: (tasks[:5], 170001),
    )
    arm = tmp_path / "arm"
    arm.mkdir()
    args = Namespace(eval_tasks=5, num_envs=5, horizon=1, obstacles=0)
    assert stratified_evaluate(ZeroPolicy(), arm, tmp_path, args)
    summary = __import__("json").loads((arm / "summary.json").read_text())
    assert summary["tasks"] == 5 and summary["timeouts"] == 5
    assert all(summary["distance_buckets"][name]["tasks"] == 1 for name in BUCKETS)
