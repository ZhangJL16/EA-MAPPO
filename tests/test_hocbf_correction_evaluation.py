"""Tiny correctness/resume checks, not performance gates."""
from argparse import Namespace
import json
import os

import numpy as np
import pytest

from scripts import evaluate_hocbf_correction_fixed500 as evaluator
from scripts import queue_hocbf_fixed500 as queue


class ZeroPolicy:
    def predict(self, observations, deterministic=True):
        assert deterministic and observations.shape[1] == 2056
        return np.zeros((len(observations), 3), np.float32), None


@pytest.mark.parametrize("hocbf", [False, True])
def test_worker_parks(hocbf):
    env = evaluator.ParkedCorrectionEvaluation(seed_start=1, seed_stride=1,
                                               horizon=1, obstacles=0, hocbf=hocbf)
    try:
        env.reset(seed=31)
        assert env.step(np.zeros(3))[2]
        cursor = env.next_episode_index
        obs, _ = env.reset()
        assert env.parked and obs.shape == (2056,)
        assert env.step(np.zeros(3))[4]["parked"]
        assert env.next_episode_index == cursor
        env.reset(seed=32)
        assert not env.parked and env.steps == 0
    finally:
        env.close()


@pytest.mark.parametrize("hocbf", [False, True])
def test_batch_resume(tmp_path, monkeypatch, hocbf):
    tasks, seed = evaluator.load_tasks()
    tasks = tasks[:3]
    args = Namespace(num_envs=2, horizon=1, obstacles=0)
    original = evaluator.atomic_json

    def save_then_pause(path, value):
        original(path, value)
        if path.name == "evaluation_stratified.json":
            evaluator.STOP = True

    monkeypatch.setattr(evaluator, "STOP", False)
    monkeypatch.setattr(evaluator, "atomic_json", save_then_pause)
    assert not evaluator.evaluate(ZeroPolicy(), tmp_path, args, tasks, seed, hocbf)
    path = tmp_path / "evaluation_stratified.json"
    before = json.loads(path.read_text())
    assert len(before) == 2
    monkeypatch.setattr(evaluator, "STOP", False)
    monkeypatch.setattr(evaluator, "atomic_json", original)
    assert evaluator.evaluate(ZeroPolicy(), tmp_path, args, tasks, seed, hocbf)
    after = json.loads(path.read_text())
    assert len(after) == 3 and after[:2] == before
    assert all(r["hocbf"] == hocbf for r in after)
    after[0]["source_task_index"] += 1
    with pytest.raises(ValueError, match="immutable"):
        evaluator.validate_prefix(after, tasks, seed)


def test_successor_never_uses_mid_training_or_paused_checkpoint(tmp_path):
    status = tmp_path / "status.json"
    queue.write_json(status, {"status": "TRAINING", "pid": os.getpid(),
                              "checkpoint_transitions": 32768})
    assert queue.source_state(tmp_path, 131072) == "WAITING_FOR_TRAINING"
    queue.write_json(status, {"status": "PAUSED"})
    assert queue.source_state(tmp_path, 131072) == "SOURCE_PAUSED"
    queue.write_json(status, {"status": "TRAINING_COMPLETE_AWAITING_USER",
                              "adaptation_transitions": 131072, "checkpoint_transitions": 131072})
    assert queue.source_state(tmp_path, 131072) == "READY"
    assert queue.source_state(tmp_path, 524288) == "SOURCE_BUDGET_CHANGED"
    queue.write_json(tmp_path / "ERROR.json", {"error": "test"})
    assert queue.source_state(tmp_path, 131072) == "SOURCE_ERROR"


def test_successor_command_resumes_only_existing_evaluation(tmp_path):
    args = Namespace(source=tmp_path / "training", evaluation_dir=tmp_path,
                     checkpoint_transitions=131072, num_envs=8, device="cpu")
    assert "--resume" not in queue.evaluation_command(args)
    queue.write_json(tmp_path / "manifest.json", {})
    command = queue.evaluation_command(args)
    assert "--resume" in command and "131072" in command
    assert not any("energy" in part for part in command)
