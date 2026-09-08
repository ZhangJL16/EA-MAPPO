"""Necessary label, observation, data-commit and changed-resume checks."""
import json
import subprocess
import sys

import numpy as np
import pytest

from experiments.directional_navigation.return_energy_data import (
    ReturnEnergyRecovery, scene_split, suffix_labels,
)
from scripts.collect_new_navigation_return_energy import commit_episode


def test_suffix_labels_are_future_contact_not_episode_contact():
    energy, safe = suffix_labels([1., 2., 3., 4.], [0, 1, 0, 0], True)
    np.testing.assert_array_equal(energy, [10., 9., 7., 4.])
    np.testing.assert_array_equal(safe, [False, False, True, True])
    assert not suffix_labels([1., 2.], [0, 0], False)[1].any()
    with pytest.raises(ValueError, match="censored"):
        suffix_labels([1.], [0], False, complete=False)
    with pytest.raises(ValueError, match="binary"):
        suffix_labels([1.], [2], True)


def test_known_charger_observation_and_exact_worker_restore():
    make = lambda: ReturnEnergyRecovery(seed_start=1040000001, seed_stride=1,
                                        horizon=7, obstacles=2, hocbf=True)
    first, restored = make(), make()
    try:
        obs, _ = first.reset()
        assert obs.shape == (2056,) and np.isfinite(obs).all()
        np.testing.assert_array_equal(first.base.active_goal, first.base.charger_position)
        obs2, _ = restored.reset(seed=1040000001)
        np.testing.assert_array_equal(obs, obs2)
        first.step(np.array([.1, -.1, .2], np.float32))
        restored.restore(first.snapshot())
        action = np.array([-.1, .1, .2], np.float32)
        a, b = first.step(action), restored.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:] == b[1:]
        assert a[4]["realized_energy"] > 0
        assert not any("boundary" in k or "obstacle" in k for k in a[4])
    finally:
        first.close()
        restored.close()


def test_episode_commit_replay_and_scene_split(tmp_path):
    data = {"observations": [np.zeros(2056, np.float32)] * 2, "indices": [0, 2],
            "energy": [1., 2., 3.], "contact": [0, 1, 0]}
    episode = {"seed": 123, "steps": 3, "goal_reached": True, "safe_goal": False,
               "collision_count": 1, "energy": 6.}
    row = commit_episode(tmp_path, data, episode)
    assert row == commit_episode(tmp_path, data, episode)
    assert row["split"] == scene_split(123)
    with np.load(tmp_path / row["path"]) as saved:
        np.testing.assert_array_equal(saved["safe_return"], [False, True])
    data["observations"][0] = np.ones(2056, np.float32)
    with pytest.raises(ValueError, match="differs"):
        commit_episode(tmp_path, data, episode)


def test_one_tiny_full_runner_and_resumption(tmp_path):
    """The single new-path smoke also tests resumability, not model performance."""
    import torch
    output = tmp_path / "run"
    command = [sys.executable, "scripts/collect_new_navigation_return_energy.py",
               "--source", "artifacts/hocbf_correction_sac_20260908_v1",
               "--output-dir", str(output), "--device", "cpu", "--num-envs", "1",
               "--steps", "8", "--checkpoint-steps", "8", "--sample-stride", "2", "--smoke"]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    first = json.loads((output / "dataset_index.json").read_text())
    assert first["steps"] == 8 and first["completed_steps"] == 7 and first["censored_steps"] == 1
    assert first["policy_hash_before"] == first["policy_hash_after"]
    resumed_command = command.copy()
    resumed_command[resumed_command.index("--steps") + 1] = "16"
    subprocess.run(resumed_command + ["--resume"], check=True, capture_output=True, text=True, timeout=120)
    second = json.loads((output / "dataset_index.json").read_text())
    assert second["steps"] == 16 and second["completed_steps"] == 14 and second["censored_steps"] == 2
    assert second["episodes"][:1] == first["episodes"]
    state = torch.load(output / "checkpoint_000000016/state.pt", weights_only=False)
    assert len(state["pending"][0]["energy"]) == 2
    assert state["workers"] and state["steps"] == 16
