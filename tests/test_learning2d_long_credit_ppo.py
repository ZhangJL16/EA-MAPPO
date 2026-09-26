"""The long-credit diagnostic must isolate GAE lambda from frozen v3."""

import json

import pytest

from learning2d.evaluate_long_credit_ppo import evaluate
from learning2d.train_finite_horizon_ppo import training_manifest as v3_manifest
from learning2d.train_long_credit_ppo import training_manifest as v4_manifest


def test_v4_manifest_changes_only_credit_assignment_and_provenance():
    old = v3_manifest(101, 51200, 512, 512, 1.0, "cuda")
    new = v4_manifest(101, 51200, 512, 512, "cuda")
    assert old["ppo"]["gae_lambda"] == 0.95
    assert new["ppo"]["gae_lambda"] == 1.0
    assert new["paired_v3_model_sha256"]
    for manifest in (old, new):
        manifest.pop("protocol")
        manifest.pop("source_sha256")
        manifest.pop("paired_v3_model_sha256", None)
        manifest["ppo"].pop("gae_lambda")
    assert old == new


def test_v4_evaluator_rejects_incomplete_training(tmp_path):
    train = tmp_path / "train"
    train.mkdir()
    (train / "manifest.json").write_text(json.dumps(
        v4_manifest(101, 51200, 512, 512, "cuda")
    ))
    (train / "status.json").write_text(json.dumps({"complete": False}))
    with pytest.raises(ValueError, match="training must complete"):
        evaluate(tmp_path / "eval", map_id=32, training_output=train)
