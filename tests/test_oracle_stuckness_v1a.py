from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.directional_navigation.oracle_stuckness_intervention import METHODS, OracleStucknessEnv
from scripts.analyze_oracle_stuckness_v1a import (
    count_at, load_grid, simultaneous_max_t, survival_explanation,
)
from scripts.run_oracle_stuckness_intervention_v1a import run_method
from scripts.validate_oracle_stuckness_v1a_contract import file_hash, object_hash

ROOT = Path(__file__).resolve().parents[1]


class ZeroNavigator:
    def goal_action(self, observation, *, returning):
        return np.zeros(3, np.float32)


def physical_world(seed: int) -> dict:
    job = {"job_id": 0, "world_seed": seed, "head_seed": 0, "method": 0,
           "method_name": METHODS[0], "threshold": .4, "obstacles": 48, "soc": 1.}
    env = OracleStucknessEnv(job)
    try:
        env.reset()
        return {"world_index": 1, "world_seed": seed, "split": "SMOKE_DEBUG",
                "physical_world_identity": object_hash(env.physical_world_record())}
    finally:
        env.close()


def test_v1_intervention_source_is_unchanged_from_parent_freeze():
    parent = json.loads((ROOT / "artifacts/oracle_stuckness_intervention_20260916/contract.json").read_text())
    rel = "experiments/directional_navigation/oracle_stuckness_intervention.py"
    assert file_hash(ROOT / rel) == parent["source_registry"][rel]


def test_v1a_runner_logs_read_only_completion_timestamp_field():
    world = physical_world(1_139_100_001)
    row = run_method(ZeroNavigator(), world, 0, 2.0)
    assert row["schema_version"] == "oracle-stuckness-method-result-v1a"
    assert row["configured_denominator_seconds"] == 2.0
    assert row["tasks_completed"] == len(row["task_completion_timestamps"])
    assert all(t <= 2.0 for t in row["task_completion_timestamps"])


def test_count_at_uses_completion_timestamps_inclusively():
    row = {"task_completion_timestamps": [1.0, 4.0, 4.1, 9.0]}
    assert count_at(row, 4.0) == 2


def test_common_alive_survival_fraction_detects_post_death_gain():
    grid = {0: {
        0: {"time_alive_seconds": 1000.0, "task_completion_timestamps": [100.0]},
        4: {"time_alive_seconds": 7200.0, "task_completion_timestamps": [100.0, 2000.0]},
    }}
    fraction, denominator = survival_explanation(grid)
    assert denominator == 1.0
    assert fraction == 1.0


def test_zero_positive_gain_denominator_is_not_favorably_defined():
    grid = {0: {
        0: {"time_alive_seconds": 7200.0, "task_completion_timestamps": [100.0]},
        4: {"time_alive_seconds": 7200.0, "task_completion_timestamps": [100.0]},
    }}
    fraction, denominator = survival_explanation(grid)
    assert denominator == 0.0
    assert fraction is None


def test_simultaneous_family_contains_all_four_frozen_contrasts():
    rng = np.random.default_rng(4)
    result = simultaneous_max_t(rng.normal(size=(24, 4)))
    assert result["labels"] == ["M4-M0", "M4-M1", "M4-M2", "M4-M3"]
    assert len(result["lower"]) == 4


def test_v1a_analyzer_refuses_partial_grid(tmp_path: Path):
    contract = {"worlds": [
        {"world_index": i, "world_seed": 10 + i, "split": "DEV",
         "physical_world_identity": f"sha256:{i:064x}"} for i in range(24)
    ]}
    with pytest.raises(RuntimeError, match="missing"):
        load_grid(tmp_path, contract)


def test_runner_has_no_startup_only_dev_guard():
    source = (ROOT / "scripts/run_oracle_stuckness_intervention_v1a.py").read_text()
    assert 'args.stop_after_worlds != 1' not in source
    assert 'maximum = 1 if args.split == "SMOKE_DEBUG" else 24' in source

