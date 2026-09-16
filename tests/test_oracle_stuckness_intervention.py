from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.directional_navigation.battery_sortie import CAPACITY
from experiments.directional_navigation.oracle_stuckness_intervention import (
    FrozenIntervention,
    NO_PROGRESS_STEPS,
    OracleStucknessEnv,
    state_rng_fingerprint,
)
from scripts.analyze_oracle_stuckness_dev import load_complete_grid
from scripts.validate_oracle_stuckness_contract import object_hash, validate


def make_env(method: int = 0) -> OracleStucknessEnv:
    env = OracleStucknessEnv({
        "job_id": method,
        "world_seed": 1_139_000_001,
        "head_seed": 0,
        "method": method,
        "method_name": f"M{method}",
        "threshold": .4,
        "obstacles": 0,
        "soc": 1.,
    })
    env.reset()
    return env


def zero_action(_observation, _returning):
    return np.zeros(3, np.float32)


def test_snapshot_probe_restore_is_byte_identical_and_does_not_leak():
    env = make_env(4)
    try:
        before = env.snapshot()
        before_fingerprint = state_rng_fingerprint(env)
        clocks = (env.base.simulation_time, env.base.agent.energy,
                  env.base.tasks_completed, len(env.base.trajectory_log))
        label = env.completion_within_probe(zero_action, horizon=3)
        assert label["probe_steps"] == 3
        assert state_rng_fingerprint(env) == before_fingerprint
        assert clocks == (env.base.simulation_time, env.base.agent.energy,
                          env.base.tasks_completed, len(env.base.trajectory_log))
    finally:
        env.close()


def test_probe_label_matches_direct_continuation_fixture():
    env = make_env(4)
    try:
        anchor = env.snapshot()
        label = env.completion_within_probe(zero_action, horizon=4)
        env.restore(anchor)
        start = env.base.tasks_completed
        completed = False
        observation = env.observation()
        for _ in range(4):
            observation, _, _, _, info = env.step(np.zeros(4, np.float32))
            completed |= env.base.tasks_completed > start
            if completed or info.get("exhausted"):
                break
        assert label["completed_within_4000"] is completed
    finally:
        env.close()


def test_probe_restoration_has_identical_next_continuation():
    env = make_env(4)
    try:
        anchor = env.snapshot()
        env.completion_within_probe(zero_action, horizon=2)
        env.step(np.zeros(4, np.float32))
        after_probe_then_step = state_rng_fingerprint(env)
        env.restore(anchor)
        env.step(np.zeros(4, np.float32))
        direct_step = state_rng_fingerprint(env)
        assert after_probe_then_step == direct_step
    finally:
        env.close()


def test_timeout_triggers_exactly_at_4000():
    env = make_env(1)
    try:
        policy = FrozenIntervention(1)
        env.base.steps_in_current_task = 3999
        assert policy.decide(env, zero_action)[0] is False
        env.base.steps_in_current_task = 4000
        commit, event = policy.decide(env, zero_action)
        assert commit and event["kind"] == "timeout"
    finally:
        env.close()


def test_no_progress_requires_full_256_step_interval_and_positive_soc_margin():
    env = make_env(2)
    try:
        policy = FrozenIntervention(2)
        for _ in range(NO_PROGRESS_STEPS):
            assert policy.no_progress_trigger(env) is False
        assert policy.no_progress_trigger(env) is True
        env.base.agent.energy = .4 * CAPACITY
        assert policy.no_progress_trigger(env) is False
    finally:
        env.close()


def test_replan_preserves_plant_state_and_advances_keyed_task():
    env = make_env(3)
    try:
        env.base.agent.vel[:] = np.array([.1, -.2, .3], np.float32)
        position = env.base.agent.pos.copy()
        energy = env.base.agent.energy
        tasks = env.base.tasks_completed
        old = env.task_key()
        detail = env.replan_current_task()
        assert np.array_equal(env.base.agent.pos, position)
        assert env.base.agent.energy == energy
        assert np.array_equal(env.base.agent.vel, np.zeros(3, np.float32))
        assert env.base.tasks_completed == tasks
        assert env.task_key() != old
        assert detail["old_task_key"] == old
    finally:
        env.close()


def test_collision_count_is_one_boolean_union_per_step(monkeypatch):
    env = make_env(0)
    try:
        original = env.base.step
        def both_contacts(action):
            observation, reward, terminated, truncated, info = original(action)
            info["obstacle_collision"] = True
            info["boundary_contact"] = True
            return observation, reward, terminated, truncated, info
        monkeypatch.setattr(env.base, "step", both_contacts)
        before = env.contacts
        env.step(np.zeros(4, np.float32))
        assert env.contacts == before + 1
    finally:
        env.close()


def test_analyzer_refuses_incomplete_formal_grid(tmp_path: Path):
    contract = {"worlds": [
        {"world_index": i, "world_seed": 100 + i, "split": "DEV",
         "physical_world_identity": f"sha256:{i:064x}"}
        for i in range(24)
    ]}
    with pytest.raises(RuntimeError, match="missing"):
        load_complete_grid(tmp_path, contract)


def test_validator_rejects_duplicate_world_identity(tmp_path: Path):
    worlds = [{"world_index": i, "world_seed": 100 + i,
               "split": "SMOKE_DEBUG" if i == 0 else "DEV",
               "physical_world_identity": "sha256:" + "0" * 64} for i in range(25)]
    pre = {"contract_id": "test"}
    pre["record_hash"] = object_hash(pre)
    contract = {"schema_version": "oracle-stuckness-contract-v1",
                "confirm_allocated": False,
                "methods": [{"method": i} for i in range(5)],
                "worlds": worlds, "source_registry": {}, "preaccess": pre,
                "access_events": []}
    (tmp_path / "contract.json").write_text(json.dumps(contract))
    (tmp_path / "freeze_receipt.json").write_text(json.dumps({
        "preaccess_record_hash": pre["record_hash"],
        "source_registry_hash": object_hash({}),
    }))
    result = validate(tmp_path / "contract.json", scan_prior=False)
    assert not result["valid"]
    assert any("25 unique" in error for error in result["errors"])
