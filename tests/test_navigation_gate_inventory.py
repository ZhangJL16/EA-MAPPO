from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts.audit_navigation_gate_inventory import (
    build_inventory,
    classify_navigation_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def _metrics() -> dict[str, object]:
    return {
        "num_tasks": 500,
        "global_env_transitions": 500_000,
        "overall_success_rate": 0.99,
        "distance_bucket_success": {"near": 0.98, "far": 0.96},
        "mean_path_ratio": 1.05,
        "obstacle_collision_steps": 0,
        "boundary_contact_step_rate": 0.0,
    }


def test_inventory_requires_explicit_wrapper_authorization() -> None:
    raw = classify_navigation_payload(
        artifact_label="navigation_result.json",
        payload=_metrics(),
        source="test",
    )
    assert raw is not None
    assert raw["classification"] == "FORMAL_METRICS_PASS_BUT_UNAUTHORIZED"
    assert raw["formal_gate_passed"] is False


def test_inventory_accepts_only_fully_authorized_wrapper() -> None:
    payload = {
        "final_navigation": _metrics(),
        "checkpoint_sha256": "a" * 64,
        "navigation_gate_passed": True,
        "downstream_navigation_ready": True,
        "downstream_stages_authorized": True,
    }
    row = classify_navigation_payload(
        artifact_label="EVALUATION_COMPLETED.json",
        payload=payload,
        source="test",
    )
    assert row is not None
    assert row["classification"] == "FORMAL_PASS_AUTHORIZED"
    report = build_inventory(directory_rows=[row], archive_rows=[])
    assert report["status"] == "PASSING_NAVIGATION_ARTIFACT_FOUND"
    assert report["num_authorized_passes"] == 1


def test_inventory_does_not_promote_smoke_or_terminal_marker() -> None:
    smoke_metrics = _metrics()
    smoke_metrics["num_tasks"] = 10
    smoke_metrics["global_env_transitions"] = 0
    smoke = classify_navigation_payload(
        artifact_label="fixed_navigation_smoke/navigation_result.json",
        payload=smoke_metrics,
        source="test",
    )
    stopped = classify_navigation_payload(
        artifact_label="R5/STOPPED_NAVIGATION_NOT_READY.json",
        payload={"status": "STOPPED_NAVIGATION_NOT_READY"},
        source="test",
    )
    assert smoke is not None and smoke["classification"] == "SMOKE_OR_DIAGNOSTIC"
    assert stopped is not None and stopped["classification"] == "TERMINAL_STOP_MARKER"
    report = build_inventory(directory_rows=[smoke, stopped], archive_rows=[])
    assert report["status"] == "NO_PASSING_NAVIGATION_ARTIFACT_FOUND"


def test_inventory_script_is_directly_executable() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/audit_navigation_gate_inventory.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--artifacts-root" in result.stdout


def test_inventory_separates_fixed_research_contract_from_deployment_pass() -> None:
    payload = {
        "schema": "fixed_navigation_energy_research_contract_v1",
        "status": "AUTHORIZED",
        "baseline_id": "R3",
        "energy_research_authorized": True,
        "frozen_policy_required": True,
        "performance_metrics_are_descriptive": True,
        "checkpoint_sha256": "b" * 64,
        "navigation_metrics": {
            **_metrics(),
            "overall_success_rate": 0.96,
            "mean_path_ratio": 1.188,
            "obstacle_collision_steps": 2242,
        },
    }
    row = classify_navigation_payload(
        artifact_label="navigation_platform_contract.json",
        payload=payload,
        source="test",
    )
    assert row is not None
    assert row["classification"] == "FIXED_BASELINE_RESEARCH_AUTHORIZED"
    assert row["formal_gate_passed"] is False
    assert row["energy_research_authorized"] is True
    report = build_inventory(directory_rows=[row], archive_rows=[])
    assert report["status"] == "NO_PASSING_NAVIGATION_ARTIFACT_FOUND"
    assert report["num_fixed_baseline_research_contracts"] == 1
    assert "fixed-baseline" in report["next_legal_action"]
