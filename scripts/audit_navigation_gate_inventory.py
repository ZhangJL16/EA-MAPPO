#!/usr/bin/env python3
"""Inventory navigation evidence without promoting completion to Gate PASS."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path
from typing import Any, Iterable, Mapping

# Keep the documented ``uv run python scripts/...`` entry point usable.  Python
# otherwise places only ``scripts/`` on ``sys.path`` when this file is executed
# directly, while the prerequisite normalizer lives in the repository package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.energy_mc.gate_b_prerequisites import (
    FIXED_BASELINE_CONTRACT_SCHEMA,
    navigation_artifact_view,
)


FORMAL_TASKS = 500
FORMAL_TRANSITIONS = 500_000


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None


def _optional_sha(value: object) -> str | None:
    text = str(value).lower() if value is not None else ""
    if len(text) == 64 and all(character in "0123456789abcdef" for character in text):
        return text
    return None


def _metric_failures(metrics: Mapping[str, object]) -> list[str]:
    failures: list[str] = []
    success = _optional_float(metrics.get("overall_success_rate"))
    if success is None or success < 0.98:
        failures.append("overall_success_rate<0.98_or_missing")
    buckets = metrics.get("distance_bucket_success")
    if isinstance(buckets, Mapping) and buckets:
        bucket_values = [_optional_float(value) for value in buckets.values()]
        minimum_bucket = (
            min(value for value in bucket_values if value is not None)
            if all(value is not None for value in bucket_values)
            else None
        )
    else:
        minimum_bucket = _optional_float(
            metrics.get("minimum_distance_bucket_success")
        )
    if minimum_bucket is None or minimum_bucket < 0.95:
        failures.append("minimum_distance_bucket_success<0.95_or_missing")
    path_ratio = _optional_float(metrics.get("mean_path_ratio"))
    if path_ratio is None or path_ratio > 1.10:
        failures.append("mean_path_ratio>1.10_or_missing")
    boundary = _optional_float(metrics.get("boundary_contact_step_rate"))
    if boundary is None or boundary >= 0.01:
        failures.append("boundary_contact_step_rate>=0.01_or_missing")
    collisions = _optional_int(metrics.get("obstacle_collision_steps"))
    if collisions is None or collisions != 0:
        failures.append("obstacle_collision_steps_nonzero_or_missing")
    return failures


def _select_metrics(
    payload: Mapping[str, object],
) -> tuple[Mapping[str, object], str, bool | None, str | None]:
    view = navigation_artifact_view(payload)
    if view.wrapped:
        return (
            view.metrics,
            view.schema,
            view.authorization_passed,
            view.checkpoint_sha256,
        )
    legacy = payload.get("best_checkpoint_by_selection_rule")
    if isinstance(legacy, Mapping):
        return (
            legacy,
            "legacy_checkpoint_selection_completion",
            None,
            _optional_sha(legacy.get("checkpoint_sha256")),
        )
    return (
        payload,
        "raw_navigation_metrics",
        None,
        _optional_sha(payload.get("checkpoint_sha256")),
    )


def classify_navigation_payload(
    *,
    artifact_label: str,
    payload: Mapping[str, object],
    source: str,
) -> dict[str, object] | None:
    label_lower = artifact_label.lower()
    if Path(artifact_label).name == "STOPPED_NAVIGATION_NOT_READY.json":
        return {
            "artifact": artifact_label,
            "source": source,
            "schema": "terminal_navigation_stop_marker",
            "classification": "TERMINAL_STOP_MARKER",
            "explicit_status": payload.get("status"),
            "authorization_passed": False,
            "formal_gate_passed": False,
        }

    metrics, schema, authorization, checkpoint_sha = _select_metrics(payload)
    navigation_keys = {
        "overall_success_rate",
        "distance_bucket_success",
        "mean_path_ratio",
        "obstacle_collision_steps",
        "boundary_contact_step_rate",
        "num_tasks",
    }
    if len(navigation_keys.intersection(metrics)) < 3:
        return None

    tasks = _optional_int(metrics.get("num_tasks"))
    transitions = _optional_int(metrics.get("global_env_transitions"))
    metric_failures = _metric_failures(metrics)
    formal_count_contract = bool(
        tasks == FORMAL_TASKS and transitions == FORMAL_TRANSITIONS
    )
    smoke_or_diagnostic = bool(
        "smoke" in label_lower
        or tasks is None
        or tasks < FORMAL_TASKS
        or transitions is None
        or transitions < FORMAL_TRANSITIONS
    )
    metrics_passed = not metric_failures
    if schema == FIXED_BASELINE_CONTRACT_SCHEMA:
        classification = "FIXED_BASELINE_RESEARCH_AUTHORIZED"
    elif formal_count_contract and metrics_passed and authorization is True and checkpoint_sha:
        classification = "FORMAL_PASS_AUTHORIZED"
    elif formal_count_contract and metrics_passed:
        classification = "FORMAL_METRICS_PASS_BUT_UNAUTHORIZED"
    elif formal_count_contract:
        classification = "FORMAL_FAIL"
    elif smoke_or_diagnostic:
        classification = "SMOKE_OR_DIAGNOSTIC"
    else:
        classification = "INCOMPLETE_FORMAL_CONTRACT"

    buckets = metrics.get("distance_bucket_success")
    if isinstance(buckets, Mapping) and buckets:
        finite_buckets = [_optional_float(value) for value in buckets.values()]
        minimum_bucket = (
            min(value for value in finite_buckets if value is not None)
            if all(value is not None for value in finite_buckets)
            else None
        )
    else:
        minimum_bucket = _optional_float(
            metrics.get("minimum_distance_bucket_success")
        )
    return {
        "artifact": artifact_label,
        "source": source,
        "schema": schema,
        "classification": classification,
        "num_tasks": tasks,
        "global_env_transitions": transitions,
        "overall_success_rate": _optional_float(metrics.get("overall_success_rate")),
        "minimum_distance_bucket_success": minimum_bucket,
        "mean_path_ratio": _optional_float(metrics.get("mean_path_ratio")),
        "obstacle_collision_steps": _optional_int(
            metrics.get("obstacle_collision_steps")
        ),
        "boundary_contact_step_rate": _optional_float(
            metrics.get("boundary_contact_step_rate")
        ),
        "metric_failures": metric_failures,
        "formal_count_contract": formal_count_contract,
        "metrics_passed": metrics_passed,
        "authorization_passed": authorization,
        "checkpoint_sha256": checkpoint_sha,
        "formal_gate_passed": classification == "FORMAL_PASS_AUTHORIZED",
        "energy_research_authorized": bool(
            classification == "FIXED_BASELINE_RESEARCH_AUTHORIZED"
            and authorization is True
            and checkpoint_sha
        ),
        "baseline_id": payload.get("baseline_id"),
    }


def _candidate_json_name(name: str) -> bool:
    basename = Path(name).name
    return bool(
        basename
        in {
            "EVALUATION_COMPLETED.json",
            "COMPLETED.json",
            "STOPPED_NAVIGATION_NOT_READY.json",
            "navigation_result.json",
        }
        or ("navigation" in basename.lower() and basename.endswith(".json"))
    )


def scan_directory(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        relative = path.relative_to(root).as_posix()
        if not _candidate_json_name(relative):
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, Mapping):
            continue
        row = classify_navigation_payload(
            artifact_label=relative,
            payload=payload,
            source="current_workspace",
        )
        if row is not None:
            rows.append(row)
    return rows


def scan_archive(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with tarfile.open(path, mode="r:gz") as archive:
        for member in archive:
            if not member.isfile() or not _candidate_json_name(member.name):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                continue
            try:
                payload = json.loads(handle.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(payload, Mapping):
                continue
            row = classify_navigation_payload(
                artifact_label=member.name,
                payload=payload,
                source=f"archive:{path.name}",
            )
            if row is not None:
                rows.append(row)
    return rows


def build_inventory(
    *, directory_rows: Iterable[dict[str, object]], archive_rows: Iterable[dict[str, object]]
) -> dict[str, object]:
    rows = sorted(
        [*directory_rows, *archive_rows],
        key=lambda row: (str(row["source"]), str(row["artifact"])),
    )
    counts: dict[str, int] = {}
    for row in rows:
        classification = str(row["classification"])
        counts[classification] = counts.get(classification, 0) + 1
    passed = [row for row in rows if row.get("formal_gate_passed") is True]
    fixed_baselines = [
        row for row in rows if row.get("energy_research_authorized") is True
    ]
    formal_failures = [
        row for row in rows if row.get("classification") == "FORMAL_FAIL"
    ]
    result = {
        "status": (
            "PASSING_NAVIGATION_ARTIFACT_FOUND"
            if passed
            else "NO_PASSING_NAVIGATION_ARTIFACT_FOUND"
        ),
        "formal_gate_thresholds": {
            "num_tasks": FORMAL_TASKS,
            "global_env_transitions": FORMAL_TRANSITIONS,
            "overall_success_rate_minimum": 0.98,
            "minimum_distance_bucket_success_minimum": 0.95,
            "mean_path_ratio_maximum": 1.10,
            "obstacle_collision_steps": 0,
            "boundary_contact_step_rate_strict_maximum": 0.01,
            "explicit_downstream_authorization_required": True,
            "checkpoint_sha256_required": True,
        },
        "classification_counts": counts,
        "num_inventory_rows": len(rows),
        "num_formal_failures": len(formal_failures),
        "num_authorized_passes": len(passed),
        "authorized_passes": passed,
        "num_fixed_baseline_research_contracts": len(fixed_baselines),
        "fixed_baseline_research_contracts": fixed_baselines,
        "rows": rows,
        "no_r6_authorized": True,
        "next_legal_action": (
            "validate checkpoint identity and begin battery calibration preflight"
            if passed
            else (
                "continue the fixed-baseline calibration/endurance chain without "
                "claiming deployment-quality navigation PASS"
                if fixed_baselines
                else "an explicitly authorized fixed navigation research contract is required before calibration or Oracle Headroom"
            )
        ),
        "non_claim": (
            "A COMPLETED file, checkpoint filename, or metric-only PASS never "
            "authorizes downstream stages without the full formal count, metric, "
            "wrapper-authorization, and checkpoint-identity contract."
        ),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    directory_rows = scan_directory(args.artifacts_root)
    archive_rows: list[dict[str, object]] = []
    for archive in args.archive:
        archive_rows.extend(scan_archive(archive))
    report = build_inventory(
        directory_rows=directory_rows,
        archive_rows=archive_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
