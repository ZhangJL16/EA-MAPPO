"""Collect the frozen successful-task return-now negative controls."""

from __future__ import annotations

from collections import defaultdict
import json
from statistics import mean, median

from .collect_shield_feedback_cpu import read_json
from .freeze_return_success_roots import OUTPUT as ROOTS_PATH, LAGS
from .freeze_return_timing_roots import digest


DIRECTORY = ROOTS_PATH.parent / "branches"


def collect() -> dict:
    manifest = read_json(ROOTS_PATH)
    roots_hash = digest(ROOTS_PATH)
    rows = []
    parents = defaultdict(list)
    for root in manifest["roots"]:
        value = read_json(DIRECTORY / f"{root['id']}.json")
        if value["root"] != root or value["roots_sha256"] != roots_hash:
            raise ValueError(f"success-control provenance mismatch: {root['id']}")
        immediate = value["branches"]["return_now"]
        completed = value["branches"]["task_then_return"]
        row = {
            "id": root["id"], "parent_id": root["parent_id"],
            "panel": root["panel"], "seed": root["seed"], "map_id": root["map_id"],
            "lag_decisions": root["lag_decisions"],
            "root_soc": value["state"]["energy"] / value["state"]["capacity"],
            "return_now_docked": immediate["post_mode"] == "docked",
            "task_then_return_docked": completed["post_mode"] == "docked",
            "either_failure": bool(immediate["failure_reason"] or completed["failure_reason"]),
            "either_censored": immediate["horizon_censored"] or completed["horizon_censored"],
            "return_now_tasks": immediate["task_completed"],
            "task_then_return_tasks": completed["task_completed"],
            "return_now_taken_over": immediate["return_takeovers"] > 0,
            "task_then_return_taken_over": completed["return_takeovers"] > 0,
            "task_completion_elapsed_s": completed["first_task_completion_elapsed_s"],
            "task_then_return_minus_now_elapsed_s": completed["elapsed_s"] - immediate["elapsed_s"],
            "task_then_return_minus_now_energy": completed["energy_spent"] - immediate["energy_spent"],
            "task_then_return_minus_now_feedback_reward": (
                completed["feedback_shaped_reward"] - immediate["feedback_shaped_reward"]),
        }
        rows.append(row)
        parents[root["parent_id"]].append(row)
    if (len(rows) != manifest["root_count"] or len(rows) != 97
            or len(list(DIRECTORY.glob("*.json"))) != len(rows)
            or len(parents) != 34):
        raise ValueError("missing, duplicate, or unplanned success-control branch")
    by_lag = []
    for lag in LAGS:
        group = [row for row in rows if row["lag_decisions"] == lag]
        by_lag.append({
            "lag_decisions": lag, "roots": len(group),
            "both_docked": sum(r["return_now_docked"] and r["task_then_return_docked"] for r in group),
            "either_failure": sum(r["either_failure"] for r in group),
            "either_censored": sum(r["either_censored"] for r in group),
            "return_now_taken_over": sum(r["return_now_taken_over"] for r in group),
            "task_then_return_taken_over": sum(r["task_then_return_taken_over"] for r in group),
            "return_now_completed_tasks": sum(r["return_now_tasks"] for r in group),
            "task_then_return_completed_tasks": sum(r["task_then_return_tasks"] for r in group),
            "task_then_return_gained_task": sum(r["task_then_return_tasks"] > r["return_now_tasks"]
                                                 for r in group),
            "median_completion_elapsed_s": median(r["task_completion_elapsed_s"] for r in group),
            "median_extra_elapsed_s": median(r["task_then_return_minus_now_elapsed_s"] for r in group),
            "mean_extra_elapsed_s": mean(r["task_then_return_minus_now_elapsed_s"] for r in group),
            "median_extra_energy": median(r["task_then_return_minus_now_energy"] for r in group),
            "median_extra_feedback_reward": median(
                r["task_then_return_minus_now_feedback_reward"] for r in group),
        })
    return {
        "protocol": "return_success_control_cpu_collection_v1",
        "roots_sha256": roots_hash,
        "source_maps": 48,
        "no_completion_map_count": manifest["no_completion_map_count"],
        "source_parent_count": len(parents),
        "root_count": len(rows),
        "by_lag": by_lag,
        "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
