"""Collect the fixed route-4 alternatives against prior return-now branches."""

from __future__ import annotations

from collections import defaultdict
import json
from statistics import mean, median

from .collect_shield_feedback_cpu import read_json
from .freeze_return_route4_roots import OUTPUT as ROOTS_PATH, WINDOW_BRANCHES, LAGS
from .freeze_return_timing_roots import digest


DIRECTORY = ROOTS_PATH.parent / "branches"


def collect() -> dict:
    manifest = read_json(ROOTS_PATH)
    roots_hash = digest(ROOTS_PATH)
    rows = []
    parents = defaultdict(list)
    for root in manifest["roots"]:
        result = read_json(DIRECTORY / f"{root['id']}.json")
        if result["root"] != root or result["roots_sha256"] != roots_hash:
            raise ValueError(f"route-4 branch provenance differs: {root['id']}")
        old_path = WINDOW_BRANCHES / f"{root['window_root_id']}.json"
        if digest(old_path) != root["window_branch_sha256"]:
            raise ValueError(f"linked return-now branch changed: {old_path}")
        old = read_json(old_path)
        if (result["state"] != old["state"]
                or result["parent_pickle_sha256"] != old["parent_pickle_sha256"]):
            raise ValueError(f"route-4 branch did not start from identical state: {root['id']}")
        route = result["route4_branch"]
        returned = old["branches"]["return_now"]
        waited = old["branches"]["wait_until_archived_takeover"]
        row = {
            "id": root["id"], "parent_id": root["parent_first_takeover_id"],
            "panel": root["panel"], "seed": root["seed"], "map_id": root["map_id"],
            "lag_decisions": root["lag_decisions"],
            "route4_docked": route["post_mode"] == "docked",
            "return_now_docked": returned["post_mode"] == "docked",
            "route4_failure": route["failure_reason"],
            "route4_censored": route["horizon_censored"],
            "route4_completed_tasks": route["task_completed"],
            "return_now_completed_tasks": returned["task_completed"],
            "historical_wait_completed_tasks": waited["task_completed"],
            "route4_return_takeovers": route["return_takeovers"],
            "route4_task_completion_elapsed_s": route["task_completion_elapsed_s"],
            "route4_minus_return_now_elapsed_s": route["elapsed_s"] - returned["elapsed_s"],
            "route4_minus_return_now_energy": route["energy_spent"] - returned["energy_spent"],
        }
        rows.append(row)
        parents[row["parent_id"]].append(row)
    if (len(rows) != 86 or len(rows) != manifest["root_count"]
            or len(list(DIRECTORY.glob("*.json"))) != len(rows)
            or len(parents) != 44):
        raise ValueError("missing, duplicate, or unplanned route-4 result")
    by_lag = []
    for lag in LAGS:
        group = [row for row in rows if row["lag_decisions"] == lag]
        docked = [row for row in group if row["route4_docked"] and row["return_now_docked"]]
        by_lag.append({
            "lag_decisions": lag, "roots": len(group),
            "route4_docked": sum(row["route4_docked"] for row in group),
            "route4_failure": sum(bool(row["route4_failure"]) for row in group),
            "route4_censored": sum(row["route4_censored"] for row in group),
            "route4_completed_task": sum(row["route4_completed_tasks"] > 0 for row in group),
            "route4_completed_task_and_docked": sum(
                row["route4_completed_tasks"] > 0 and row["route4_docked"] for row in group),
            "route4_gained_task_over_return_now": sum(
                row["route4_completed_tasks"] > row["return_now_completed_tasks"] for row in group),
            "route4_gained_task_over_historical_wait": sum(
                row["route4_completed_tasks"] > row["historical_wait_completed_tasks"] for row in group),
            "route4_return_takeovers": sum(row["route4_return_takeovers"] > 0 for row in group),
            "both_docked": len(docked),
            "median_extra_elapsed_s_when_both_docked": (
                median(row["route4_minus_return_now_elapsed_s"] for row in docked)
                if docked else None),
            "mean_extra_elapsed_s_when_both_docked": (
                mean(row["route4_minus_return_now_elapsed_s"] for row in docked)
                if docked else None),
            "median_extra_energy_when_both_docked": (
                median(row["route4_minus_return_now_energy"] for row in docked)
                if docked else None),
        })
    return {"protocol": "return_route4_alternative_cpu_collection_v1",
            "roots_sha256": roots_hash,
            "parent_trajectories": len(parents), "root_count": len(rows),
            "by_lag": by_lag, "rows": rows}


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
