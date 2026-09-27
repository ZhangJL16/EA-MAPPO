"""Collect all pre-frozen online-state branches after completion."""

from __future__ import annotations

import json
from statistics import median

from .collect_shield_feedback_cpu import read_json
from .freeze_return_online_roots import OUTPUT as ROOTS_PATH, THRESHOLDS
from .freeze_return_timing_roots import digest


DIRECTORY = ROOTS_PATH.parent / "branches"
BRANCHES = ("return_now", "route4_then_return", "ppo_then_return")


def collect() -> dict:
    manifest = read_json(ROOTS_PATH)
    roots_hash = digest(ROOTS_PATH)
    rows = []
    for root in manifest["roots"]:
        result = read_json(DIRECTORY / f"{root['id']}.json")
        if (result["root"] != root or result["roots_sha256"] != roots_hash
                or set(result["branches"]) != set(BRANCHES)):
            raise ValueError(f"online-state branch provenance differs: {root['id']}")
        outcomes = {}
        for name in BRANCHES:
            branch = result["branches"][name]
            outcomes[name] = {
                "task_and_dock": branch["task_completed"] > 0 and branch["post_mode"] == "docked",
                "docked": branch["post_mode"] == "docked",
                "task_completed": branch["task_completed"],
                "takeover": branch["return_takeovers"] > 0,
                "failure_reason": branch["failure_reason"],
                "censored": branch["horizon_censored"],
                "elapsed_s": branch["elapsed_s"],
                "energy_spent": branch["energy_spent"],
            }
        rows.append({
            "id": root["id"], "parent_id": root["parent_id"],
            "panel": root["panel"], "seed": root["seed"], "map_id": root["map_id"],
            "threshold": root["threshold"], "decision": root["decision"],
            "soc": result["state"]["soc"],
            "target_distance_m": result["state"]["route_distance_to_target"],
            "station_distance_m": result["state"]["route_distance_to_station"],
            "outcomes": outcomes,
        })
    if (len(rows) != 98 or len(rows) != manifest["root_count"]
            or len(list(DIRECTORY.glob("*.json"))) != len(rows)
            or len({row["parent_id"] for row in rows}) != 42):
        raise ValueError("missing, duplicate, or unplanned online-state branch")
    by_threshold = []
    for threshold in THRESHOLDS:
        group = [row for row in rows if row["threshold"] == threshold]
        entry = {"threshold": threshold, "roots": len(group),
                 "parents": len({row["parent_id"] for row in group}),
                 "target_distance_median_m": median(row["target_distance_m"] for row in group)}
        for name in BRANCHES:
            entry[name] = {
                "task_and_dock": sum(row["outcomes"][name]["task_and_dock"] for row in group),
                "docked": sum(row["outcomes"][name]["docked"] for row in group),
                "takeover": sum(row["outcomes"][name]["takeover"] for row in group),
                "failure": sum(bool(row["outcomes"][name]["failure_reason"]) for row in group),
                "censored": sum(row["outcomes"][name]["censored"] for row in group),
            }
        entry["route4_rescued_over_ppo"] = sum(
            row["outcomes"]["route4_then_return"]["task_and_dock"]
            and not row["outcomes"]["ppo_then_return"]["task_and_dock"]
            for row in group)
        entry["ppo_succeeded_route4_not"] = sum(
            row["outcomes"]["ppo_then_return"]["task_and_dock"]
            and not row["outcomes"]["route4_then_return"]["task_and_dock"]
            for row in group)
        by_threshold.append(entry)
    return {
        "protocol": "return_online_state_branch_cpu_collection_v1",
        "roots_sha256": roots_hash,
        "root_count": len(rows), "parent_trajectories": 42,
        "by_threshold": by_threshold, "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
