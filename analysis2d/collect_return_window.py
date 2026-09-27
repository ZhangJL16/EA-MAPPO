"""Collect the fixed-lag return-now versus archived-wait branches."""

from __future__ import annotations

from collections import defaultdict
import json
from statistics import mean, median

from .collect_shield_feedback_cpu import read_json
from .audit_return_window_roots import audit
from .freeze_return_timing_roots import OUTPUT as FIRST_ROOTS, digest
from .freeze_return_window_roots import OUTPUT as ROOTS_PATH, LAGS


DIRECTORY = ROOTS_PATH.parent / "branches"
FIRST_BRANCHES = FIRST_ROOTS.parent / "branches"


def collect() -> dict:
    manifest = read_json(ROOTS_PATH)
    eligibility = audit()
    valid_ids = set(eligibility["valid_root_ids"])
    roots_sha = digest(ROOTS_PATH)
    rows = []
    by_parent: dict[str, list[dict]] = defaultdict(list)
    for root in manifest["roots"]:
        if root["id"] not in valid_ids:
            continue
        value = read_json(DIRECTORY / f"{root['id']}.json")
        if value["root"] != root or value["roots_sha256"] != roots_sha:
            raise ValueError(f"window branch provenance mismatch: {root['id']}")
        returned, waited = (value["branches"]["return_now"],
                            value["branches"]["wait_until_archived_takeover"])
        first = read_json(FIRST_BRANCHES / f"{root['parent_first_takeover_id']}.json")
        if first["root"]["decision"] != root["first_takeover_decision"]:
            raise ValueError(f"parent takeover root mismatch: {root['id']}")
        row = {
            "id": root["id"], "parent_first_takeover_id": root["parent_first_takeover_id"],
            "panel": root["panel"], "seed": root["seed"], "map_id": root["map_id"],
            "lag_decisions": root["lag_decisions"],
            "lead_s": first["state"]["time_s"] - value["state"]["time_s"],
            "root_energy_fraction": value["state"]["energy"] / value["state"]["capacity"],
            "return_now_taken_over": returned["return_takeovers"] > 0,
            "both_docked": returned["post_mode"] == waited["post_mode"] == "docked",
            "either_failure": bool(returned["failure_reason"] or waited["failure_reason"]),
            "either_censored": returned["horizon_censored"] or waited["horizon_censored"],
            "return_now_task_completed": returned["task_completed"],
            "waited_task_completed": waited["task_completed"],
            "delta_elapsed_s": returned["elapsed_s"] - waited["elapsed_s"],
            "delta_energy": returned["energy_spent"] - waited["energy_spent"],
            "delta_feedback_reward": (
                returned["feedback_shaped_reward"] - waited["feedback_shaped_reward"]),
        }
        rows.append(row)
        by_parent[row["parent_first_takeover_id"]].append(row)
    if (len(rows) != eligibility["valid_roots"]
            or len(list(DIRECTORY.glob("*.json"))) != len(rows)
            or len(by_parent) != 48):
        raise ValueError("missing, duplicate, or unplanned window branch")
    def summarize_lags(selected: list[dict]) -> list[dict]:
        output = []
        for lag in LAGS:
            group = [row for row in selected if row["lag_decisions"] == lag]
            output.append({
            "lag_decisions": lag, "roots": len(group),
            "median_actual_lead_s": median(r["lead_s"] for r in group),
            "both_docked": sum(r["both_docked"] for r in group),
            "either_failure": sum(r["either_failure"] for r in group),
            "either_censored": sum(r["either_censored"] for r in group),
            "return_now_taken_over": sum(r["return_now_taken_over"] for r in group),
            "return_now_completed_tasks": sum(r["return_now_task_completed"] for r in group),
            "waited_completed_tasks": sum(r["waited_task_completed"] for r in group),
            "return_now_faster": sum(r["delta_elapsed_s"] < -1e-9 for r in group),
            "median_delta_elapsed_s": median(r["delta_elapsed_s"] for r in group),
            "mean_delta_elapsed_s": mean(r["delta_elapsed_s"] for r in group),
            "median_delta_energy": median(r["delta_energy"] for r in group),
            "mean_delta_energy": mean(r["delta_energy"] for r in group),
            "median_delta_feedback_reward": median(r["delta_feedback_reward"] for r in group),
            "mean_delta_feedback_reward": mean(r["delta_feedback_reward"] for r in group),
            })
        return output

    by_lag = summarize_lags(rows)
    common_parents = {parent for parent, group in by_parent.items()
                      if {r["lag_decisions"] for r in group} == set(LAGS)}
    common_rows = [r for r in rows if r["parent_first_takeover_id"] in common_parents]
    common_by_lag = summarize_lags(common_rows)
    per_parent = []
    nonmonotonic_parent_ids = []
    for parent, group in sorted(by_parent.items()):
        accepted = [r for r in group if not r["return_now_taken_over"]]
        if any(a["lag_decisions"] < b["lag_decisions"]
               and not a["return_now_taken_over"] and b["return_now_taken_over"]
               for a in group for b in group):
            nonmonotonic_parent_ids.append(parent)
        per_parent.append({
            "parent_first_takeover_id": parent,
            "tested_lags": sorted(r["lag_decisions"] for r in group),
            "accepted_lags": sorted(r["lag_decisions"] for r in accepted),
            "first_tested_accepted_lag": min((r["lag_decisions"] for r in accepted), default=None),
        })
    return {
        "protocol": "return_timing_window_cpu_collection_v1",
        "roots_sha256": roots_sha,
        "frozen_root_count": manifest["root_count"],
        "excluded_roots": eligibility["excluded"],
        "root_count": len(rows),
        "by_lag": by_lag,
        "common_five_lag_parent_count": len(common_parents),
        "common_five_lag_by_lag": common_by_lag,
        "nonmonotonic_parent_ids": nonmonotonic_parent_ids,
        "first_tested_accepted_lag_counts": {
            str(lag): sum(p["first_tested_accepted_lag"] == lag for p in per_parent)
            for lag in LAGS
        } | {"none": sum(p["first_tested_accepted_lag"] is None for p in per_parent)},
        "per_parent": per_parent, "rows": rows,
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
