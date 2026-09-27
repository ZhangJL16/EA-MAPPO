"""Audit and summarize the 56 frozen real-state return-timing branches."""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean, median

from .collect_shield_feedback_cpu import read_json
from .freeze_return_timing_roots import OUTPUT as ROOTS, digest


DIRECTORY = ROOTS.parent / "branches"


def collect() -> dict:
    manifest = read_json(ROOTS)
    roots_sha = digest(ROOTS)
    if len(manifest["roots"]) != 56:
        raise ValueError("unexpected frozen root count")
    rows = []
    for root in manifest["roots"]:
        path = DIRECTORY / f"{root['id']}.json"
        value = read_json(path)
        if value["root"] != root or value["roots_sha256"] != roots_sha:
            raise ValueError(f"branch provenance mismatch: {path}")
        rows.append(value)
    if len(list(DIRECTORY.glob("*.json"))) != len(rows):
        raise ValueError("unplanned branch results present")
    primary = [row for row in rows if row["root"]["kind"] == "first_takeover"]
    exploratory = [row for row in rows if row["root"]["kind"] == "early_voluntary"]
    if len(primary) != 48 or len(exploratory) != 8:
        raise ValueError("incorrect primary/exploratory split")
    paired = []
    for row in primary:
        original, voluntary = (row["branches"]["original_action"],
                               row["branches"]["voluntary_return"])
        paired.append({
            "panel": row["root"]["panel"], "seed": row["root"]["seed"],
            "map_id": row["root"]["map_id"], "decision": row["root"]["decision"],
            "voluntary_return_taken_over": voluntary["return_takeovers"] > 0,
            "both_docked": original["post_mode"] == voluntary["post_mode"] == "docked",
            "either_failure": bool(original["failure_reason"] or voluntary["failure_reason"]),
            "either_censored": original["horizon_censored"] or voluntary["horizon_censored"],
            "voluntary_minus_original_elapsed_s": voluntary["elapsed_s"] - original["elapsed_s"],
            "voluntary_minus_original_energy": voluntary["energy_spent"] - original["energy_spent"],
            "voluntary_minus_original_feedback_reward": (
                voluntary["feedback_shaped_reward"] - original["feedback_shaped_reward"]),
        })
    grouped = []
    for panel, seed in [("vectorized", 101)] + [("independent_seeds", s) for s in range(102, 107)]:
        group = [r for r in paired if (r["panel"], r["seed"]) == (panel, seed)]
        if len(group) != 8:
            raise ValueError("a paired seed is missing a validation map")
        grouped.append({
            "panel": panel, "seed": seed, "maps": len(group),
            "voluntary_return_taken_over": sum(r["voluntary_return_taken_over"] for r in group),
            "both_docked": sum(r["both_docked"] for r in group),
            "mean_delta_elapsed_s": mean(r["voluntary_minus_original_elapsed_s"] for r in group),
            "mean_delta_energy": mean(r["voluntary_minus_original_energy"] for r in group),
            "mean_delta_feedback_reward": mean(
                r["voluntary_minus_original_feedback_reward"] for r in group),
        })
    overreturn = []
    for row in exploratory:
        voluntary = row["branches"]["voluntary_return"]
        route = row["branches"]["route_four_to_task_or_dock"]
        overreturn.append({
            "map_id": row["root"]["map_id"], "decision": row["root"]["decision"],
            "root_energy_fraction": row["state"]["energy"] / row["state"]["capacity"],
            "root_route_distance_to_target": row["state"]["route_distance_to_target"],
            "voluntary_return_docked": voluntary["post_mode"] == "docked",
            "voluntary_return_taken_over": voluntary["return_takeovers"] > 0,
            "route_four_task_completed": route["task_completed"] > 0,
            "route_four_taken_over": route["return_takeovers"] > 0,
            "route_four_failure": route["failure_reason"],
            "route_four_elapsed_s": route["elapsed_s"],
        })
    return {
        "protocol": "return_timing_real_state_branch_cpu_collection_v1",
        "roots_sha256": roots_sha,
        "primary_roots": len(primary), "exploratory_roots": len(exploratory),
        "primary_summary": {
            "both_docked": sum(r["both_docked"] for r in paired),
            "voluntary_return_taken_over": sum(r["voluntary_return_taken_over"] for r in paired),
            "either_failure": sum(r["either_failure"] for r in paired),
            "either_censored": sum(r["either_censored"] for r in paired),
            "voluntary_faster": sum(r["voluntary_minus_original_elapsed_s"] < -1e-9 for r in paired),
            "voluntary_slower": sum(r["voluntary_minus_original_elapsed_s"] > 1e-9 for r in paired),
            "median_delta_elapsed_s": median(r["voluntary_minus_original_elapsed_s"] for r in paired),
            "mean_delta_elapsed_s": mean(r["voluntary_minus_original_elapsed_s"] for r in paired),
            "median_delta_energy": median(r["voluntary_minus_original_energy"] for r in paired),
            "mean_delta_energy": mean(r["voluntary_minus_original_energy"] for r in paired),
        },
        "primary_by_seed": grouped, "primary_pairs": paired,
        "exploratory_summary": {
            "route_four_task_completed": sum(r["route_four_task_completed"] for r in overreturn),
            "voluntary_return_docked": sum(r["voluntary_return_docked"] for r in overreturn),
            "voluntary_return_taken_over": sum(r["voluntary_return_taken_over"] for r in overreturn),
        },
        "exploratory_states": overreturn,
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2, sort_keys=True))
