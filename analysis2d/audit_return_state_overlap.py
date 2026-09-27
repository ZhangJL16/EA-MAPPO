"""Descriptive state overlap between success- and takeover-conditioned roots."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

from .collect_return_success_controls import collect as collect_success
from .collect_return_window import collect as collect_takeover
from .collect_shield_feedback_cpu import read_json


ROOT = Path(__file__).resolve().parents[1]
SUCCESS_DIR = ROOT / "evidence" / "return_success_control_cpu_20260927" / "branches"
TAKEOVER_DIR = ROOT / "evidence" / "return_timing_window_cpu_20260927" / "branches"
OUTPUT = ROOT / "evidence" / "return_success_control_cpu_20260927" / "state_overlap.json"


def pairwise_auc(success: list[float], takeover: list[float]) -> float:
    return sum((a > b) + 0.5 * (a == b) for a in success for b in takeover) / (
        len(success) * len(takeover))


def audit() -> dict:
    success = collect_success()
    takeover = collect_takeover()
    result = []
    features = {
        "soc": lambda state: state["energy"] / state["capacity"],
        "route_distance_to_target": lambda state: state["route_distance_to_target"],
        "route_distance_to_station": lambda state: state["route_distance_to_station"],
    }
    for lag in (5, 10, 20):
        s = {}
        f = {}
        for row in success["rows"]:
            if row["lag_decisions"] == lag:
                s[(row["panel"], row["seed"], row["map_id"])] = read_json(
                    SUCCESS_DIR / f"{row['id']}.json")["state"]
        for row in takeover["rows"]:
            if row["lag_decisions"] == lag:
                f[(row["panel"], row["seed"], row["map_id"])] = read_json(
                    TAKEOVER_DIR / f"{row['id']}.json")["state"]
        keys = sorted(s.keys() & f.keys())
        summaries = {}
        for name, get in features.items():
            sv = [get(state) for state in s.values()]
            fv = [get(state) for state in f.values()]
            summaries[name] = {
                "median_success": median(sv),
                "median_takeover": median(fv),
                "auc_larger_value_favors_success": pairwise_auc(sv, fv),
                "same_map_success_higher": sum(get(s[key]) > get(f[key]) for key in keys),
                "same_map_takeover_higher": sum(get(s[key]) < get(f[key]) for key in keys),
                "same_map_equal": sum(get(s[key]) == get(f[key]) for key in keys),
            }
        result.append({"lag_decisions": lag, "success_roots": len(s),
                       "takeover_roots": len(f), "same_map_pairs": len(keys),
                       "features": summaries})
    return {"protocol": "return_state_overlap_descriptive_v1",
            "interpretation": "retrospective, outcome-conditioned descriptive comparison",
            "by_lag": result}


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(audit(), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"wrote {OUTPUT}")
