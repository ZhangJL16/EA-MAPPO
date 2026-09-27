"""Summarize the frozen paired return-logit diagnostic by training seed."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, median

from .collect_shield_feedback_cpu import MAPS, SEEDS


ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "evidence" / "shield_feedback_cpu_20260927" / "logit_audit"


def summarize() -> dict:
    results = []
    for panel, seeds in SEEDS.items():
        for seed in seeds:
            maps = []
            observations = []
            for map_id in MAPS:
                path = DIRECTORY / f"{panel}_seed_{seed}_map_{map_id:03d}.json"
                row = json.loads(path.read_text(encoding="utf-8"))
                if (row["panel"], row["seed"], row["map_id"]) != (panel, seed, map_id):
                    raise ValueError(f"unexpected audit identity: {path}")
                states = row["flight_observations"]
                takeovers = [x for x in states if x["immediate_takeover"]]
                if not takeovers:
                    raise ValueError(f"expected at least one takeover: {path}")
                maps.append({
                    "map_id": map_id, "flight_states": len(states),
                    "takeover_states": len(takeovers),
                    "mean_p_return_control_at_takeover": mean(x["p_return_control"] for x in takeovers),
                    "mean_p_return_feedback_at_takeover": mean(x["p_return_feedback"] for x in takeovers),
                    "feedback_argmax_return_at_takeover": sum(x["feedback_action"] == 9 for x in takeovers),
                })
                observations.extend(states)
            takeovers = [x for x in observations if x["immediate_takeover"]]
            results.append({
                "panel": panel, "seed": seed, "maps": maps,
                "flight_states": len(observations), "takeover_states": len(takeovers),
                "median_energy_fraction_at_takeover": median(x["energy_fraction"] for x in takeovers),
                "median_p_return_control_all_flight": median(x["p_return_control"] for x in observations),
                "median_p_return_feedback_all_flight": median(x["p_return_feedback"] for x in observations),
                "median_p_return_control_at_takeover": median(x["p_return_control"] for x in takeovers),
                "median_p_return_feedback_at_takeover": median(x["p_return_feedback"] for x in takeovers),
                "mean_p_return_control_at_takeover": mean(x["p_return_control"] for x in takeovers),
                "mean_p_return_feedback_at_takeover": mean(x["p_return_feedback"] for x in takeovers),
                "feedback_argmax_return_all_flight": sum(x["feedback_action"] == 9 for x in observations),
                "feedback_argmax_return_at_takeover": sum(x["feedback_action"] == 9 for x in takeovers),
                "fraction_takeovers_with_higher_feedback_p_return": mean(
                    x["p_return_feedback"] > x["p_return_control"] for x in takeovers),
            })
    return {"protocol": "paired_return_logits_on_archived_control_trajectories_v1",
            "jobs": sum(len(seeds) for seeds in SEEDS.values()) * len(MAPS),
            "results": results}


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, sort_keys=True))
