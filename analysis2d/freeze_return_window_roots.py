"""Freeze fixed-lag states along the archived first-takeover flight legs."""

from __future__ import annotations

import json
from pathlib import Path

from .collect_shield_feedback_cpu import evaluation_dir, read_json
from .freeze_return_timing_roots import OUTPUT as FIRST_ROOTS, archived_actions, digest


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "return_timing_window_cpu_20260927" / "roots.json"
LOGITS = ROOT / "evidence" / "shield_feedback_cpu_20260927" / "logit_audit"
LAGS = (1, 5, 10, 20, 40)


def freeze() -> dict:
    first_roots = read_json(FIRST_ROOTS)
    roots = []
    for parent in first_roots["roots"]:
        if parent["kind"] != "first_takeover":
            continue
        panel, seed, map_id, decision = (parent["panel"], parent["seed"],
                                         parent["map_id"], parent["decision"])
        logit = read_json(LOGITS / f"{panel}_seed_{seed}_map_{map_id:03d}.json")
        flight = {row["decision"] for row in logit["flight_observations"]}
        event_path = evaluation_dir(panel, seed, "control", "shielded", map_id) / "events.jsonl"
        actions = archived_actions(event_path)
        with event_path.open(encoding="utf-8") as stream:
            completed = [json.loads(line)["completed_targets"] for line in stream]
        for lag in LAGS:
            start = decision - lag
            if (start < 0 or not all(index in flight for index in range(start, decision + 1))
                    or any(value != completed[decision - 1]
                           for value in completed[start:decision])):
                continue
            roots.append({
                "id": f"window_{panel}_s{seed}_m{map_id:03d}_lag{lag:02d}",
                "parent_first_takeover_id": parent["id"],
                "panel": panel, "seed": seed, "map_id": map_id,
                "decision": start, "first_takeover_decision": decision,
                "lag_decisions": lag,
                "archived_action": actions[start],
                "model_sha256": parent["model_sha256"],
                "events_sha256": parent["events_sha256"],
                "summary_sha256": parent["summary_sha256"],
            })
    expected = {1: 48, 5: 47, 10: 44, 20: 42, 40: 30}
    if (len(roots) != 211 or len({r["id"] for r in roots}) != len(roots)
            or {lag: sum(r["lag_decisions"] == lag for r in roots) for lag in LAGS}
            != expected):
        raise ValueError("unexpected fixed-lag root matrix")
    return {
        "protocol": "return_timing_window_cpu_v1",
        "protocol_sha256": digest(ROOT / "RETURN_TIMING_WINDOW_PROTOCOL_20260927.md"),
        "parent_roots_sha256": digest(FIRST_ROOTS),
        "lags": list(LAGS), "root_count": len(roots), "roots": roots,
    }


if __name__ == "__main__":
    payload = freeze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and read_json(OUTPUT) != payload:
        raise ValueError("fixed window root selection changed")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze {payload['root_count']} fixed-lag states")
