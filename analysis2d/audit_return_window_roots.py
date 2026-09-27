"""Audit same-task eligibility against archived pre/post-decision counters."""

from __future__ import annotations

import json

from .collect_shield_feedback_cpu import evaluation_dir, read_json
from .freeze_return_timing_roots import digest
from .freeze_return_window_roots import OUTPUT as ROOTS_PATH


OUTPUT = ROOTS_PATH.parent / "eligibility_audit.json"
EXPECTED_EXCLUSION = "window_independent_seeds_s102_m038_lag05"


def audit() -> dict:
    manifest = read_json(ROOTS_PATH)
    events_cache = {}
    excluded = []
    valid = []
    for root in manifest["roots"]:
        key = (root["panel"], root["seed"], root["map_id"])
        if key not in events_cache:
            event_path = evaluation_dir(*key[:2], "control", "shielded", key[2]) / "events.jsonl"
            if digest(event_path) != root["events_sha256"]:
                raise ValueError(f"archived events changed: {event_path}")
            with event_path.open(encoding="utf-8") as stream:
                events_cache[key] = [json.loads(line)["completed_targets"] for line in stream]
        counters = events_cache[key]
        before = counters[root["decision"] - 1] if root["decision"] else 0
        after = counters[root["first_takeover_decision"]]
        if before == after:
            valid.append(root["id"])
        else:
            excluded.append({
                "id": root["id"], "reason": "task_completed_within_window",
                "completed_before_root": before,
                "completed_after_takeover": after,
                "events_sha256": root["events_sha256"],
            })
    if (len(manifest["roots"]) != 211 or len(valid) != 210
            or [item["id"] for item in excluded] != [EXPECTED_EXCLUSION]):
        raise ValueError("unexpected fixed-lag eligibility correction")
    return {
        "protocol": "return_timing_window_eligibility_erratum_v1",
        "original_roots_sha256": digest(ROOTS_PATH),
        "frozen_roots": 211, "valid_roots": 210,
        "valid_root_ids": valid, "excluded": excluded,
    }


if __name__ == "__main__":
    result = audit()
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"eligibility: {result['valid_roots']}/{result['frozen_roots']}")
