"""Freeze successful-task negative-control roots before branching."""

from __future__ import annotations

import json
from pathlib import Path

from .collect_shield_feedback_cpu import MAPS, SEEDS, evaluation_dir, read_json, training_dir
from .freeze_return_timing_roots import LOGITS, ROOT, archived_actions, digest


OUTPUT = ROOT / "evidence" / "return_success_control_cpu_20260927" / "roots.json"
LAGS = (5, 10, 20)


def freeze() -> dict:
    roots = []
    no_completion = []
    for panel, seeds in SEEDS.items():
        for seed in seeds:
            train = training_dir(panel, seed, "control")
            status = read_json(train / "status.json")
            if not status["complete"] or status["timesteps"] != 25600:
                raise ValueError(f"incomplete training checkpoint: {train}")
            model = train / status["latest_model"]
            model_hash = digest(model)
            for map_id in MAPS:
                archive = evaluation_dir(panel, seed, "control", "shielded", map_id)
                events_path, summary_path = archive / "events.jsonl", archive / "summary.json"
                actions = archived_actions(events_path)
                with events_path.open(encoding="utf-8") as stream:
                    events = [json.loads(line) for line in stream]
                if len(events) != len(actions) or read_json(summary_path)["policy_decisions"] != len(actions):
                    raise ValueError(f"incomplete archived validation: {events_path}")
                first = next((row["decision"] for row in events
                              if row["completed_targets"] >= 1), None)
                if first is None:
                    no_completion.append({"panel": panel, "seed": seed, "map_id": map_id,
                                          "events_sha256": digest(events_path)})
                    continue
                if events[first]["completed_targets"] != 1:
                    raise ValueError("first completion skipped a task counter")
                logit = read_json(LOGITS / f"{panel}_seed_{seed}_map_{map_id:03d}.json")
                flight = {row["decision"] for row in logit["flight_observations"]}
                for lag in LAGS:
                    start = first - lag
                    if start < 0 or not all(index in flight for index in range(start, first + 1)):
                        continue
                    before = events[start - 1]["completed_targets"] if start else 0
                    if (before != 0 or any(events[index]["completed_targets"] != 0
                                           for index in range(start, first))):
                        raise ValueError("selected first-completion root crosses a prior task")
                    roots.append({
                        "id": f"success_{panel}_s{seed}_m{map_id:03d}_lag{lag:02d}",
                        "parent_id": f"success_{panel}_s{seed}_m{map_id:03d}_d{first:05d}",
                        "panel": panel, "seed": seed, "map_id": map_id,
                        "decision": start, "first_completion_decision": first,
                        "lag_decisions": lag, "archived_action": actions[start],
                        "model_sha256": model_hash,
                        "events_sha256": digest(events_path),
                        "summary_sha256": digest(summary_path),
                    })
    expected = {5: 34, 10: 33, 20: 30}
    if (len(roots) != 97 or len(no_completion) != 14
            or len({r["id"] for r in roots}) != len(roots)
            or {lag: sum(r["lag_decisions"] == lag for r in roots) for lag in LAGS} != expected):
        raise ValueError("unexpected successful-task negative-control selection")
    return {
        "protocol": "return_success_control_cpu_v1",
        "protocol_sha256": digest(ROOT / "RETURN_SUCCESS_CONTROL_PROTOCOL_20260927.md"),
        "lags": list(LAGS), "root_count": len(roots),
        "no_completion_map_count": len(no_completion),
        "no_completion_maps": no_completion,
        "roots": roots,
    }


if __name__ == "__main__":
    payload = freeze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and read_json(OUTPUT) != payload:
        raise ValueError("frozen success-control root selection changed")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"froze {payload['root_count']} successful-task negative-control roots")
