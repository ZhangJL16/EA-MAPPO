"""Freeze observable first-flight SOC crossing states without future outcomes."""

from __future__ import annotations

import json

from .collect_shield_feedback_cpu import MAPS, SEEDS, evaluation_dir, read_json, training_dir
from .freeze_return_timing_roots import LOGITS, ROOT, archived_actions, digest


OUTPUT = ROOT / "evidence" / "return_online_state_branch_cpu_20260927" / "roots.json"
THRESHOLDS = (0.80, 0.60, 0.50)


def freeze() -> dict:
    roots = []
    absence = []
    for panel, seeds in SEEDS.items():
        for seed in seeds:
            train = training_dir(panel, seed, "control")
            status = read_json(train / "status.json")
            if not status["complete"] or status["timesteps"] != 25600:
                raise ValueError(f"incomplete frozen training: {train}")
            model_path = train / status["latest_model"]
            for map_id in MAPS:
                archive = evaluation_dir(panel, seed, "control", "shielded", map_id)
                events_path, summary_path = archive / "events.jsonl", archive / "summary.json"
                with events_path.open(encoding="utf-8") as stream:
                    events = [json.loads(line) for line in stream]
                actions = archived_actions(events_path)
                if (len(events) != len(actions)
                        or len(events) != read_json(summary_path)["policy_decisions"]):
                    raise ValueError(f"incomplete validation archive: {events_path}")
                logit = read_json(LOGITS / f"{panel}_seed_{seed}_map_{map_id:03d}.json")
                flight = [row for row in logit["flight_observations"]
                          if (events[row["decision"] - 1]["completed_targets"]
                              if row["decision"] else 0) == 0
                          and (events[row["decision"] - 1]["charge_events"]
                               if row["decision"] else 0) == 0]
                chosen_decisions = set()
                for threshold in THRESHOLDS:
                    selected = next((row for row in flight
                                     if row["energy_fraction"] <= threshold), None)
                    if selected is None:
                        absence.append({"panel": panel, "seed": seed, "map_id": map_id,
                                        "threshold": threshold})
                        continue
                    decision = selected["decision"]
                    if decision in chosen_decisions or actions[decision] != selected["control_action"]:
                        raise ValueError("duplicate or mismatched SOC crossing root")
                    chosen_decisions.add(decision)
                    roots.append({
                        "id": f"online_{panel}_s{seed}_m{map_id:03d}_soc{int(threshold*100):02d}",
                        "parent_id": f"{panel}_s{seed}_m{map_id:03d}",
                        "panel": panel, "seed": seed, "map_id": map_id,
                        "threshold": threshold, "decision": decision,
                        "archived_action": actions[decision],
                        "archived_soc": selected["energy_fraction"],
                        "model_sha256": digest(model_path),
                        "events_sha256": digest(events_path),
                        "summary_sha256": digest(summary_path),
                    })
    counts = {str(threshold): sum(row["threshold"] == threshold for row in roots)
              for threshold in THRESHOLDS}
    if (len(roots) != 98 or counts != {"0.8": 42, "0.6": 34, "0.5": 22}
            or len({row["id"] for row in roots}) != len(roots)
            or len(absence) != 48 * len(THRESHOLDS) - len(roots)):
        raise ValueError(f"unexpected observable crossing selection: {counts}")
    return {
        "protocol": "return_online_state_branch_cpu_v1",
        "protocol_sha256": digest(ROOT / "RETURN_ONLINE_STATE_BRANCH_PROTOCOL_20260927.md"),
        "thresholds": list(THRESHOLDS), "root_count": len(roots),
        "counts": counts, "absence": absence, "roots": roots,
    }


if __name__ == "__main__":
    payload = freeze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and read_json(OUTPUT) != payload:
        raise ValueError("frozen online-state root selection changed")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print(f"froze {payload['root_count']} observable SOC crossing states")
