"""Freeze real archived states for the return-timing branching diagnostic."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re

from .collect_shield_feedback_cpu import MAPS, SEEDS, evaluation_dir, read_json, training_dir


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evidence" / "return_timing_branch_cpu_20260927" / "roots.json"
LOGITS = ROOT / "evidence" / "shield_feedback_cpu_20260927" / "logit_audit"
ACTION_ID = re.compile(rb'^\{"action_id": (\d+),')
EXPLORATORY_MAPS = (33, 36, 37, 38)


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def archived_actions(path: Path) -> list[int]:
    actions = []
    with path.open("rb") as stream:
        for line in stream:
            match = ACTION_ID.match(line)
            if match is None:
                raise ValueError(f"invalid event row: {path}")
            actions.append(int(match.group(1)))
    return actions


def make_root(kind: str, panel: str, seed: int, map_id: int,
              condition: str, decision: int, action_id: int) -> dict:
    train = training_dir(panel, seed, condition)
    status = read_json(train / "status.json")
    model = train / status["latest_model"]
    archive = evaluation_dir(panel, seed, condition, "shielded", map_id)
    events = archive / "events.jsonl"
    summary = archive / "summary.json"
    actions = archived_actions(events)
    if not (status["complete"] and status["timesteps"] == 25600
            and 0 <= decision < len(actions) and actions[decision] == action_id
            and read_json(summary)["policy_decisions"] == len(actions)):
        raise ValueError("incomplete or mismatched archived root")
    return {
        "id": f"{kind}_{panel}_s{seed}_m{map_id:03d}_d{decision:05d}",
        "kind": kind, "panel": panel, "seed": seed, "map_id": map_id,
        "source_condition": condition, "decision": decision,
        "archived_action": action_id,
        "model_sha256": digest(model),
        "events_sha256": digest(events),
        "summary_sha256": digest(summary),
    }


def freeze() -> dict:
    roots = []
    for panel, seeds in SEEDS.items():
        for seed in seeds:
            for map_id in MAPS:
                logit_file = LOGITS / f"{panel}_seed_{seed}_map_{map_id:03d}.json"
                logit = read_json(logit_file)
                takeovers = [row for row in logit["flight_observations"]
                             if row["immediate_takeover"]]
                if not takeovers:
                    raise ValueError(f"no archived takeover: {logit_file}")
                first = takeovers[0]
                roots.append(make_root("first_takeover", panel, seed, map_id,
                                       "control", first["decision"], first["control_action"]))
    for map_id in EXPLORATORY_MAPS:
        path = evaluation_dir("independent_seeds", 105, "feedback", "shielded", map_id)
        matches = [index for index, action in enumerate(archived_actions(path / "events.jsonl"))
                   if action == 9]
        if not matches:
            raise ValueError(f"no archived voluntary return on map {map_id}")
        for decision in matches[:3]:
            roots.append(make_root("early_voluntary", "independent_seeds", 105,
                                   map_id, "feedback", decision, 9))
    if (len(roots) != 56 or len({row["id"] for row in roots}) != len(roots)
            or sum(row["kind"] == "first_takeover" for row in roots) != 48
            or sum(row["kind"] == "early_voluntary" for row in roots) != 8):
        raise ValueError("unexpected root manifest size")
    return {
        "protocol": "return_timing_real_state_branch_cpu_v1",
        "protocol_sha256": digest(ROOT / "RETURN_TIMING_BRANCH_PROTOCOL_20260927.md"),
        "root_count": len(roots), "roots": roots,
    }


if __name__ == "__main__":
    payload = freeze()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT.exists() and read_json(OUTPUT) != payload:
        raise ValueError("frozen root selection changed")
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"froze {payload['root_count']} real historical decision states")
