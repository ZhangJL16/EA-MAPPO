"""Validate and summarize one completed retrospective return-audit flight."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from .return_audit import _atomic_json, _hash
from . import return_audit, return_audit_v2, return_audit_v3, return_audit_v4


def summarize(output: Path) -> dict:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    engine = {return_audit.PROTOCOL: return_audit,
              return_audit_v2.PROTOCOL: return_audit_v2,
              return_audit_v3.PROTOCOL: return_audit_v3,
              return_audit_v4.PROTOCOL: return_audit_v4}.get(manifest["protocol"])
    if engine is None:
        raise RuntimeError("unknown return audit protocol")
    if (manifest["source_sha256"] != engine.source_hash()
            or checkpoint["source_sha256"] != manifest["source_sha256"]
            or checkpoint["total"] != manifest["states"]):
        raise RuntimeError("audit manifest or checkpoint is inconsistent")
    if checkpoint["completed"] != manifest["states"]:
        raise RuntimeError("return audit is incomplete")

    rows = []
    for index in range(1, manifest["states"] + 1):
        path = output / "results" / f"step_{index:05d}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["step_index"] != index or len(row["samples"]) != len(manifest["profiles"]) * len(manifest["payloads_kg"]):
            raise RuntimeError(f"invalid result for step {index}")
        if "return_trace" in row:
            trace_path = output / row["return_trace"]
            if _hash(trace_path) != row["return_trace_sha256"]:
                raise RuntimeError(f"return trace changed for step {index}")
            with np.load(trace_path, allow_pickle=False) as trace:
                if (len(trace["positions"]) != row["return_steps"] + 1
                        or len(trace["velocities"]) != row["return_steps"] + 1
                        or len(trace["accelerations"]) != row["return_steps"]):
                    raise RuntimeError(f"return trace length differs for step {index}")
        rows.append(row)

    samples = [(row, sample) for row in rows for sample in row["samples"]]
    invalid = Counter((sample["payload_kg"], sample["profile"])
                      for _row, sample in samples if not sample["return_evidence_valid"])
    group_counts = [
        {"payload_kg": payload, "profile": profile["name"],
         "invalid_states": invalid[(payload, profile["name"])]}
        for payload in manifest["payloads_kg"] for profile in manifest["profiles"]
    ]
    margins = [(sample["margin_wh"], row["step_index"], sample["payload_kg"], sample["profile"])
               for row, sample in samples if sample["margin_wh"] is not None]
    minimum = min(margins) if margins else None
    steps = np.asarray([row.get("return_steps", 0) for row in rows])
    wall = np.asarray([row["query_wall_seconds"] for row in rows])
    return {
        "protocol": manifest["protocol"], "manifest_sha256": _hash(manifest_path),
        "source_sha256": manifest["source_sha256"], "job_id": manifest["job"]["job_id"],
        "map_id": manifest["job"]["map_id"], "flight_state_count": len(rows),
        "independent_navigation_flights": 1, "model_label_count": len(samples),
        "outcomes": dict(Counter(row["outcome"] for row in rows)),
        "contact_states": sum(row.get("return_collisions", 0) > 0 for row in rows),
        "controller_infeasible_states": sum(row.get("return_controller_infeasible_steps", 0) > 0 for row in rows),
        "return_intervention_steps_sum": sum(row.get("return_interventions", 0) for row in rows),
        "return_qp_retries": sum(row.get("return_qp_retry_count", 0) for row in rows),
        "model_invalid_states": len({row["step_index"] for row, sample in samples
                                     if not sample["return_evidence_valid"]}),
        "model_invalid_labels": sum(invalid.values()), "model_invalid_by_group": group_counts,
        "minimum_margin": ({"wh": minimum[0], "step_index": minimum[1],
                            "payload_kg": minimum[2], "profile": minimum[3]} if minimum else None),
        "return_steps": {"p50": float(np.quantile(steps, .5)),
                         "p95": float(np.quantile(steps, .95)), "max": int(np.max(steps))},
        "retrospective_query_wall_seconds": {
            "p50": float(np.quantile(wall, .5)), "p95": float(np.quantile(wall, .95)),
            "max": float(np.max(wall)),
            "meaning": "parallel worker wall time including 57 post-flight energy labels; not online decision latency",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.output)
    _atomic_json(args.output / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in (
        "flight_state_count", "outcomes", "model_invalid_states", "minimum_margin",
    )}, ensure_ascii=False))


if __name__ == "__main__":
    main()
