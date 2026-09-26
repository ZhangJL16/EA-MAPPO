"""Verify the frozen-trace energy audit and summarize paired OAT effects."""

from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .energy_audit import NAVIGATION, PROTOCOL, ROOT


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def summarize(output: Path, navigation: Path = NAVIGATION) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
    if manifest["protocol"] != PROTOCOL or checkpoint["source_sha256"] != manifest["source_sha256"]:
        raise RuntimeError("energy audit protocol or source hash differs")
    if checkpoint["completed"] != checkpoint["total"] or checkpoint["total"] != len(manifest["jobs"]):
        raise RuntimeError("energy audit is incomplete")
    if _hash(navigation / "manifest.json") != manifest["navigation_manifest_sha256"]:
        raise RuntimeError("navigation manifest changed")

    profiles = [profile["name"] for profile in manifest["profiles"]]
    payloads = manifest["payloads_kg"]
    expected = {(payload, profile) for payload in payloads for profile in profiles}
    rows: list[dict] = []
    result_hashes: dict[str, str] = {}
    for job in manifest["jobs"]:
        nav_result = navigation / "results" / f"{job['job_id']}.json"
        if _hash(nav_result) != job["result_sha256"] or _hash(navigation / job["trace"]) != job["trace_sha256"]:
            raise RuntimeError(f"navigation source changed for {job['job_id']}")
        result_path = output / "results" / f"{job['job_id']}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["job_id"] != job["job_id"] or len(result["samples"]) != len(expected):
            raise RuntimeError(f"energy result missing or malformed for {job['job_id']}")
        if {(sample["payload_kg"], sample["profile"]) for sample in result["samples"]} != expected:
            raise RuntimeError(f"energy labels duplicate or missing for {job['job_id']}")
        rows.append(result)
        result_hashes[job["job_id"]] = _hash(result_path)

    summary: dict = {
        "protocol": PROTOCOL,
        "source_sha256": manifest["source_sha256"],
        "navigation_manifest_sha256": manifest["navigation_manifest_sha256"],
        "jobs": len(rows),
        "energy_labels": len(rows) * len(expected),
        "independent_flights": len(rows),
        "navigation_collision_count": sum(row["collision_count"] for row in rows),
        "result_sha256": result_hashes,
        "payloads": {},
    }
    for payload in payloads:
        samples = [
            {sample["profile"]: sample for sample in row["samples"] if sample["payload_kg"] == payload}
            for row in rows
        ]
        base = np.array([row["base"]["energy_wh"] for row in samples])
        base_key = str(payload)
        summary["payloads"][base_key] = {
            "base_energy_wh_mean": float(np.mean(base)),
            "base_energy_wh_quantiles_0_25_50_75_95_100": np.quantile(base, [0, .25, .5, .75, .95, 1]).tolist(),
            "base_infeasible_routes": sum(row["base"]["infeasible_steps"] > 0 for row in samples),
            "profiles": {},
        }
        for profile in profiles:
            values = np.array([row[profile]["energy_wh"] for row in samples])
            relative = 100 * (values - base) / base
            infeasible = [row[profile]["infeasible_steps"] for row in samples]
            summary["payloads"][base_key]["profiles"][profile] = {
                "mean_paired_energy_change_pct": float(np.mean(relative)),
                "min_paired_energy_change_pct": float(np.min(relative)),
                "max_paired_energy_change_pct": float(np.max(relative)),
                "infeasible_routes": sum(steps > 0 for steps in infeasible),
                "infeasible_steps": sum(infeasible),
                "violation_types_by_route": dict(Counter(
                    violation for row in samples for violation in row[profile]["violated"]
                )),
            }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/delivery_1km_energy_oat_20260925")
    parser.add_argument("--navigation", type=Path, default=NAVIGATION)
    args = parser.parse_args()
    summary = summarize(args.output, args.navigation)
    path = args.output / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("jobs", "energy_labels", "independent_flights", "navigation_collision_count")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
