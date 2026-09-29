"""Integrity-first collection for the fixed full-window diagnostic."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from run_full_window_pilot import (Config, contract, sha, stream_hash, validate_row,
                                   workload)


PROJECT = Path(__file__).resolve().parents[1]


def collect(output):
    output = Path(output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    record, frozen = contract()
    if manifest != record:
        raise RuntimeError("current source/model/regime contract differs from frozen manifest")
    manifest_hash = sha(manifest_path)
    expected_hashes = {}
    rows = {}
    for index, job in enumerate(record["jobs"]):
        path = output / "rows" / f"job_{index:04d}.json"
        if not path.exists():
            continue
        pair = job["regime"], job["seed"]
        if pair not in expected_hashes:
            config = Config(**frozen["regimes"][job["regime"]]["config"])
            expected_hashes[pair] = stream_hash(workload(job["seed"], config, frozen["layout"]))
        row = json.loads(path.read_text())
        validate_row(row, job, manifest_hash, expected_hashes[pair])
        rows[index] = row
    planned = len(record["jobs"])
    result = dict(kind=record["kind"], planned_runs=planned, validated_runs=len(rows),
                  complete=len(rows) == planned, methods=record["methods"],
                  seeds=record["seeds"], frozen_sha256=record["frozen_sha256"],
                  caveat="Exploratory paired full-T screen, not 5% risk certification.")
    if len(rows) != planned:
        return result
    cells = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows.values():
        job = row["job"]
        cells[job["regime"], job["method"]].append(row["summary"])
        paired[job["regime"], job["seed"]][job["method"]] = row["summary"]["completed"]
    result["cells"] = []
    for regime in range(27):
        for method in record["methods"]:
            summaries = cells[regime, method]
            if len(summaries) != len(record["seeds"]):
                raise RuntimeError("complete job set lacks a full cell")
            result["cells"].append(dict(
                regime=regime, method=method, n=len(summaries),
                depletion=sum(x["depletion"] for x in summaries),
                navigation_failure=sum(x["navigation_failure"] for x in summaries),
                mean_completed=sum(x["completed"] for x in summaries) / len(summaries),
                mean_completed_recharges=sum(x["completed_recharges"] for x in summaries) / len(summaries),
                mean_collisions=sum(x["collision_count"] for x in summaries) / len(summaries)))
    result["paired_mean_completed_difference_vs_reserve_sjf"] = []
    for regime in range(27):
        for method in record["methods"][:2]:
            differences = [paired[regime, seed][method] - paired[regime, seed]["reserve_sjf"]
                           for seed in record["seeds"]]
            result["paired_mean_completed_difference_vs_reserve_sjf"].append(dict(
                regime=regime, method=method, n=len(differences),
                mean_difference=sum(differences) / len(differences)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/full_window_pilot_v1")
    args = parser.parse_args()
    print(json.dumps(collect(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
