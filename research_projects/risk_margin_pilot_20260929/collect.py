"""Validate pilot rows; reveal method summaries only when the fixed set is complete."""

from collections import defaultdict
import json
from pathlib import Path

from run import Config, PROJECT, contract, sha, stream_hash, validate_row, workload


def collect(output):
    output = Path(output)
    manifest_path = output / "manifest.json"
    record, frozen = contract()
    if json.loads(manifest_path.read_text()) != record:
        raise RuntimeError("manifest/source contract differs")
    manifest_hash = sha(manifest_path)
    rows = []
    for index, job in enumerate(record["jobs"]):
        path = output / "rows" / f"job_{index:04d}.json"
        if not path.exists():
            continue
        config = Config(**frozen["regimes"][job["regime"]]["config"])
        expected_stream_hash = stream_hash(workload(job["seed"], config, frozen["layout"]))
        row = json.loads(path.read_text())
        validate_row(row, job, manifest_hash, expected_stream_hash, config.cutoff)
        rows.append(row)
    result = dict(planned=len(record["jobs"]), validated=len(rows),
                  complete=len(rows) == len(record["jobs"]),
                  interpretation=record["interpretation"])
    if not result["complete"]:
        return result
    cells = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        job, summary = row["job"], row["summary"]
        cells[job["regime"], job["method"]].append((summary, row["diagnostics"]))
        paired[job["regime"], job["seed"]][job["method"]] = summary["completed"]
    result["cells"] = []
    result["paired_mean_completed_difference_vs_U10"] = []
    for regime in record["regimes"]:
        for method in record["methods"]:
            entries = cells[regime, method]
            if len(entries) != len(record["seeds"]):
                raise RuntimeError("missing complete method cell")
            result["cells"].append(dict(
                regime=regime, method=method, n=len(entries),
                depletion=sum(s["depletion"] for s, _ in entries),
                navigation_failure=sum(s["navigation_failure"] for s, _ in entries),
                mean_completed=sum(s["completed"] for s, _ in entries) / len(entries),
                mean_overflow=sum(s["overflow"] for s, _ in entries) / len(entries),
                mean_recharges=sum(s["completed_recharges"] for s, _ in entries) / len(entries),
                mean_collisions=sum(s["collision_count"] for s, _ in entries) / len(entries),
                forced_fallbacks=sum(d["forced_station_fallback"] for _, d in entries),
            ))
        for method in record["methods"]:
            if method == "U10":
                continue
            differences = [paired[regime, seed][method] - paired[regime, seed]["U10"]
                           for seed in record["seeds"]]
            result["paired_mean_completed_difference_vs_U10"].append(dict(
                regime=regime, method=method, n=len(differences),
                mean_difference=sum(differences) / len(differences)))
    return result


def main():
    print(json.dumps(collect(PROJECT / "runs/margin_pilot_v1"), indent=2,
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
