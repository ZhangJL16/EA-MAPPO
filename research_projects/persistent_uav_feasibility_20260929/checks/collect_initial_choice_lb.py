"""Validate first-option rows and report exact, scoped binomial tests.

Incomplete runs remain descriptive.  No result from this script certifies a
safe policy; a significant Z event instead excludes every legal high-level
policy for the corresponding *simulator* capacity group.
"""

import argparse
from hashlib import sha256
import json
from math import comb
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT.parents[1] / "persistent_uav_throughput_v1"
sys.path.insert(0, str(OLD))

from persistent_uav.config import Config  # noqa: E402
from persistent_uav.streams import stream_hash, workload  # noqa: E402


def binomial_upper_tail(k, n, p):
    return sum(comb(n, j) * p ** j * (1 - p) ** (n - j)
               for j in range(k, n + 1))


def one_sided_lower(k, n, alpha):
    if k == 0:
        return 0.0
    left, right = 0.0, 1.0
    for _ in range(80):
        mid = (left + right) / 2
        if binomial_upper_tail(k, n, mid) < alpha:
            left = mid
        else:
            right = mid
    return (left + right) / 2


def validate_row(path, row, manifest, frozen, manifest_sha, index, group, seed):
    path = Path(path)
    if path.name != f"job_{index:04d}.json":
        raise ValueError(f"wrong job file index: {path}")
    if (row.get("contract_sha256") != manifest_sha
            or row.get("capacity") != group["capacity"]
            or row.get("representative_regime") != group["representative_regime"]
            or row.get("regime_ids") != group["regime_ids"]
            or row.get("seed") != seed):
        raise ValueError(f"row contract mismatch: {path}")
    config = Config(**group["config"])
    tasks = workload(seed, config, frozen["layout"])
    expected_positions = [list(task.position) for task in tasks[:3]]
    expected_stream = stream_hash(tasks)
    if row.get("initial_task_positions") != expected_positions:
        raise ValueError(f"initial task mismatch: {path}")
    branches = row.get("branches")
    if not isinstance(branches, list) or not 1 <= len(branches) <= 3:
        raise ValueError(f"invalid branch count: {path}")
    if [b.get("task_id") for b in branches] != list(range(len(branches))):
        raise ValueError(f"invalid branch order: {path}")
    expected_failure = {
        "energy_depletion": ("energy_depletion", 0),
        "navigation_failure": ("navigation_failure", 0),
        "completed_first_service": (None, 1),
        "cutoff_before_first_completion": (None, 0),
    }
    for b in branches:
        outcome = b.get("outcome")
        if outcome not in expected_failure:
            raise ValueError(f"unknown endpoint outcome: {path}")
        if (b.get("failure"), b.get("completed")) != expected_failure[outcome]:
            raise ValueError(f"endpoint label disagrees with failure/completed: {path}")
        if (b.get("stream_sha256") != expected_stream
                or not isinstance(b.get("policy_steps"), int) or b["policy_steps"] < 0
                or not isinstance(b.get("collision_count"), int)
                or not 0 <= b["collision_count"] <= b["policy_steps"]
                or not 0 <= b.get("endpoint_time", -1) <= config.cutoff + 1e-6
                or not -1e-6 <= b.get("endpoint_battery", -1) <= config.capacity + 1e-6):
            raise ValueError(f"invalid branch physics summary: {path}")
        if (outcome == "cutoff_before_first_completion"
                and abs(b["endpoint_time"] - config.cutoff) > 1e-6):
            raise ValueError(f"cutoff endpoint time mismatch: {path}")
    if any(b["outcome"] != "energy_depletion" for b in branches[:-1]):
        raise ValueError(f"row failed to stop after nondepletion: {path}")
    derived_z = len(branches) == 3 and branches[-1]["outcome"] == "energy_depletion"
    if (row.get("all_legal_first_actions_deplete") != derived_z
            or row.get("skipped_task_ids") != list(range(len(branches), 3))):
        raise ValueError(f"incorrect Z or skipped branch label: {path}")
    return derived_z


def collect(output):
    manifest_bytes = (output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["kind"] != "initial_choice_unavoidable_depletion_lb_v1":
        raise ValueError("wrong diagnostic manifest")
    manifest_sha = sha256(manifest_bytes).hexdigest()
    frozen_path = Path(manifest["frozen_calibration"])
    frozen_bytes = frozen_path.read_bytes()
    if sha256(frozen_bytes).hexdigest() != manifest["frozen_sha256"]:
        raise ValueError("frozen calibration changed")
    frozen = json.loads(frozen_bytes)
    expected = {
        (group["capacity"], seed): (index, group, seed)
        for index, (group, seed) in enumerate(
            (group, seed) for group in manifest["capacity_groups"]
            for seed in manifest["seeds"])
    }
    observed = {}
    for path in sorted((output / "rows").glob("job_*.json")):
        row = json.loads(path.read_text())
        key = (row["capacity"], row["seed"])
        if key not in expected or key in observed:
            raise ValueError(f"unexpected or duplicate row: {path}")
        index, group, seed = expected[key]
        validate_row(path, row, manifest, frozen, manifest_sha, index, group, seed)
        observed[key] = row
    groups = []
    alpha_per_group = manifest["risk_alpha_total"] / len(manifest["capacity_groups"])
    for group in manifest["capacity_groups"]:
        entries = [observed[(group["capacity"], seed)] for seed in manifest["seeds"]
                   if (group["capacity"], seed) in observed]
        n, k = len(entries), sum(r["all_legal_first_actions_deplete"] for r in entries)
        complete = n == manifest["n_per_capacity"]
        groups.append(dict(capacity=group["capacity"], regime_ids=group["regime_ids"],
                           observed_initial_states=n, planned_initial_states=manifest["n_per_capacity"],
                           all_deplete_count=k, complete=complete,
                           exact_one_sided_p_value=(binomial_upper_tail(k, n, .05) if complete else None),
                           exact_one_sided_lower_95_familywise=(
                               one_sided_lower(k, n, alpha_per_group) if complete else None),
                           rejects_all_policy_feasibility_at_5pct=(
                               bool(binomial_upper_tail(k, n, .05) <= alpha_per_group)
                               if complete else None)))
    return dict(kind=manifest["kind"], source_manifest_sha256=manifest_sha,
                complete=all(x["complete"] for x in groups),
                familywise_alpha=manifest["risk_alpha_total"], groups=groups,
                caveat=("Only a complete group gives its predeclared fixed-n test. "
                        "Failure to reject does not imply a feasible policy. "
                        "All conclusions concern the frozen simulator and original initial distribution."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=PROJECT / "runs/initial_choice_lb_v1")
    args = parser.parse_args()
    print(json.dumps(collect(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
