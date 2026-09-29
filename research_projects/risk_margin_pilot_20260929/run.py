"""Resumable full-window margin comparison on the frozen UAV environment."""

import argparse
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import NamedTemporaryFile

import numpy as np


PROJECT = Path(__file__).resolve().parent
OLD = PROJECT.parents[1] / "persistent_uav_throughput_v1"
sys.path.insert(0, str(OLD))

from persistent_uav.baselines import Action  # noqa: E402
from persistent_uav.config import Config  # noqa: E402
from persistent_uav.environment import PersistentUAVThroughput  # noqa: E402
from persistent_uav.estimates import EstimateModel  # noqa: E402
from persistent_uav.navigation import FrozenNavigator  # noqa: E402
from persistent_uav.provenance import provenance, verify_provenance  # noqa: E402
from persistent_uav.streams import workload, stream_hash  # noqa: E402


FROZEN = OLD / "evidence/v1_2/calibration/frozen_regimes.json"
CALIBRATION = OLD / "evidence/v1_2/calibration/annotated_jobs.json"
METHODS = ("U0", "U5", "U10", "U15", "C")
REGIMES = (4, 13, 22)
SEEDS = tuple(range(1402609290, 1402609300))
DISTANCE_CUT = 2695.106420962591


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def horizontal_distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=float)[:2] - np.asarray(b, dtype=float)[:2]))


def margin(method, position, target, charger):
    if method == "C":
        return sum(6.5 if horizontal_distance(a, b) >= DISTANCE_CUT else 4.1
                   for a, b in ((position, target), (target, charger)))
    if method in METHODS[:4]:
        return float(method[1:])
    raise ValueError("unknown margin method")


def choose(observation, method, model):
    """Return a legal action and a small diagnostic, using only public observation."""
    if not observation["decision_required"] or method not in METHODS:
        raise ValueError("unknown method or nondecision state")
    queue = observation["queue"]
    if not queue:
        if not observation["can_recharge"]:
            raise RuntimeError("full dock with empty queue is not a decision state")
        return Action("recharge", reason="empty_queue_immediate_return"), "empty_queue_return"
    rows = []
    for task in queue:
        duration, task_energy = model.predict(observation["position"], task["position"])
        _, return_energy = model.predict(task["position"], observation["charger_position"])
        total = task_energy + return_energy
        reserve = margin(method, observation["position"], task["position"],
                         observation["charger_position"])
        rows.append((task, duration, total, reserve))
    feasible = [r for r in rows if observation["battery"] > r[2] + r[3]]
    if feasible:
        selected = min(feasible, key=lambda r: (r[1], r[0]["id"]))
        return Action("serve", selected[0]["id"], "margin_feasible_sjf"), "margin_feasible"
    if observation["can_recharge"]:
        return Action("recharge", reason="no_margin_feasible_task"), "empty_filter_return"
    selected = min(rows, key=lambda r: (r[2], r[0]["id"]))
    return Action("serve", selected[0]["id"],
                  "full_station_infeasible_estimate_fallback"), "forced_station_fallback"


def contract():
    frozen = json.loads(FROZEN.read_text())
    if not frozen.get("pilot_complete") or frozen.get("pilot_jobs") != 1000:
        raise RuntimeError("frozen calibration incomplete")
    if [frozen["regimes"][i]["id"] for i in REGIMES] != list(REGIMES):
        raise RuntimeError("preselected regime IDs changed")
    if len(SEEDS) != 10 or set(SEEDS) & set(range(1302609290, 1302609300)):
        raise RuntimeError("seed contract changed")
    jobs = [dict(regime=regime, seed=seed, method=method)
            for regime in REGIMES for seed in SEEDS for method in METHODS]
    record = dict(
        kind="state_dependent_return_margin_pilot_v1",
        interpretation="exploratory complete-window paired policy comparison; no 5% certificate",
        protocol_sha256=sha(PROJECT / "PROTOCOL.md"),
        runner_sha256=sha(__file__), frozen_sha256=sha(FROZEN),
        calibration_sha256=sha(CALIBRATION), provenance=provenance(),
        regimes=list(REGIMES), seeds=list(SEEDS), methods=list(METHODS), jobs=jobs,
        distance_cut=DISTANCE_CUT,
        margins=dict(U0=0, U5=5, U10=10, U15=15,
                     C=dict(short_leg=4.1, long_leg=6.5, per_leg=True)),
        primary=dict(risk="actual depletion by original T",
                     utility="completed tasks by original T"),
        paired_streams=True, future_or_privileged_online_inputs=False,
    )
    return record, frozen


def init(output, record):
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("new output directory must be empty")
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "manifest.json", record)


def validate_row(row, job, manifest_hash, expected_stream_hash, cutoff):
    summary = row.get("summary", {})
    if row.get("contract_sha256") != manifest_hash or row.get("job") != job:
        raise RuntimeError("row contract or job mismatch")
    if (summary.get("stream_sha256") != expected_stream_hash
            or summary.get("evaluation_horizon") != cutoff
            or not summary.get("done")
            or abs(summary.get("simulation_time", -1) - cutoff) > 1e-6):
        raise RuntimeError("row is not a complete matching episode")
    if summary.get("depletion") != (summary.get("failure") == "energy_depletion"):
        raise RuntimeError("depletion mismatch")
    if summary.get("navigation_failure") != (summary.get("failure") == "navigation_failure"):
        raise RuntimeError("navigation mismatch")
    if not isinstance(summary.get("completed"), int) or not isinstance(row.get("events"), list):
        raise RuntimeError("incomplete episode row")
    if sum(row.get("diagnostics", {}).values()) != sum(
            event["event"] == "decision" for event in row["events"]):
        raise RuntimeError("decision diagnostics do not match events")


def one_job(job, frozen, manifest_hash):
    regime = frozen["regimes"][job["regime"]]
    config = Config(**regime["config"])
    tasks = workload(job["seed"], config, frozen["layout"])
    navigator = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit,
                                layout=frozen["layout"])
    try:
        environment = PersistentUAVThroughput(config, navigator, tasks)
        model = EstimateModel(**frozen["model"])
        diagnostics = {name: 0 for name in (
            "margin_feasible", "empty_filter_return", "forced_station_fallback",
            "empty_queue_return")}
        while not environment.done:
            observation = environment.observe()
            if environment.decision_required:
                action, category = choose(observation, job["method"], model)
                diagnostics[category] += 1
            else:
                action = None
            environment.step(action)
        row = dict(contract_sha256=manifest_hash, job=job, cutoff=config.cutoff,
                   summary=environment.summary(), diagnostics=diagnostics,
                   events=environment.events)
        validate_row(row, job, manifest_hash, stream_hash(tasks), config.cutoff)
        return row
    finally:
        navigator.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/margin_pilot_v1")
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument("--stop-after-jobs", type=int, default=0)
    args = parser.parse_args()
    if args.stop_after_jobs < 0:
        parser.error("negative stop count")
    record, frozen = contract()
    verify_provenance(record["provenance"])
    if args.init_only:
        init(args.output, record)
        return
    manifest_path = args.output / "manifest.json"
    if json.loads(manifest_path.read_text()) != record:
        raise RuntimeError("resume contract mismatch; use a new output directory")
    manifest_hash = sha(manifest_path)
    with open(args.output / "worker.lock", "w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("a margin pilot worker is already running") from error
        stopping = [False]
        def stop_after_current(signum, frame):
            stopping[0] = True
        signal.signal(signal.SIGINT, stop_after_current)
        signal.signal(signal.SIGTERM, stop_after_current)
        completed = 0
        for index, job in enumerate(record["jobs"]):
            row_path = args.output / "rows" / f"job_{index:04d}.json"
            config = Config(**frozen["regimes"][job["regime"]]["config"])
            expected_hash = stream_hash(workload(job["seed"], config, frozen["layout"]))
            if row_path.exists():
                validate_row(json.loads(row_path.read_text()), job, manifest_hash,
                             expected_hash, config.cutoff)
                continue
            atomic_json(row_path, one_job(job, frozen, manifest_hash))
            completed += 1
            atomic_json(args.output / "status.json", dict(last_job_index=index,
                                                          completed_this_call=completed,
                                                          stopping=stopping[0]))
            print(json.dumps(dict(event="checkpoint", job_index=index,
                                  completed_this_call=completed)), flush=True)
            if stopping[0] or (args.stop_after_jobs and completed >= args.stop_after_jobs):
                return
        atomic_json(args.output / "status.json", dict(complete=True))


if __name__ == "__main__":
    main()
