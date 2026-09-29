"""Resumable, independent full-horizon policy screen in the frozen simulator.

This diagnostic does not certify the 5% risk target or modify the old project.
One worker owns one shard; each completed episode is committed atomically.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import NamedTemporaryFile
import fcntl

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT.parents[1] / "persistent_uav_throughput_v1"
sys.path.insert(0, str(OLD))

from persistent_uav.baselines import Action, Scheduler  # noqa: E402
from persistent_uav.config import Config  # noqa: E402
from persistent_uav.environment import PersistentUAVThroughput  # noqa: E402
from persistent_uav.estimates import EstimateModel  # noqa: E402
from persistent_uav.navigation import FrozenNavigator  # noqa: E402
from persistent_uav.provenance import provenance, verify_provenance  # noqa: E402
from persistent_uav.streams import workload, stream_hash  # noqa: E402


FROZEN = OLD / "evidence/v1_2/calibration/frozen_regimes.json"
METHODS = ("full_recharge_nearest", "full_recharge_fifo", "reserve_sjf")
SEEDS = tuple(range(1302609290, 1302609300))


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                            prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temp = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def choose_full_recharge(observation, method, model):
    """No map/future inputs; return when legal, then select a task at full dock."""
    if not observation["decision_required"]:
        raise ValueError("not a decision state")
    if method not in METHODS[:2]:
        raise ValueError("unknown full-recharge method")
    if observation["can_recharge"]:
        return Action("recharge", reason="recharge_whenever_legal")
    queue = observation["queue"]
    if not queue:
        raise RuntimeError("full dock without backlog requires no action")
    if method == "full_recharge_nearest":
        key = lambda task: (float(np.linalg.norm(
            np.asarray(task["position"]) - np.asarray(observation["position"]))), task["id"])
    else:
        key = lambda task: (task["arrival"], task["id"])
    return Action("serve", min(queue, key=key)["id"], method)


def contract():
    frozen = json.loads(FROZEN.read_text())
    if not frozen.get("pilot_complete") or frozen.get("pilot_jobs") != 1000:
        raise RuntimeError("frozen calibration incomplete")
    rows = sorted(frozen["regimes"], key=lambda x: x["id"])
    if [r["id"] for r in rows] != list(range(27)):
        raise RuntimeError("expected all 27 frozen regimes")
    if len(SEEDS) != 10 or set(SEEDS) & set(range(1202609290, 1202609390)):
        raise RuntimeError("new pilot seed contract changed")
    jobs = [dict(regime=r["id"], seed=seed, method=method)
            for r in rows for seed in SEEDS for method in METHODS]
    record = dict(
        kind="full_window_fixed_policy_screen_v1",
        interpretation="descriptive independent-seed screening only; no 5% risk certificate",
        original_horizon_seconds=rows[0]["config"]["cutoff"],
        methods=list(METHODS), seeds=list(SEEDS), jobs=jobs,
        frozen_calibration=str(FROZEN), frozen_sha256=sha(FROZEN),
        runner_sha256=sha(__file__), provenance=provenance(),
        paired_streams=True, no_future_or_privileged_online_inputs=True,
        primary=dict(risk="actual energy depletion by original T",
                     utility="completed tasks by original T",
                     competing_failure="navigation timeout separately retained"),
        inference="all 27 cells and three policies reported; exploratory, no post-hoc 5% claim",
    )
    if any(Config(**r["config"]).cutoff != record["original_horizon_seconds"] for r in rows):
        raise RuntimeError("horizon differs among frozen regimes")
    return record, frozen


def ensure_manifest(output, record, *, init_only):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise RuntimeError("resume contract mismatch; use a new output directory")
    elif not init_only:
        raise RuntimeError("manifest absent; run --init-only before workers")
    else:
        if any(output.iterdir()):
            raise RuntimeError("new output directory is not empty")
        write_json_atomic(path, record)


def validate_row(row, job, manifest_hash, expected_stream_hash):
    if row.get("contract_sha256") != manifest_hash or row.get("job") != job:
        raise RuntimeError("saved row contract or job mismatch")
    summary = row.get("summary", {})
    if (summary.get("stream_sha256") != expected_stream_hash
            or summary.get("evaluation_horizon") != row.get("cutoff")
            or not summary.get("done")
            or abs(summary.get("simulation_time", -1) - row.get("cutoff", -2)) > 1e-6):
        raise RuntimeError("saved row is not a complete matching episode")
    if summary.get("depletion") != (summary.get("failure") == "energy_depletion"):
        raise RuntimeError("depletion flag mismatch")
    if summary.get("navigation_failure") != (summary.get("failure") == "navigation_failure"):
        raise RuntimeError("navigation flag mismatch")
    if not isinstance(summary.get("completed"), int) or summary["completed"] < 0:
        raise RuntimeError("invalid completed count")
    if not isinstance(row.get("events"), list) or not row["events"]:
        raise RuntimeError("missing episode event trace")


def one_job(job, frozen, manifest_hash):
    regime = frozen["regimes"][job["regime"]]
    config = Config(**regime["config"])
    tasks = workload(job["seed"], config, frozen["layout"])
    nav = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit,
                          layout=frozen["layout"])
    try:
        env = PersistentUAVThroughput(config, nav, tasks)
        model = EstimateModel(**frozen["model"])
        scheduler = Scheduler("reserve_sjf", model, 0.25) if job["method"] == "reserve_sjf" else None
        while not env.done:
            observation = env.observe()
            if env.decision_required:
                action = (scheduler.choose(observation) if scheduler is not None else
                          choose_full_recharge(observation, job["method"], model))
            else:
                action = None
            env.step(action)
        summary = env.summary()
        row = dict(contract_sha256=manifest_hash, job=job, cutoff=config.cutoff,
                   summary=summary, events=env.events)
        validate_row(row, job, manifest_hash, stream_hash(tasks))
        return row
    finally:
        nav.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/full_window_pilot_v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--stop-after-jobs", type=int, default=0)
    parser.add_argument("--init-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or not 0 <= args.worker_index < args.workers or args.stop_after_jobs < 0:
        parser.error("invalid worker count/index/stop setting")
    record, frozen = contract()
    verify_provenance(record["provenance"])
    ensure_manifest(args.output, record, init_only=args.init_only)
    if args.init_only:
        return
    manifest_hash = sha(args.output / "manifest.json")
    lock_path = args.output / f"worker_{args.worker_index:02d}.lock"
    with open(lock_path, "w", encoding="utf-8") as lock_handle:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"worker shard {args.worker_index} already running") from error
        stopping = [False]
        def stop_after_current(signum, frame):
            stopping[0] = True
        signal.signal(signal.SIGINT, stop_after_current)
        signal.signal(signal.SIGTERM, stop_after_current)
        completed_this_call = 0
        for index, job in enumerate(record["jobs"]):
            if index % args.workers != args.worker_index:
                continue
            row_path = args.output / "rows" / f"job_{index:04d}.json"
            config = Config(**frozen["regimes"][job["regime"]]["config"])
            expected_stream_hash = stream_hash(workload(job["seed"], config, frozen["layout"]))
            if row_path.exists():
                validate_row(json.loads(row_path.read_text()), job, manifest_hash, expected_stream_hash)
                continue
            row = one_job(job, frozen, manifest_hash)
            write_json_atomic(row_path, row)
            completed_this_call += 1
            write_json_atomic(args.output / f"status_worker_{args.worker_index:02d}.json",
                              dict(worker_index=args.worker_index, last_job_index=index,
                                   completed_this_call=completed_this_call, stopping=stopping[0]))
            print(json.dumps(dict(event="checkpoint", worker=args.worker_index,
                                  job_index=index, completed_this_call=completed_this_call)), flush=True)
            if stopping[0] or (args.stop_after_jobs and completed_this_call >= args.stop_after_jobs):
                return
        write_json_atomic(args.output / f"status_worker_{args.worker_index:02d}.json",
                          dict(worker_index=args.worker_index, complete=True))


if __name__ == "__main__":
    main()
