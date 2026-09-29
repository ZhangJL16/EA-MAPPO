"""Keep the eight resumable full-window worker processes in one live session."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "checks/run_full_window_pilot.py"


def active(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/full_window_pilot_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("worker count must be 1..8 for this host")
    if not args.wait:
        parser.error("--wait required: detached child processes may be killed by session cleanup")
    if not (args.output / "manifest.json").exists():
        parser.error("initialize the manifest first")
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    launch_path = args.output / "launch.json"
    if launch_path.exists():
        earlier = json.loads(launch_path.read_text())
        if any(active(x["pid"]) for x in earlier["workers"]):
            parser.error("an earlier worker may still be active")
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = "/tmp/mpl-uav-feasibility"
    launched, processes = [], []
    for index in range(args.workers):
        log_path = logs / f"worker_{index:02d}.log"
        with log_path.open("ab") as log:
            process = subprocess.Popen(
                [sys.executable, str(RUNNER), "--output", str(args.output),
                 "--workers", str(args.workers), "--worker-index", str(index)],
                stdout=log, stderr=subprocess.STDOUT,
                cwd=PROJECT, env=environment, start_new_session=True)
        launched.append(dict(index=index, pid=process.pid, log=str(log_path)))
        processes.append(process)
    payload = dict(started_utc=datetime.now(timezone.utc).isoformat(), workers=launched,
                   interpretation="exploratory full-window screen; inspect startup health only")
    temp = launch_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, launch_path)
    print(json.dumps(payload, sort_keys=True), flush=True)
    failures = []
    for index, process in enumerate(processes):
        code = process.wait()
        if code:
            failures.append(dict(worker_index=index, exit_code=code))
    print(json.dumps(dict(event="workers_finished", failures=failures)), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
