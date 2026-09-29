"""Launch resumable initial-choice workers once, with separate logs and PIDs."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT / "checks/run_initial_choice_lb.py"


def active(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=PROJECT / "runs/initial_choice_lb_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wait", action="store_true",
                        help="keep the launcher session alive while workers run")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 8:
        parser.error("worker count must be 1..8 for this host's memory budget")
    if not (args.output / "manifest.json").exists():
        parser.error("initialize manifest before launching workers")
    logs = args.output / "logs"
    logs.mkdir(exist_ok=True)
    launch_path = args.output / "launch.json"
    if launch_path.exists():
        earlier = json.loads(launch_path.read_text())
        if any(active(x["pid"]) for x in earlier["workers"]):
            parser.error("an earlier worker process is still active")
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = "/tmp/mpl-uav-feasibility"
    launched = []
    processes = []
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
                   interpretation="resumable diagnostic; inspect only startup health")
    temp = launch_path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temp, launch_path)
    print(json.dumps(payload, sort_keys=True))
    if args.wait:
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
