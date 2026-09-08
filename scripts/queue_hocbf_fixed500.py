#!/usr/bin/env python3
"""One authorized successor: wait for final training commit, evaluate, then stop."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
STOP = False


def request_stop(_signum, _frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_state(source, expected_steps):
    if (source / "PAUSE").exists():
        return "SOURCE_PAUSED"
    if (source / "ERROR.json").exists():
        return "SOURCE_ERROR"
    status = json.loads((source / "status.json").read_text())
    if status["status"] == "TRAINING_COMPLETE_AWAITING_USER":
        if (status["adaptation_transitions"] != expected_steps
                or status["checkpoint_transitions"] != expected_steps):
            return "SOURCE_BUDGET_CHANGED"
        return "READY"
    if status["status"] == "PAUSED":
        return "SOURCE_PAUSED"
    if status["status"] != "TRAINING":
        return "SOURCE_NOT_TRAINING"
    try:
        os.kill(int(status["pid"]), 0)
    except (ProcessLookupError, KeyError):
        return "SOURCE_NOT_RUNNING"
    return "WAITING_FOR_TRAINING"


def evaluation_command(args):
    command = [sys.executable, str(ROOT / "scripts/evaluate_hocbf_correction_fixed500.py"),
               "--source", str(args.source), "--output-dir", str(args.evaluation_dir),
               "--checkpoint-transitions", str(args.checkpoint_transitions),
               "--num-envs", str(args.num_envs), "--device", args.device]
    if (args.evaluation_dir / "manifest.json").exists():
        command.append("--resume")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-transitions", type=int, default=131072)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.source = args.source.resolve()
    args.output_dir = args.output_dir.resolve()
    args.evaluation_dir = args.evaluation_dir.resolve()
    roots = [args.source, args.output_dir, args.evaluation_dir]
    if any(a == b or a in b.parents or b in a.parents
           for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("training, queue and evaluation outputs must be separate")
    if args.checkpoint_transitions <= 0 or args.num_envs <= 0:
        raise ValueError("positive budget and worker count required")
    training_manifest = args.source / "manifest.json"
    training = json.loads(training_manifest.read_text())
    if (training["contract"]["protocol"] != "recovery_sac_fresh_hocbf_correction_v1"
            or training["planned_additional_steps"] != args.checkpoint_transitions):
        raise ValueError("queue must bind the currently planned correction experiment")
    contract = {
        "source": str(args.source), "training_manifest_sha256": digest(training_manifest),
        "checkpoint_transitions": args.checkpoint_transitions,
        "evaluation_dir": str(args.evaluation_dir), "num_envs": args.num_envs, "device": args.device,
        "source_hashes": {p: digest(ROOT / p) for p in (
            "scripts/queue_hocbf_fixed500.py", "scripts/evaluate_hocbf_correction_fixed500.py",
            "scripts/evaluate_deployable_observation_v2.py",
            "scripts/resume_recovery_sac_ppo_stratified.py")},
        "authorized_successor": "paired_fixed500_only", "automatic_energy_training": False,
    }
    root = args.output_dir
    if args.resume:
        if json.loads((root / "manifest.json").read_text())["contract"] != contract:
            raise ValueError("queue resume contract mismatch")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "manifest.json", {"contract": contract, "created_unix": time.time()})
    # Held across waiting and child execution; prevents a duplicate successor.
    with (root / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)

        def status(name, **extra):
            write_json(root / "status.json", {"status": name, "pid": os.getpid(),
                       "updated_unix": time.time(), **extra})

        def stopped():
            return STOP or (root / "PAUSE").exists()

        child = None
        try:
            while not stopped():
                if (digest(training_manifest) != contract["training_manifest_sha256"]
                        or any(digest(ROOT / p) != h for p, h in contract["source_hashes"].items())):
                    raise ValueError("queued source/plan changed; explicit audit required")
                state = source_state(args.source, args.checkpoint_transitions)
                if state == "READY":
                    break
                status(state)
                if state != "WAITING_FOR_TRAINING":
                    return  # Never restart paused/failed training automatically.
                for _ in range(30):
                    if stopped():
                        break
                    time.sleep(1)
            if stopped():
                status("PAUSED")
                return
            command = evaluation_command(args)
            with (root / "evaluation_console.log").open("a") as console:
                child = subprocess.Popen(command, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT)
                status("EVALUATING", child_pid=child.pid, command=command)
                signalled = False
                while child.poll() is None:
                    if (stopped() or (args.source / "PAUSE").exists()) and not signalled:
                        child.send_signal(signal.SIGTERM)
                        signalled = True
                    time.sleep(1)
                result_path = args.evaluation_dir / "status.json"
                result = json.loads(result_path.read_text()) if result_path.exists() else {}
                if child.returncode == 0 and result.get("status") == "COMPLETE":
                    status("COMPLETE_AWAITING_USER", automatic_energy_training=False)
                elif signalled or result.get("status") == "PAUSED":
                    status("PAUSED", child_returncode=child.returncode)
                else:
                    raise RuntimeError(f"evaluation did not complete: code={child.returncode}, status={result}")
        except Exception:
            if child is not None and child.poll() is None:
                child.send_signal(signal.SIGTERM)
                child.wait()
            write_json(root / "ERROR.json", {"error": traceback.format_exc()})
            status("ERROR")
            raise


if __name__ == "__main__":
    main()
