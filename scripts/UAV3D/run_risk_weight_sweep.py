import argparse
import os
import shlex
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


SWEEP_VALUES = {
    "warning_penalty_weight": [0.02, 0.05, 0.1, 0.2, 0.5],
    "safety_beta": [0.1, 0.2, 0.5, 0.8, 1.2],
}


def _enable_line_buffering():
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run sensitivity sweeps for warning penalty weight and safety_beta."
    )
    parser.add_argument("--alg", default="mappo_safe_Comm", help="Algorithm name.")
    parser.add_argument("--map", default="UAV3D", help="Environment map name.")
    parser.add_argument(
        "--sweep-target",
        choices=sorted(SWEEP_VALUES.keys()),
        default="warning_penalty_weight",
        help="Which parameter to sweep.",
    )
    parser.add_argument(
        "--comm-lr",
        type=float,
        default=None,
        help="Fixed communication learning rate to pass through every run.",
    )
    parser.add_argument(
        "--safety-lr",
        type=float,
        default=None,
        help="Fixed safety learning rate to pass through every run.",
    )
    parser.add_argument(
        "--warning-penalty-weight",
        type=float,
        default=None,
        help="Fixed warning penalty weight to pass through every run when sweeping safety_beta.",
    )
    parser.add_argument(
        "--safety-beta",
        type=float,
        default=None,
        help="Fixed safety beta to pass through every run when sweeping warning_penalty_weight.",
    )
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id when CUDA is enabled.")
    parser.add_argument("--cuda", action="store_true", help="Enable CUDA for launched runs.")
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to launch main.py.",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help='Extra args appended to every run, e.g. "--n_steps 200000 --evaluate_cycle 5000".',
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running the remaining values even if one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def build_command(args, value):
    param_name = args.sweep_target
    cmd = [
        args.python,
        "main.py",
        "--alg",
        args.alg,
        "--map",
        args.map,
        f"--{param_name}",
        f"{value}",
    ]
    if args.comm_lr is not None:
        cmd.extend(["--comm_lr", f"{args.comm_lr}"])
    if args.safety_lr is not None:
        cmd.extend(["--safety_lr", f"{args.safety_lr}"])
    if args.warning_penalty_weight is not None and param_name != "warning_penalty_weight":
        cmd.extend(["--warning_penalty_weight", f"{args.warning_penalty_weight}"])
    if args.safety_beta is not None and param_name != "safety_beta":
        cmd.extend(["--safety_beta", f"{args.safety_beta}"])
    if args.cuda:
        cmd.extend(["--cuda", "True", "--gpu_id", str(args.gpu_id)])
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))
    return cmd


def get_next_log_path(alg, map_name):
    log_dir = os.path.join("train_logs", alg, map_name)
    os.makedirs(log_dir, exist_ok=True)
    log_idx = 0
    while True:
        log_path = os.path.join(log_dir, f"{map_name}_log_{log_idx}.csv")
        if not os.path.exists(log_path):
            return log_idx, log_path
        log_idx += 1


def main():
    os.chdir(REPO_ROOT)
    _enable_line_buffering()
    args = parse_args()
    sweep_values = SWEEP_VALUES[args.sweep_target]

    for run_idx, sweep_value in enumerate(sweep_values, start=1):
        log_idx, log_path = get_next_log_path(args.alg, args.map)
        cmd = build_command(args, sweep_value)
        cmd_str = " ".join(shlex.quote(part) for part in cmd)
        print(
            f"[{run_idx}/{len(sweep_values)}] "
            f"{args.sweep_target}={sweep_value} | expected_log={log_idx} ({log_path}): {cmd_str}",
            flush=True,
        )
        if args.dry_run:
            continue

        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            print(
                f"Run failed for {args.sweep_target}={sweep_value} with exit code {completed.returncode}.",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
