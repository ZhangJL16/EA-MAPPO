import argparse
import csv
import os
import random
import re
import subprocess
import sys


def positive_int(value):
    int_value = int(value)
    if int_value <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive integer.")
    return int_value


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run test_model.py repeatedly and report the seed with the fewest collisions."
    )
    parser.add_argument("--trials", type=positive_int, default=100)
    parser.add_argument("--alg", default="mappo_safe_Comm")
    parser.add_argument("--map", default="UAV3D")
    parser.add_argument("--episodes", type=positive_int, default=1)
    parser.add_argument("--model-dir", default="./model")
    parser.add_argument("--result-root", default="./result/test_seed_search")
    parser.add_argument("--output", default="./test_result/seed_search_results.csv")
    parser.add_argument(
        "--base-seed",
        type=int,
        default=None,
        help="If set, trial seeds are base_seed, base_seed+1, ...; otherwise random seeds are used.",
    )
    parser.add_argument(
        "--save-visuals",
        action="store_true",
        help="Keep GIF/PNG visual outputs for every trial. By default visuals are disabled for speed.",
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--uav-n-agents", type=positive_int, default=4)
    return parser.parse_args()


def build_seed_list(base_seed, trials):
    if base_seed is not None:
        return [int(base_seed) + idx for idx in range(trials)]

    rng = random.SystemRandom()
    seeds = set()
    while len(seeds) < trials:
        seeds.add(rng.randrange(1, 2**32 - 1))
    return list(seeds)


def parse_metric(output, pattern, name):
    matches = re.findall(pattern, output)
    if not matches:
        raise RuntimeError(f"Could not parse {name} from test_model.py output.")
    return float(matches[-1])


def parse_episode_metric(output, metric_name):
    episode_lines = [
        line for line in output.splitlines() if re.match(r"Episode\s+\d+:", line)
    ]
    if not episode_lines:
        raise RuntimeError("Could not find episode summary line in test_model.py output.")
    line = episode_lines[-1]
    return parse_metric(
        line,
        rf"(?<![A-Za-z_]){re.escape(metric_name)}=([-+0-9.]+)",
        metric_name,
    )


def run_one_trial(args, seed, trial_idx):
    cmd = [
        sys.executable,
        "test_model.py",
        "--alg",
        args.alg,
        "--map",
        args.map,
        "--episodes",
        str(args.episodes),
        "--model-dir",
        args.model_dir,
        "--result-root",
        args.result_root,
        "--seed",
        str(seed),
        "--uav-n-agents",
        str(args.uav_n_agents),
        "--render-mode",
        "rgb_array" if args.save_visuals else "none",
    ]
    if args.cuda:
        cmd.extend(["--cuda", "--gpu-id", str(args.gpu_id)])
    if not args.save_visuals:
        cmd.extend(["--no-render", "--no-save-frames", "--no-save-xy"])

    completed = subprocess.run(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if completed.returncode != 0:
        return {
            "trial": trial_idx,
            "seed": seed,
            "status": "failed",
            "reward": "",
            "steps": "",
            "collisions": "",
            "obstacle_collisions": "",
            "agent_collisions": "",
            "output": completed.stdout,
        }

    output = completed.stdout
    return {
        "trial": trial_idx,
        "seed": seed,
        "status": "ok",
        "reward": parse_episode_metric(output, "reward"),
        "steps": parse_episode_metric(output, "steps"),
        "collisions": parse_episode_metric(output, "collisions"),
        "obstacle_collisions": parse_episode_metric(output, "obstacle_collisions"),
        "agent_collisions": parse_episode_metric(output, "agent_collisions"),
        "output": output,
    }


def write_results(path, rows):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = [
        "trial",
        "seed",
        "status",
        "reward",
        "steps",
        "collisions",
        "obstacle_collisions",
        "agent_collisions",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main():
    args = parse_args()
    seeds = build_seed_list(args.base_seed, args.trials)
    rows = []

    for trial_idx, seed in enumerate(seeds):
        row = run_one_trial(args, seed, trial_idx)
        rows.append(row)
        if row["status"] == "ok":
            print(
                f"[{trial_idx + 1}/{args.trials}] seed={seed} "
                f"collisions={row['collisions']:.1f} "
                f"obstacle={row['obstacle_collisions']:.1f} "
                f"agent={row['agent_collisions']:.1f} "
                f"reward={row['reward']:.3f}"
            )
        else:
            print(f"[{trial_idx + 1}/{args.trials}] seed={seed} failed")

    write_results(args.output, rows)
    ok_rows = [row for row in rows if row["status"] == "ok"]
    if not ok_rows:
        print(f"No successful trials. See {args.output} and command output logs.")
        return 1

    best = min(
        ok_rows,
        key=lambda row: (
            float(row["collisions"]),
            float(row["obstacle_collisions"]),
            float(row["agent_collisions"]),
            -float(row["reward"]),
            int(row["steps"]),
        ),
    )
    print("\nBest seed by total collisions:")
    print(
        f"seed={best['seed']}, collisions={best['collisions']:.1f}, "
        f"obstacle_collisions={best['obstacle_collisions']:.1f}, "
        f"agent_collisions={best['agent_collisions']:.1f}, "
        f"reward={best['reward']:.3f}, steps={best['steps']:.0f}"
    )
    print(f"All trial results saved to {os.path.abspath(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
