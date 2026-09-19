"""One debug episode, no training, sweep, or experiment-resume entry point."""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from .problem import load_problem
from .environment import Environment, PrivateTruth
from .teachers.exact_bayes import ExactBayes, known_model_value
from .policies.channel_cover import ChannelCover


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", choices=("channel_cover", "exact_bayes"), required=True)
    parser.add_argument("--truth", type=int, default=0, help="evaluator-only hypothesis index")
    parser.add_argument("--noise-seed", type=int, default=0)
    parser.add_argument("--capacity", type=int)
    parser.add_argument("--budget", type=int)
    parser.add_argument("--bundling-off", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; refusing overwrite")
    problem = load_problem(args.config)
    overrides = {name: getattr(args, name) for name in ("capacity", "budget") if getattr(args, name) is not None}
    if args.bundling_off:
        overrides["max_measurements"] = 1
    problem = replace(problem, **overrides)
    if problem.split_id != "debug-only" or problem.budget > 12:
        parser.error("this entry point is limited to debug-only budget <= 12")
    plant = Environment(problem, PrivateTruth(args.truth, args.noise_seed))
    policy = ExactBayes(problem) if args.policy == "exact_bayes" else ChannelCover(problem)
    events, reward, decision_time = [], 0, 0.
    while plant.time < problem.budget:
        state = plant.planner_state()
        start = perf_counter()
        route = policy.select(state)  # Neither truth nor plant passed to learner.
        seconds = perf_counter()-start
        decision_time += seconds
        if route is None:
            break
        result = plant.execute(route.operations)
        reward += result.reward
        events.append(dict(epoch=len(events), state=asdict(state), protocol=asdict(route),
                           result=asdict(result), decision_seconds=seconds,
                           solver_states=getattr(policy, "states", None),
                           likelihood_branches=getattr(policy, "likelihood_branches", None)))
    # Oracle is computed only after policy execution and goes to evaluator output.
    oracle = known_model_value(problem, args.truth)
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root/"src").rglob("*.py")) + [root/"pyproject.toml"]
    seals = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    report = dict(schema="fpl-debug-episode-v1", status="completed_debug_episode",
                  scientific_comparison=False, problem=asdict(problem), instance_hash=problem.instance_hash,
                  config_file_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  source_sha256=seals, policy=args.policy, evaluator_truth=args.truth,
                  noise_seed=args.noise_seed, reward=reward, known_model_value=oracle,
                  realized_shortfall=oracle-reward,
                  metric_note="Single-sample shortfall is not expected regret; it can be negative.",
                  decision_seconds=decision_time, consumed_time=plant.time, terminal="reset", events=events)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, default=str)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "decisions": len(events),
                      "consumed_time": plant.time, "terminal": "reset", "output": str(args.output)}))


if __name__ == "__main__":
    main()
