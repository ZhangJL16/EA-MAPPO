from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agents import StatelessGeneratorPolicy
from experiments.metrics import write_csv
from experiments.paper_analysis import ABLATION_METRICS, paired_bootstrap, summarize_evaluations
from experiments.runner import _evaluate


def _copy_reference_results(source_root: Path, target_root: Path, scenarios, methods, seeds) -> None:
    for scenario in scenarios:
        for method in methods:
            for seed in seeds:
                source = source_root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                if not source.exists():
                    raise FileNotFoundError(f"missing reference evaluation: {source}")
                target = target_root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(encoding="utf-8") as handle:
                    evaluation = list(csv.DictReader(handle))
                for row in evaluation:
                    # Evaluation episodes and training episodes are not paired.
                    # Never modulo-join or overwrite evaluation outcomes with
                    # training diagnostics; analyze the two tables separately.
                    row["metric_source"] = "unaltered checked-in evaluation_metrics.csv"
                    row["training_diagnostic_source"] = str(source.parent / "trajectory_diagnostics.csv")
                write_csv(target, evaluation)


def _normalize_stateless_residual_fields(root: Path, scenarios, seeds, preserve_random: bool) -> None:
    for scenario in scenarios:
        for method in ("center_only", "random_generator"):
            for seed in seeds:
                path = root / scenario / method / f"seed_{seed}" / "evaluation_metrics.csv"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                for row in rows:
                    if method == "center_only":
                        row["mean_residual_norm"] = 0.0
                        row["mean_residual_to_center_ratio"] = 0.0
                        row["mean_cos_residual_goal"] = 0.0
                    elif not preserve_random:
                        row["mean_residual_norm"] = ""
                        row["mean_residual_to_center_ratio"] = ""
                        row["mean_cos_residual_goal"] = ""
                        row["residual_metric_note"] = "not reused when accepted-branch provenance is unavailable"
                write_csv(path, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["mission_open", "mission_obstacle"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output-dir", default="artifacts/rl_contribution")
    parser.add_argument("--reference-root", default="artifacts/comparison")
    parser.add_argument("--center-mode", default="task_oriented", choices=("task_oriented", "zero", "braking", "max_volume"))
    parser.add_argument("--reuse-stateless", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not args.reuse_stateless:
        for scenario in args.scenarios:
            for method in ("center_only", "random_generator"):
                for seed in args.seeds:
                    started = monotonic()
                    rows = _evaluate(
                        method,
                        StatelessGeneratorPolicy(method, seed),
                        scenario,
                        seed * 1000,
                        args.episodes,
                        args.center_mode,
                        "functional",
                    )
                    run_root = output / scenario / method / f"seed_{seed}"
                    write_csv(run_root / "evaluation_metrics.csv", rows)
                    (run_root / "runtime_profile.json").write_text(json.dumps({
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "evaluation_episodes": len(rows),
                        "wall_time_seconds": monotonic() - started,
                        "training": "not_applicable",
                        "evidence_scope": "synthetic empirical stateless ablation",
                    }, indent=2), encoding="utf-8")
                    print(json.dumps({
                        "method": method,
                        "scenario": scenario,
                        "seed": seed,
                        "task_success": sum(row["task_success"] for row in rows) / len(rows),
                        "return_success": sum(row["return_success"] for row in rows) / len(rows),
                    }), flush=True)
    _copy_reference_results(
        Path(args.reference_root), output, args.scenarios,
        ("generator_sac", "shield_sac", "sac"), args.seeds,
    )
    _normalize_stateless_residual_fields(output, args.scenarios, args.seeds, preserve_random=not args.reuse_stateless)
    paper = Path("artifacts/paper")
    summaries = summarize_evaluations(output, paper / "rl_contribution_ablation.csv")
    bootstrap = [
        paired_bootstrap(output, scenario, metric)
        for scenario in args.scenarios
        for metric in (
            "mission_completion_steps", "total_path_length", "total_energy_consumed",
            "episode_return", "terminal_energy",
        )
    ]
    write_csv(paper / "rl_contribution_bootstrap.csv", bootstrap)
    completeness = all(
            any(row["scenario"] == scenario and row["method"] == method for row in summaries)
            for scenario in args.scenarios
            for method in ("center_only", "random_generator", "generator_sac")
        )
    gate = {
        "RL_ABLATION_ARTIFACT_GATE": "PASS" if completeness else "FAIL",
        "claim_supported": False,
        "claim_status": "requires matched non-saturated rerun; artifact presence is not contribution evidence",
        "scenarios": args.scenarios,
        "seeds": args.seeds,
        "episodes_per_seed": args.episodes,
        "metrics": list(ABLATION_METRICS),
        "scope": "synthetic empirical ablation; no physical-safety inference",
    }
    (paper / "rl_contribution_gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
