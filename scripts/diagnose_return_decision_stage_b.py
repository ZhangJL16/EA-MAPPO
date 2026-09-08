from __future__ import annotations

import argparse
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_return_decision_stage_b import (
    evaluate_formal_seed_families,
    load_prerequisite_audit,
)


PATH_ARGUMENTS = {
    "battery_calibration_json",
    "battery_validation_json",
    "jseb_estimator_checkpoint",
    "navigation_artifact",
    "navigation_checkpoint",
    "navigation_evaluation_json",
    "oracle_headroom_json",
    "td_checkpoint",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnostic_args(
    source_launch: Path,
    output_dir: Path,
    evaluation_seeds: list[int],
    *,
    evaluation_num_envs: int | None = None,
    formal_seed_retries: int = 0,
) -> Namespace:
    payload = json.loads(source_launch.read_text(encoding="utf-8"))
    raw = payload.get("arguments")
    if not isinstance(raw, dict):
        raise TypeError("source launch must contain an arguments object")
    values = dict(raw)
    for key in PATH_ARGUMENTS:
        value = values.get(key)
        values[key] = None if value is None else Path(value)
    source_workers = int(values["evaluation_num_envs"])
    requested_workers = (
        source_workers
        if evaluation_num_envs is None
        else int(evaluation_num_envs)
    )
    if requested_workers <= 0:
        raise ValueError("diagnostic evaluation_num_envs must be positive")
    if formal_seed_retries < 0:
        raise ValueError("diagnostic formal_seed_retries must be nonnegative")
    values.update(
        {
            "output_dir": output_dir,
            "evaluation_seeds": [int(seed) for seed in evaluation_seeds],
            "evaluation_num_envs": min(
                requested_workers,
                len(evaluation_seeds),
            ),
            "formal_seed_retries": int(formal_seed_retries),
            "resume": False,
        }
    )
    return Namespace(**values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce an instrumented Stage-B seed-family failure without "
            "claiming formal statistical evidence"
        )
    )
    parser.add_argument("--source-launch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-seeds", type=int, nargs="+", required=True)
    parser.add_argument(
        "--evaluation-num-envs",
        type=int,
        help=(
            "diagnostic worker count; use one to isolate native worker crashes "
            "without changing the seed family or scientific contract"
        ),
    )
    parser.add_argument(
        "--formal-seed-retries",
        type=int,
        default=0,
        help=(
            "whole-pool retries after a native worker failure; completed atomic "
            "seed families are reused"
        ),
    )
    args = parser.parse_args(argv)
    if len(set(args.evaluation_seeds)) != len(args.evaluation_seeds):
        parser.error("--evaluation-seeds must be unique")
    if args.evaluation_num_envs is not None and args.evaluation_num_envs <= 0:
        parser.error("--evaluation-num-envs must be positive")
    if args.formal_seed_retries < 0:
        parser.error("--formal-seed-retries must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> None:
    cli = parse_args(argv)
    if cli.output_dir.exists():
        raise FileExistsError(
            f"diagnostic output directory already exists: {cli.output_dir}"
        )
    cli.output_dir.mkdir(parents=True)
    args = diagnostic_args(
        cli.source_launch,
        cli.output_dir,
        cli.evaluation_seeds,
        evaluation_num_envs=cli.evaluation_num_envs,
        formal_seed_retries=cli.formal_seed_retries,
    )
    manifest = {
        "protocol": "stage_b_oracle_rollout_failure_diagnostic_v1",
        "formal_evidence": False,
        "source_launch": str(cli.source_launch.resolve()),
        "source_launch_sha256": file_sha256(cli.source_launch),
        "evaluation_seeds": list(args.evaluation_seeds),
        "candidate_grid": {
            "methods": list(args.methods),
            "soc_thresholds": list(args.soc_thresholds),
            "reserve_fractions": list(args.reserve_fractions),
        },
        "oracle_max_policy_steps": int(args.oracle_max_policy_steps),
        "maximum_policy_steps": int(args.maximum_policy_steps),
        "evaluation_num_envs": int(args.evaluation_num_envs),
        "torch_threads": int(args.torch_threads),
    }
    (cli.output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        prerequisite = load_prerequisite_audit(args)
        (cli.output_dir / "prerequisite_audit.json").write_text(
            json.dumps(prerequisite.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not prerequisite.passed:
            raise RuntimeError("diagnostic source no longer passes Gate-B prerequisites")
        families = evaluate_formal_seed_families(args)
        (cli.output_dir / "COMPLETED_DIAGNOSTIC.json").write_text(
            json.dumps(
                {
                    **manifest,
                    "status": "COMPLETED_WITHOUT_REPRODUCING_FAILURE",
                    "completed_seed_families": len(families),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        failure = {
            **manifest,
            "status": "REPRODUCED_FAILURE",
            "error_type": type(error).__name__,
            "error": str(error),
            "seed_failure_files": sorted(
                str(path.relative_to(cli.output_dir))
                for path in (cli.output_dir / "seed_failures").glob("*.json")
            ),
        }
        for attribute in ("stage_b_context", "rollout_diagnostics"):
            context = getattr(error, attribute, None)
            if context is not None:
                failure[attribute] = context
        (cli.output_dir / "FAILED_DIAGNOSTIC.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
