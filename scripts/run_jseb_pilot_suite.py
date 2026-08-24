from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PILOT_TOTAL_TRANSITIONS = 50_000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def finite_tree(value: object) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def compact_variant(summary: dict[str, object]) -> dict[str, object]:
    phase1 = summary["phase1"]
    navigation = phase1["final_evaluation"]
    phase2a = summary["phase2a_models"]
    phase2b = summary["phase2b"]
    phase2c = summary["phase2c_models"]
    persistent = summary["persistent_delivery"]
    point_models = {
        name: values["test"]
        for name, values in phase2c["models"].items()
    }
    return {
        "training_transition_accounting": summary["training_transition_accounting"],
        "navigation": {
            "success": navigation["overall_success_rate"],
            "path_ratio": navigation["mean_path_ratio"],
            "collision_steps": navigation["obstacle_collision_steps"],
            "intervention_rate": navigation["hocbf_intervention_step_rate"],
            "emergency_rate": navigation["hocbf_emergency_brake_step_rate"],
            "nominal_safe_rate": navigation["nominal_safe_action_rate"],
            "valid_jacobian_rate": navigation["projection_valid_step_rate"],
        },
        "phase1_bridge_training": phase1["bridge_training"],
        "phase1_bridge_runtime": phase1["bridge_metrics"],
        "phase2a_action_gradient_norm": phase2a["action_conditioned_critic"][
            "action_gradient_norm"
        ],
        "phase2b_bridge_training": phase2b["bridge_training"],
        "phase2b_last_bridge": phase2b["last_bridge_loss_metrics"],
        "energy_models": point_models,
        "selected_energy_model": phase2c["selected_by_validation"],
        "persistent": persistent,
        "finite_metrics": finite_tree(summary),
    }


def pilot_gate(ablation: str, compact: dict[str, object]) -> dict[str, object]:
    accounting = compact["training_transition_accounting"]
    navigation = compact["navigation"]
    phase1_bridge = compact["phase1_bridge_training"]
    phase2b_bridge = compact["phase2b_bridge_training"]
    checks = {
        "exact_50k_budget": bool(
            accounting["total"] == PILOT_TOTAL_TRANSITIONS
            and all(
                accounting[name] == expected
                for name, expected in {
                    "phase1": 25_000,
                    "phase2a": 5_000,
                    "phase2b": 15_000,
                    "phase2c": 5_000,
                }.items()
            )
        ),
        "finite_metrics": bool(compact["finite_metrics"]),
        "zero_obstacle_collision": int(navigation["collision_steps"]) == 0,
        "positive_training_throughput": float(
            compact["phase1_bridge_runtime"]["transitions"]
        ) > 0.0,
    }
    if ablation in {"B", "C", "D"}:
        checks["jacobian_samples_available"] = bool(
            compact["phase1_bridge_runtime"]["valid_jacobian_rate"] > 0.0
        )
        checks["shield_gradient_exercised"] = bool(
            phase1_bridge["shield_nonzero_steps"] > 0
        )
    if ablation == "D":
        checks["energy_critic_action_gradient_nonzero"] = bool(
            compact["phase2a_action_gradient_norm"] > 0.0
        )
        checks["energy_bridge_gradient_exercised"] = bool(
            phase2b_bridge["energy_nonzero_steps"] > 0
        )
    return {"checks": checks, "passed": all(checks.values())}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A/B/C/D JSEB 50k pilot suite")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    results: dict[str, object] = {}
    started = utc_now()
    for ablation in ("A", "B", "C", "D"):
        output = root / f"ablation_{ablation}"
        command = [
            sys.executable,
            "-u",
            "scripts/run_jacobian_safety_energy_1m.py",
            "--output-dir",
            str(output),
            "--pilot",
            "--ablation",
            ablation,
            "--device",
            args.device,
            "--seed",
            str(args.seed),
        ]
        if args.allow_dirty:
            command.append("--allow-dirty")
        (root / f"ablation_{ablation}_command.txt").write_text(
            " ".join(command) + "\n",
            encoding="utf-8",
        )
        with (root / f"ablation_{ablation}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        if completed.returncode != 0:
            failure = {
                "status": "FAILED",
                "returncode": completed.returncode,
                "log": str(root / f"ablation_{ablation}.log"),
            }
            results[ablation] = failure
            write_json(
                root / "FAILED.json",
                {
                    "status": "FAILED",
                    "failed_ablation": ablation,
                    "started_at": started,
                    "failed_at": utc_now(),
                    "variants": results,
                },
            )
            return completed.returncode
        summary = json.loads((output / "COMPLETED.json").read_text(encoding="utf-8"))
        compact = compact_variant(summary)
        results[ablation] = {
            "status": "COMPLETED",
            "metrics": compact,
            "gate": pilot_gate(ablation, compact),
            "artifact": str(output),
        }
        write_json(root / "pilot_progress.json", results)
    passed = all(bool(result["gate"]["passed"]) for result in results.values())
    final = {
        "status": "COMPLETED" if passed else "PILOT_GATE_FAILED",
        "started_at": started,
        "completed_at": utc_now(),
        "formal_parameters_frozen": passed,
        "variants": results,
    }
    write_json(root / "pilot_summary.json", final)
    write_json(root / ("COMPLETED.json" if passed else "FAILED.json"), final)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
