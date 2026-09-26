"""Resumable validation-only PPO dock-stall intervention matrix.

The 24 jobs reuse frozen PPO checkpoints and validation maps 32–39. This
diagnostic is not a main-table learned-policy result. Use --startup-only to
write one first checkpoint and hand the run back before the full matrix.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from dual_constraint_2d.calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from dual_constraint_2d.matrix_runner import _hash_sources
from dual_constraint_2d.train_ppo import TRAIN_SEEDS, VALIDATION_MAP_IDS


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = ROOT / "artifacts" / "dual_constraint_2d_static_matrix_v2_horizon_fix_20260926" / "train"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ppo_dock_supervision_validation_20260926"


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _models() -> dict[int, Path]:
    result = {}
    for seed in TRAIN_SEEDS:
        directory = TRAIN_ROOT / f"seed_{seed}"
        status = json.loads((directory / "status.json").read_text())
        if not status.get("complete") or status.get("timesteps") != 51200:
            raise RuntimeError(f"frozen PPO seed {seed} is incomplete")
        model = directory / status["latest_model"]
        if not model.is_file():
            raise FileNotFoundError(model)
        result[seed] = model
    return result


def _manifest(models: dict[int, Path]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    sources = _hash_sources()
    for path in sorted((ROOT / "diagnostics2d").glob("*.py")):
        sources[path.relative_to(ROOT).as_posix()] = sha256(path.read_bytes()).hexdigest()
    return {
        "protocol": "ppo_dock_supervision_validation_matrix_v0",
        "interpreter": str(Path(sys.executable).resolve()),
        "validation_maps": list(VALIDATION_MAP_IDS),
        "ppo_seeds": list(TRAIN_SEEDS),
        "total_jobs": len(VALIDATION_MAP_IDS) * len(TRAIN_SEEDS),
        "model_sha256": {str(seed): sha256(path.read_bytes()).hexdigest()
                         for seed, path in models.items()},
        "calibration_sha256": sha256(
            (CALIBRATION_OUTPUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "packages": {
            "torch": torch.__version__,
            "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "source_sha256": sources,
    }


def _jobs(output: Path, models: dict[int, Path]) -> list[tuple[list[str], Path, Path, Path]]:
    jobs = []
    for seed, model in models.items():
        for map_id in VALIDATION_MAP_IDS:
            directory = output / f"seed_{seed}" / f"map_{map_id:03d}"
            command = [
                sys.executable, "-m", "diagnostics2d.dock_supervision",
                "--output", str(directory), "--map-id", str(map_id),
                "--model-path", str(model),
            ]
            jobs.append((command, directory / "run.log", directory / "status.json",
                         directory / "summary.json"))
    return jobs


def _run_child(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1",
    })
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=environment,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def run(output: Path = DEFAULT_OUTPUT, *, workers: int = 8,
        startup_only: bool = False) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between one and eight")
    output.mkdir(parents=True, exist_ok=True)
    models = _models()
    manifest = _manifest(models)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("diagnostic matrix source, model, or runtime changed")
    else:
        _atomic_json(manifest_path, manifest)
    jobs = _jobs(output, models)
    if startup_only:
        command, log, status, completion = jobs[0]
        if not completion.exists():
            _run_child([*command, "--max-new-decisions", "20"], log)
        if not status.exists() or not (status.parent / "checkpoint.pkl").exists():
            raise RuntimeError("first checkpoint missing after startup")
        report = {"phase": "startup_healthy", "complete": False,
                  "startup_job": str(status.parent.relative_to(output)),
                  "startup_decisions": json.loads(status.read_text())["policy_decisions"]}
        _atomic_json(output / "status.json", report)
        return report
    remaining = [job for job in jobs if not job[3].exists()]
    _atomic_json(output / "status.json", {
        "phase": "validation", "complete": False,
        "completed_jobs": len(jobs) - len(remaining), "planned_jobs": len(jobs),
        "workers": workers,
    })
    failures = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_run_child, command, log): (log, completion)
                       for command, log, _, completion in remaining}
            for future in as_completed(futures):
                log, completion = futures[future]
                try:
                    future.result()
                    if not completion.exists() or not json.loads(completion.read_text()).get("done"):
                        raise RuntimeError(f"child did not finish: {completion}")
                except Exception as exc:
                    failures.append(f"{log}: {exc}")
                _atomic_json(output / "status.json", {
                    "phase": "validation", "complete": False,
                    "completed_jobs": sum(job[3].exists() for job in jobs),
                    "planned_jobs": len(jobs), "workers": workers,
                    "failures": failures,
                })
        if failures:
            raise RuntimeError("diagnostic jobs failed; rerun to resume")
        report = {"phase": "complete", "complete": True,
                  "completed_jobs": len(jobs), "planned_jobs": len(jobs)}
        _atomic_json(output / "status.json", report)
        return report
    except Exception:
        _atomic_json(output / "orchestrator_error.json", {
            "traceback": traceback.format_exc(),
            "completed_jobs": sum(job[3].exists() for job in jobs),
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("diagnostic matrix is already running") from exc
        print(json.dumps(run(args.output, workers=args.workers,
                             startup_only=args.startup_only), sort_keys=True))


if __name__ == "__main__":
    main()
