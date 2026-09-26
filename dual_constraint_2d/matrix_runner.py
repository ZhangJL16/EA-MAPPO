"""One resumable, hash-frozen static-full-map comparison matrix.

Run from the independent project with the recorded SB3 interpreter.  The
orchestrator starts no queue experiment and never uses test results for tuning.
Each child writes a separate checkpoint and full event log.
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
from time import perf_counter
import traceback

from .calibration import DEFAULT_OUTPUT as CALIBRATION_OUTPUT
from .train_ppo import (DEFAULT_TIMESTEPS, TEST_MAP_IDS, TRAIN_MAP_IDS,
                        TRAIN_SEEDS, VALIDATION_MAP_IDS)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "dual_constraint_2d_static_matrix_20260926"
NONLEARNING = ("route_full", "route_partial", "mpc_h1", "mpc_h2")
HASH_DIRS = ("dual_constraint_2d", "nav3d")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _hash_sources() -> dict[str, str]:
    result = {}
    for directory in HASH_DIRS:
        for path in sorted((ROOT / directory).glob("*.py")):
            relative = path.relative_to(ROOT).as_posix()
            result[relative] = sha256(path.read_bytes()).hexdigest()
    return result


def _manifest(workers: int) -> dict:
    import gymnasium
    import numpy
    import scipy
    import stable_baselines3
    import torch
    return {
        "protocol": "dual_constraint_2d_static_fullmap_matrix_v0",
        "interpreter": str(Path(sys.executable).resolve()),
        "packages": {
            "torch": torch.__version__, "numpy": numpy.__version__,
            "scipy": scipy.__version__, "gymnasium": gymnasium.__version__,
            "stable_baselines3": stable_baselines3.__version__,
        },
        "train_maps": list(TRAIN_MAP_IDS),
        "validation_maps": list(VALIDATION_MAP_IDS),
        "test_maps": list(TEST_MAP_IDS),
        "train_seeds": list(TRAIN_SEEDS),
        "training_decisions_per_seed": DEFAULT_TIMESTEPS,
        "episode_horizon_s": 600.0,
        "nonlearning_methods": list(NONLEARNING),
        "learned_method": "ppo",
        "workers": workers,
        "calibration_sha256": sha256(
            (CALIBRATION_OUTPUT / "calibration.json").read_bytes()
        ).hexdigest(),
        "source_sha256": _hash_sources(),
    }


def _run_child(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"})
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND " + json.dumps(command) + "\n")
        handle.flush()
        subprocess.run(command, cwd=ROOT, env=environment,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def _run_jobs(jobs: list[tuple[list[str], Path, Path]], workers: int) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_run_child, command, log): (log, completion)
                   for command, log, completion in jobs}
        failures = []
        for future in as_completed(futures):
            log, completion = futures[future]
            try:
                future.result()
                if not completion.exists():
                    raise RuntimeError(f"child exited without completion: {completion}")
            except Exception as exc:
                failures.append(f"{log}: {exc}")
        if failures:
            raise RuntimeError("matrix jobs failed; rerun to resume:\n" + "\n".join(failures))


def run(output: Path = DEFAULT_OUTPUT, *, workers: int = 3) -> dict:
    if not 1 <= workers <= 8:
        raise ValueError("workers must be between one and eight")
    output.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(workers)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError("matrix source, runtime, or configuration changed")
    else:
        _atomic_json(manifest_path, manifest)
    began = perf_counter()
    try:
        _atomic_json(output / "status.json", {"phase": "train", "complete": False})
        training_jobs = []
        for seed in TRAIN_SEEDS:
            directory = output / "train" / f"seed_{seed}"
            command = [sys.executable, "-m", "dual_constraint_2d.train_ppo",
                       "--output", str(directory), "--seed", str(seed)]
            training_jobs.append((command, directory / "run.log", directory / "status.json"))
        # status.json exists before completion.  Inspect its complete flag.
        training_jobs = [job for job in training_jobs if not (
            job[2].exists() and json.loads(job[2].read_text()).get("complete")
        )]
        _run_jobs(training_jobs, workers)
        models = {}
        for seed in TRAIN_SEEDS:
            directory = output / "train" / f"seed_{seed}"
            status = json.loads((directory / "status.json").read_text())
            if not status.get("complete") or status["timesteps"] != DEFAULT_TIMESTEPS:
                raise RuntimeError(f"training seed {seed} is incomplete")
            models[seed] = directory / status["latest_model"]
        for split, map_ids in (("validation", VALIDATION_MAP_IDS),
                               ("test", TEST_MAP_IDS)):
            _atomic_json(output / "status.json", {
                "phase": split, "complete": False,
                "models_sha256": {str(seed): sha256(path.read_bytes()).hexdigest()
                                  for seed, path in models.items()},
            })
            jobs = []
            for map_id in map_ids:
                for method in NONLEARNING:
                    directory = output / split / method / f"map_{map_id:03d}"
                    command = [sys.executable, "-m", "dual_constraint_2d.evaluate_matrix",
                               "--output", str(directory), "--algorithm", method,
                               "--map-id", str(map_id)]
                    jobs.append((command, directory / "run.log", directory / "summary.json"))
                for seed, model in models.items():
                    directory = output / split / f"ppo_seed_{seed}" / f"map_{map_id:03d}"
                    command = [sys.executable, "-m", "dual_constraint_2d.evaluate_matrix",
                               "--output", str(directory), "--algorithm", "ppo",
                               "--map-id", str(map_id), "--model-path", str(model)]
                    jobs.append((command, directory / "run.log", directory / "summary.json"))
            jobs = [job for job in jobs if not job[2].exists()]
            _run_jobs(jobs, workers)
        from .matrix_analysis import analyze
        result = analyze(output)
        _atomic_json(output / "status.json", {
            "phase": "complete", "complete": True,
            "wall_seconds_this_call": perf_counter() - began,
        })
        return result
    except Exception:
        _atomic_json(output / "orchestrator_error.json", {
            "traceback": traceback.format_exc(),
            "resume_command": f"{sys.executable} -m dual_constraint_2d.matrix_runner --output {output}",
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("matrix is already running") from exc
        print(json.dumps(run(args.output, workers=args.workers), sort_keys=True))


if __name__ == "__main__":
    main()
