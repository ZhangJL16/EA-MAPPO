"""Resumable parallel orchestration of unchanged GPU-MPC map jobs.

The policy, plant, exact CPU safety certificate, and per-job manifest are the
same as gpu2d.campaign.  Only independent jobs are scheduled concurrently.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fcntl
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import time
import traceback

import torch

from gpu2d.campaign import _jobs, _source_hashes
from gpu2d.evaluate import _manifest as evaluation_manifest


ROOT = Path(__file__).resolve().parent
SECONDS = 8 * 60 * 60
DEFAULT_OUTPUT = ROOT / 'artifacts/gpu_mpc_parallel_v4_scale_20260926'
GIB = 1024 ** 3


def linux_meminfo_bytes() -> tuple[int, int]:
    """Return physical total and currently available RAM, not including swap."""
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, _, amount = line.partition(':')
        if key in ('MemTotal', 'MemAvailable'):
            values[key] = int(amount.strip().split()[0]) * 1024
    if set(values) != {'MemTotal', 'MemAvailable'}:
        raise RuntimeError('cannot read physical memory capacity and availability')
    return values['MemTotal'], values['MemAvailable']


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def job_dir(root: Path, job: dict) -> Path:
    return (root / job['split'] / f"gpu_mpc_h{job['horizon_blocks']}" /
            f"map_{job['map_id']:03d}")


def seed_from(output: Path, source: Path) -> dict:
    if output.exists():
        raise RuntimeError(f'parallel output already exists: {output}')
    output.mkdir(parents=True)
    copied = []
    for job in _jobs():
        old = job_dir(source, job)
        if not old.exists():
            continue
        old_manifest = old / 'manifest.json'
        if old_manifest.exists() and json.loads(old_manifest.read_text()) != \
                evaluation_manifest(job['map_id'], job['horizon_blocks'], 3):
            raise RuntimeError(f'cannot migrate mismatched job manifest: {old}')
        new = job_dir(output, job)
        new.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(old, new)
        old_error = new / 'error.json'
        if old_error.exists():
            old_error.rename(new / 'error.pre_parallel.json')
        copied.append({
            'job': job,
            'source': str(old),
            'events_sha256': digest(old / 'events.jsonl')
            if (old / 'events.jsonl').exists() else None,
            'checkpoint_sha256': digest(old / 'checkpoint.pkl')
            if (old / 'checkpoint.pkl').exists() else None,
            'complete': (old / 'summary.json').exists(),
        })
    result = {'protocol': 'gpu_mpc_parallel_seed_v4',
              'source': str(source.resolve()), 'copied': copied,
              'algorithm_source_sha256': _source_hashes()}
    atomic_json(output / 'seed_migration.json', result)
    return result


def expected_manifest(workers: int) -> dict:
    return {
        'protocol': 'gpu_mpc_parallel_campaign_v4_scale',
        'duration_seconds': SECONDS,
        'workers': workers,
        'gpu_name': torch.cuda.get_device_name(0),
        'torch_version': torch.__version__,
        'interpreter': str(Path(sys.executable).resolve()),
        'jobs': _jobs(),
        'algorithm_source_sha256': _source_hashes(),
        'runner_sha256': digest(Path(__file__)),
    }


def run_job(output: Path, job: dict, deadline: float) -> dict:
    directory = job_dir(output, job)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'summary.json').exists():
        return {'job': job, 'status': 'already_complete'}
    if time() >= deadline:
        return {'job': job, 'status': 'deadline_before_start'}
    command = [sys.executable, '-m', 'gpu2d.evaluate',
               '--output', str(directory), '--map-id', str(job['map_id']),
               '--horizon-blocks', str(job['horizon_blocks']),
               '--deadline-epoch', str(deadline)]
    env = os.environ.copy()
    env.update({'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
                'OPENBLAS_NUM_THREADS': '1', 'PYTHONUNBUFFERED': '1'})
    with (directory / 'run.log').open('a', encoding='utf-8') as log:
        log.write('PARALLEL_COMMAND ' + json.dumps(command) + '\n')
        log.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return {'job': job, 'status': 'complete' if
            (directory / 'summary.json').exists() else 'deadline_checkpoint'}


def run(output: Path, *, workers: int) -> dict:
    if not 1 <= workers <= 16:
        raise ValueError('workers must be 1..16 before hardware-specific RAM cap')
    total_ram, available_ram = linux_meminfo_bytes()
    # On this 24 GB machine, the measured 16-process USS peak was 12.95 GB
    # and minimum MemAvailable was 8.35 GB.  Eight workers retained 14.84 GB.
    if total_ram < 32 * GIB and workers > 8:
        raise RuntimeError('RAM under 32 GiB: capped at 8 workers for headroom')
    if workers >= 8 and available_ram < 12 * GIB:
        raise RuntimeError('less than 12 GiB currently available: defer 8-worker run')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    output.mkdir(parents=True, exist_ok=True)
    path = output / 'manifest.json'
    expected = expected_manifest(workers)
    if path.exists():
        manifest = json.loads(path.read_text())
        stable = {key: value for key, value in manifest.items()
                  if key not in ('start_epoch', 'deadline_epoch')}
        if stable != expected:
            raise RuntimeError('parallel campaign manifest/source changed')
    else:
        started = time()
        manifest = {**expected, 'start_epoch': started,
                    'deadline_epoch': started + SECONDS}
        atomic_json(path, manifest)
    deadline = manifest['deadline_epoch']
    jobs = [job for job in manifest['jobs']
            if not (job_dir(output, job) / 'summary.json').exists()]
    active = {}
    completed_since_call = 0
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while jobs or active:
                while jobs and len(active) < workers and time() < deadline:
                    job = jobs.pop(0)
                    active[pool.submit(run_job, output, job, deadline)] = job
                completed = sum((job_dir(output, job) / 'summary.json').exists()
                                for job in manifest['jobs'])
                atomic_json(output / 'status.json', {
                    'phase': 'running', 'completed_jobs': completed,
                    'planned_jobs': len(manifest['jobs']),
                    'active_jobs': list(active.values()),
                    'unstarted_jobs': len(jobs), 'deadline_epoch': deadline,
                })
                if not active:
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    job = active.pop(future)
                    try:
                        result = future.result()
                    except Exception as exc:
                        atomic_json(output / 'error.json', {
                            'job': job, 'error': repr(exc),
                            'traceback': traceback.format_exc()})
                        raise
                    if result['status'] == 'complete':
                        completed_since_call += 1
        completed = sum((job_dir(output, job) / 'summary.json').exists()
                        for job in manifest['jobs'])
        result = {
            'phase': 'complete' if completed == len(manifest['jobs'])
            else 'time_budget_exhausted',
            'completed_jobs': completed, 'planned_jobs': len(manifest['jobs']),
            'completed_since_call': completed_since_call,
            'start_epoch': manifest['start_epoch'],
            'deadline_epoch': deadline,
        }
        from gpu2d.analyze import analyze
        result['paired_report'] = analyze(output)
        atomic_json(output / 'status.json', result)
        return result
    except Exception:
        atomic_json(output / 'orchestrator_error.json', {
            'traceback': traceback.format_exc()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed-from', type=Path)
    args = parser.parse_args()
    if args.seed_from is not None:
        seed_from(args.output, args.seed_from)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'run.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('parallel campaign already running') from exc
        print(json.dumps(run(args.output, workers=args.workers), sort_keys=True))


if __name__ == '__main__':
    main()
