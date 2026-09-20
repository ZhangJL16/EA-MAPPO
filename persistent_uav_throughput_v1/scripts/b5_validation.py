"""Explicit B5-only launch, bounded startup check, and manual final collection."""
import argparse
import itertools
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
WORK = ROOT.parent
os.environ['PERSISTENT_UAV_LEGACY_ROOT'] = str(ROOT)
sys.path.insert(0, str(PROJECT))

from persistent_uav.config import VALIDATION_SEEDS
from persistent_uav.evaluation import aggregate, calibration_compatibility
from persistent_uav.provenance import verify_provenance
from persistent_uav.storage import exclusive_run, restore, sha, write_json

THREADS = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
THREADS.update(CUDA_VISIBLE_DEVICES='', ASCEND_RT_VISIBLE_DEVICES='')


def key(row):
    return tuple(row[k] for k in ('regime', 'method', 'threshold', 'seed'))


def expected_jobs():
    return [dict(regime=r, method='reserve_sjf', threshold=.25, seed=s)
            for r, s in itertools.product(range(27), VALIDATION_SEEDS)]


def launch(output, compatibility, workers):
    if not 1 <= workers <= 27:
        raise ValueError('worker count must be in 1..27')
    frozen = PROJECT / 'evidence/v1_2/calibration/frozen_regimes.json'
    calibration_compatibility(frozen, json.loads(frozen.read_text()), compatibility)
    # Never reuse or overwrite a previous production output.
    output.mkdir(parents=True, exist_ok=False)
    records = []
    expected = expected_jobs()
    plan = dict(kind='B5_ONLY_VALIDATION', expected_jobs=expected, workers=workers,
                estimator_margin=0., empty_queue_soc_threshold=.25,
                frozen_sha256=sha(frozen), compatibility=str(compatibility),
                compatibility_sha256=sha(compatibility), launcher_sha256=sha(Path(__file__)),
                thread_environment=THREADS, created_unix=time.time(),
                interpretation='Diagnostic only; no automatic evaluation, training or kill test',
                shards=[dict(worker=i, regimes=list(range(i, 27, workers))) for i in range(workers)])
    assigned = [key(j) for s in plan['shards'] for j in expected if j['regime'] in s['regimes']]
    assert len(assigned) == len(set(assigned)) == 270
    assert set(assigned) == {key(j) for j in expected}
    write_json(output / 'plan.json', plan)
    env = dict(os.environ, **THREADS)
    for shard in plan['shards']:
        folder = output / f"worker_{shard['worker']:02d}"
        cmd = [str(WORK / '.venv/bin/python'), '-u', '-m', 'persistent_uav.cli', 'baselines',
               '--frozen', str(frozen), '--calibration-compatibility', str(compatibility),
               '--output', str(folder), '--split', 'validation', '--methods', 'reserve_sjf',
               '--regimes', ','.join(map(str, shard['regimes'])), '--checkpoint-policy-steps', '100']
        log = output / (folder.name + '.log')
        with log.open('xb') as stream:
            process = subprocess.Popen(cmd, cwd=PROJECT, env=env, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        records.append(dict(worker=shard['worker'], pid=process.pid, command=cmd,
                            output=str(folder), log=str(log), started_unix=time.time()))
        write_json(output / 'launch.json', dict(workers=records))
    prefix = 'cd ' + shlex.quote(str(PROJECT)) + ' && env ' + shlex.join(
        [f'{k}={v}' for k, v in dict(THREADS, PERSISTENT_UAV_LEGACY_ROOT=str(ROOT)).items()])
    (output / 'resume_commands.txt').write_text('\n'.join(
        prefix + ' ' + shlex.join(r['command'] + ['--resume']) for r in records) + '\n')
    print(f'Launched {workers} CPU workers for exactly 270 B5 validation jobs.', flush=True)


def check(output):
    """One bounded first-checkpoint check, not monitoring until completion."""
    import numpy as np
    plan = json.loads((output / 'plan.json').read_text())
    processes = json.loads((output / 'launch.json').read_text())['workers']
    assert len(processes) == plan['workers']
    checks = {}
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        for process in processes:
            worker = process['worker']
            if worker in checks:
                continue
            folder = Path(process['output'])
            if (folder / 'error.json').exists():
                raise RuntimeError((folder / 'error.json').read_text())
            proc = Path(f"/proc/{process['pid']}")
            actual = [x.decode() for x in (proc / 'cmdline').read_bytes().split(b'\0') if x]
            assert actual == process['command'], ('unexpected or exited worker', worker)
            if not (folder / 'status.json').exists():
                continue
            status = json.loads((folder / 'status.json').read_text())
            if status['status'] != 'running' or status['total_policy_steps'] < 100:
                continue
            try:
                state = restore(folder)
            except FileNotFoundError:  # Atomic generation may rotate during a read.
                continue
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert state['contract'] == manifest
            verify_provenance(manifest['provenance'])
            assigned = [j for j in plan['expected_jobs'] if j['regime'] in plan['shards'][worker]['regimes']]
            assert manifest['jobs'] == assigned and manifest['split'] == 'validation'
            assert [key(r) for r in state['rows']] == [key(j) for j in assigned[:len(state['rows'])]]
            environ = dict(x.decode().split('=', 1) for x in (proc / 'environ').read_bytes().split(b'\0') if b'=' in x)
            assert all(environ[k] == v for k, v in THREADS.items())
            threads = int(next(line.split(':')[1] for line in (proc / 'status').read_text().splitlines()
                               if line.startswith('Threads:')))
            assert threads == 1
            assert manifest['oracle_calls'] == status['training_updates'] == 0
            if state['env'] is not None:
                assert np.isfinite(state['env'].nav.observation()).all()
                decisions = [e for e in state['env'].events if e['event'] == 'decision']
                assert decisions and all('reserve_prediction' in e for e in decisions)
                for e in decisions:
                    audit = e['reserve_prediction']
                    assert audit['margin_added'] == 0. and audit['strict_feasibility']
                    for c in audit['candidates']:
                        margin = audit['starting_battery'] - c['predicted_task_energy'] - c['predicted_return_energy']
                        assert abs(margin - c['predicted_reserve_margin']) < 1e-8
            checks[worker] = dict(worker=worker, pid=process['pid'], threads=threads,
                                  manifest_sha256=sha(folder / 'manifest.json'),
                                  policy_steps=state['total_policy_steps'], completed_runs=len(state['rows']),
                                  checksum_and_provenance_valid=True, telemetry_present=True,
                                  checked_unix=time.time())
        if len(checks) == plan['workers']:
            result = dict(passed=True, workers=[checks[i] for i in sorted(checks)],
                          total_planned_runs=270, training_updates=0, oracle_calls=0,
                          note='First-checkpoint health only; no completion monitoring.')
            write_json(output / 'startup_health.json', result)
            print(json.dumps(result), flush=True)
            return
        time.sleep(.5)
    raise RuntimeError('First-checkpoint deadline reached; inspect logs and verified worker PIDs')


def collect(output):
    plan = json.loads((output / 'plan.json').read_text())
    health = json.loads((output / 'startup_health.json').read_text())
    rows = []
    for checked in health['workers']:
        folder = output / f"worker_{checked['worker']:02d}"
        with exclusive_run(folder):
            assert sha(folder / 'manifest.json') == checked['manifest_sha256']
            assert json.loads((folder / 'status.json').read_text())['status'] == 'complete'
            state = restore(folder)
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert state['contract'] == manifest
            result = json.loads((folder / 'results.json').read_text())
            assert state['rows'] == result
            assert [key(r) for r in result] == [key(j) for j in manifest['jobs']]
            for i, row in enumerate(result):
                artifact = json.loads((folder / f'run_{i:05d}.json').read_text())
                assert artifact['summary'] == row and artifact['decision_diagnostics']
            rows.extend(result)
    assert len(rows) == len({key(r) for r in rows}) == 270
    assert {key(r) for r in rows} == {key(j) for j in plan['expected_jobs']}
    rows.sort(key=key)
    summaries = aggregate(rows)
    assert len(summaries) == 27 and all(r['n'] == 10 for r in summaries)
    target = output / 'aggregate'
    target.mkdir(exist_ok=False)
    write_json(target / 'results.json', rows)
    write_json(target / 'summary.json', summaries)
    write_json(target / 'eligibility.json', dict(empirical_budget=.05,
        regimes={str(r['regime']): r['depletion_rate'] <= .05 for r in summaries},
        caution='Empirical eligibility only; not a 5% safety certificate or feasibility proof.'))
    write_json(target / 'integrity.json', dict(passed=True, unique_jobs=270, missing_jobs=0,
        duplicate_jobs=0, plan_sha256=sha(output / 'plan.json')))
    print('Collected exactly 270 B5 validation runs. No next stage launched.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'check', 'collect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compatibility', type=Path)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'launch':
        if args.compatibility is None:
            parser.error('--compatibility is required for launch')
        launch(output, args.compatibility.resolve(), args.workers)
    elif args.action == 'check':
        check(output)
    else:
        collect(output)


if __name__ == '__main__':
    main()
