"""Fixed physical-only pilot. Completion may freeze scales, never launches a scheduler."""
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import numpy as np

from .config import CALIBRATION_SEED, MAP_SEED, ceil_grid
from .estimates import EstimateModel
from .navigation import FrozenNavigator
from .provenance import provenance, verify_provenance
from .storage import restore, snapshot, write_json
from .streams import legal_position

PILOT_JOBS = 1000


def freeze_results(manifest, rows):
    if len(rows) != PILOT_JOBS or {r['job_id'] for r in rows} != set(range(PILOT_JOBS)):
        raise ValueError('complete predeclared 1000-job pilot required before freezing')
    successful = [r for r in rows if r['success']]
    statistics = dict(total=len(rows), successful=len(successful),
                      navigation_failures=sum(r['timeout'] for r in rows),
                      success_fraction=len(successful) / len(rows),
                      quantile_population='successful jobs only; failures remain in raw data',
                      collision_count=sum(r['collision_count'] for r in rows))
    if not successful:
        return statistics, None
    for name, field in [('time', 'duration'), ('energy', 'energy_used')]:
        statistics[name] = dict(zip(('q25', 'median', 'q75', 'q90'),
                                    map(float, np.quantile([r[field] for r in successful], [.25, .5, .75, .9]))))
    mt, me = statistics['time']['median'], statistics['energy']['median']
    if mt <= 0 or me <= 0:
        raise ValueError('positive physical calibration medians required')
    regimes = []
    for battery in (2., 4., 6.):
        for charge in (.5, 1., 2.):
            for load in (.5, .9, 1.2):
                capacity = battery * me
                regimes.append(dict(id=len(regimes), battery_tilde=battery, charge_tilde=charge,
                                    eta=load, config=dict(capacity=capacity,
                                    recharge_rate=capacity / (charge * mt), arrival_rate=load / mt,
                                    cutoff=ceil_grid(100 * mt))))
    frozen = dict(version='PersistentUAVThroughput-v1.2', pilot_complete=True,
                  pilot_jobs=PILOT_JOBS, calibration_seed=CALIBRATION_SEED,
                  provenance=manifest['provenance'], layout=manifest['layout'],
                  statistics=statistics, model=EstimateModel.fit(rows).to_dict(), regimes=regimes,
                  limitation='Scales and linear fits condition on navigation success. No safety guarantee; do not hide failures.',
                  next_stage='NOT_STARTED; no automatic baseline launch')
    return statistics, frozen


def run(output, *, resume=False, checkpoint_policy_steps=100, stop_after_checkpoint=False):
    output = Path(output).resolve()
    if checkpoint_policy_steps < 1:
        raise ValueError('positive checkpoint interval required')
    stopping = [False]
    def handle_stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    if resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        verify_provenance(manifest['provenance'])
        state = restore(output)
        if state['kind'] != 'calibration' or state['manifest'] != manifest:
            raise RuntimeError('resume manifest mismatch')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('output is nonempty; use --resume for the same contract')
        output.mkdir(parents=True, exist_ok=True)
        seed_nav = FrozenNavigator(1e9)
        layout = seed_nav.layout
        seed_nav.close()
        rng = np.random.default_rng(CALIBRATION_SEED)
        pairs = [dict(job_id=i, start=legal_position(rng, layout), goal=legal_position(rng, layout))
                 for i in range(PILOT_JOBS)]
        # JSON-normalize tuples before equality checks after a restart.
        manifest = json.loads(json.dumps(dict(
            kind='physical_calibration', pilot_jobs=PILOT_JOBS, map_seed=MAP_SEED,
            calibration_seed=CALIBRATION_SEED, layout=layout, pairs=pairs,
            domain=[4000, 4000, 400], obstacles=24, option_step_limit=4000,
            phase='frozen navigation only; no scheduler', provenance=provenance(),
            conditioning='success-conditioned duration/energy medians; report every failed job',
            stopping_rule='1000 jobs or explicit pause; no adaptive sample/map selection')))
        write_json(output / 'manifest.json', manifest)
        state = dict(kind='calibration', manifest=manifest, rows=[], nav=None,
                     job_energy=0., total_policy_steps=0)
    policy_at_checkpoint = state['total_policy_steps']
    last_checkpoint = time.monotonic()

    def save(status):
        ref = snapshot(output, state)
        nav = state['nav']
        write_json(output / 'status.json', dict(
            status=status, completed_jobs=len(state['rows']), planned_jobs=PILOT_JOBS,
            total_policy_steps=state['total_policy_steps'],
            active_job=None if nav is None else len(state['rows']),
            active_job_simulation_time=None if nav is None else nav.time,
            active_job_energy=None if nav is None else state['job_energy'],
            snapshot=ref, checkpoint_written_unix=time.time(),
            training_updates=0, scheduler_runs=0,
            finite_state=nav is None or bool(np.isfinite(nav.position).all() and np.isfinite(nav.energy)),
            failure_reason_counts={'navigation_timeout': sum(r['timeout'] for r in state['rows'])},
            next_stage='NAVIGATION_QUALIFICATION_AND_USER_REVIEW_REQUIRED'))

    try:
        while len(state['rows']) < PILOT_JOBS:
            job = manifest['pairs'][len(state['rows'])]
            if state['nav'] is None:
                state['nav'] = FrozenNavigator(1e9, layout=manifest['layout'], start=job['start'])
                state['nav'].start_leg(job['goal'])
                state['job_energy'] = 0.
            nav = state['nav']
            reached = nav.reached()
            timeout = False
            if not reached:
                result = nav.advance_flight(.2)
                state['total_policy_steps'] += 1
                state['job_energy'] += result.energy_used
                reached, timeout = result.reached, result.timeout
                if result.depleted:
                    raise RuntimeError('virtual pilot battery unexpectedly exhausted')
            if reached or timeout:
                state['rows'].append(dict(**job, success=bool(reached), timeout=bool(timeout and not reached),
                                          duration=nav.time, energy_used=state['job_energy'],
                                          collision_count=nav.contacts, policy_steps=nav.policy_steps,
                                          hocbf_fallback_steps=nav.hocbf_fallback_steps))
                nav.close()
                state['nav'] = None
            due = (state['total_policy_steps'] - policy_at_checkpoint >= checkpoint_policy_steps
                   or time.monotonic() - last_checkpoint >= 60 or stopping[0])
            if due:
                save('paused' if stopping[0] or stop_after_checkpoint else 'running')
                print(json.dumps(dict(event='checkpoint', completed_jobs=len(state['rows']),
                                      policy_steps=state['total_policy_steps'])), flush=True)
                if stopping[0] or stop_after_checkpoint:
                    return
                policy_at_checkpoint = state['total_policy_steps']
                last_checkpoint = time.monotonic()
        write_json(output / 'jobs.json', state['rows'])
        statistics, frozen = freeze_results(manifest, state['rows'])
        write_json(output / 'statistics.json', statistics)
        if frozen is not None:
            write_json(output / 'frozen_regimes.json', frozen)
        save('complete' if frozen is not None else 'complete_no_successful_calibration')
    except Exception as error:
        # An interrupted policy step is NOT claimed resumable. Keep last known-good snapshot.
        write_json(output / 'error.json', dict(error=repr(error), time=time.time(),
                                             resume='Use last completed checkpoint; no automatic retry.'))
        status_path = output / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        status.update(status='error', error=repr(error), last_checkpoint_only=True)
        write_json(status_path, status)
        raise
