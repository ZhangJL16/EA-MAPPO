"""Explicitly launched, resumable B0–B5 evaluation; no pilot-to-sweep promotion."""
from dataclasses import asdict
import json
from pathlib import Path
import signal
import time

import numpy as np

from .baselines import Scheduler
from .config import Config, EVALUATION_SEEDS, METHODS, THRESHOLDS, VALIDATION_SEEDS
from .environment import PersistentUAVThroughput
from .estimates import EstimateModel
from .navigation import FrozenNavigator
from .provenance import provenance, verify_provenance
from .storage import restore, sha, snapshot, write_json
from .streams import workload


def wilson_interval(failures, n):
    if not n:
        return None
    z = 1.959963984540054
    p = failures / n
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0., float(center - radius)), min(1., float(center + radius))]


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r['regime'], r['method'], r['threshold'])
        groups.setdefault(key, []).append(r)
    result = []
    for (regime, method, threshold), cells in sorted(groups.items()):
        completed = np.array([r['completed'] for r in cells])
        failures = sum(r['depletion'] for r in cells)
        offered = sum(r['arrivals'] for r in cells)
        result.append(dict(regime=regime, method=method, threshold=threshold, n=len(cells),
                           mean_completed=float(completed.mean()),
                           standard_error=float(completed.std(ddof=1) / np.sqrt(len(cells))) if len(cells) > 1 else None,
                           depletion_rate=failures / len(cells), depletion_wilson95=wilson_interval(failures, len(cells)),
                           recharge_failure_rate=sum(r['recharge_failure'] for r in cells) / len(cells),
                           navigation_failure_rate=sum(r['navigation_failure'] for r in cells) / len(cells),
                           overflow_rate=sum(r['overflow'] for r in cells) / offered if offered else None,
                           mean_residual_battery=float(np.mean([r['terminal_residual_battery'] for r in cells]))))
    return result


def choose_thresholds(rows, regimes, delta=0.05):
    chosen = {}
    for regime in regimes:
        cells = [r for r in rows if r['regime'] == regime and r['method'] == 'threshold_sjf']
        for threshold in THRESHOLDS:
            actual_seeds = {r['seed'] for r in cells if r['threshold'] == threshold}
            if actual_seeds != set(VALIDATION_SEEDS):
                raise ValueError('threshold selection requires all predeclared validation seeds and candidates')
        summaries = aggregate(cells)
        eligible = [r for r in summaries if r['depletion_rate'] <= delta]
        chosen[str(regime)] = (min(eligible, key=lambda r: (-r['mean_completed'], r['depletion_rate'], r['threshold']))
                               if eligible else None)
    return chosen


def calibration_compatibility(frozen_path, frozen, compatibility_file):
    if compatibility_file is None:
        verify_provenance(frozen['provenance'])
        return None
    record = json.loads(Path(compatibility_file).read_text())
    if (record['calibration_sha256'] != sha(frozen_path)
            or record['calibration_provenance'] != frozen['provenance']):
        raise ValueError('compatibility record refers to a different calibration')
    verify_provenance(record['diagnostic_provenance'])
    return record


def record_failure(output, error):
    """Mark failure without replacing the last valid checkpoint with partial state."""
    output = Path(output)
    details = dict(error=repr(error), time=time.time(),
                   resume='Use last completed checkpoint; no automatic retry.')
    write_json(output / 'error.json', details)
    status_path = output / 'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else dict(
        completed_runs=0, snapshot=None, training_updates=0)
    status.update(status='error', error=details['error'], error_written_unix=details['time'])
    write_json(status_path, status)


def run(frozen_path, output, *, split, regime_ids, methods=None, threshold_file=None,
        compatibility_file=None,
        resume=False, checkpoint_policy_steps=100, stop_after_checkpoint=False):
    frozen_path, output = Path(frozen_path).resolve(), Path(output).resolve()
    frozen = json.loads(frozen_path.read_text())
    if not frozen.get('pilot_complete') or frozen.get('pilot_jobs') != 1000:
        raise ValueError('real completed 1000-job physical calibration required')
    compatibility = calibration_compatibility(frozen_path, frozen, compatibility_file)
    if split not in ('validation', 'evaluation'):
        raise ValueError('unknown data split')
    if checkpoint_policy_steps < 1:
        raise ValueError('positive checkpoint interval required')
    ids = sorted(set(regime_ids))
    all_regimes = {r['id']: r for r in frozen['regimes']}
    if not ids or not set(ids) <= set(all_regimes):
        raise ValueError('invalid or empty regime subset')
    selected_methods = tuple(methods or (('threshold_sjf',) if split == 'validation' else METHODS))
    if not set(selected_methods) <= set(METHODS):
        raise ValueError('unsupported baseline')
    threshold_data = None
    if split == 'evaluation' and 'threshold_sjf' in selected_methods:
        if threshold_file is None:
            raise ValueError('B4 evaluation needs independent validation threshold artifact')
        threshold_data = json.loads(Path(threshold_file).read_text())
        if threshold_data['calibration_sha256'] != sha(frozen_path):
            raise ValueError('thresholds from another calibration')
        if threshold_data['validation_seeds'] != list(VALIDATION_SEEDS):
            raise ValueError('threshold split mismatch')
    seeds = VALIDATION_SEEDS if split == 'validation' else EVALUATION_SEEDS
    jobs, unavailable = [], []
    for regime in ids:
        for method in selected_methods:
            thresholds = (0.25,)
            if method == 'threshold_sjf':
                if split == 'validation':
                    thresholds = THRESHOLDS
                else:
                    if str(regime) not in threshold_data['selected']:
                        raise ValueError('requested regime has not completed threshold validation')
                    cell = threshold_data['selected'].get(str(regime))
                    if cell is None:
                        unavailable.append(dict(regime=regime, method=method, reason='no empirically admissible validation threshold'))
                        continue
                    thresholds = (cell['threshold'],)
            for threshold in thresholds:
                jobs.extend(dict(regime=regime, method=method, threshold=threshold, seed=seed) for seed in seeds)
    contract = dict(kind='baseline_evaluation', split=split, calibration_sha256=sha(frozen_path),
                    interpretation='B0–B5 diagnostic only; not a safety certificate or 98% kill test',
                    calibration_compatibility=compatibility,
                    provenance=provenance(), regimes=ids, jobs=jobs,
                    unavailable=unavailable, threshold_data=threshold_data,
                    dataset_seeds=list(seeds), future_stream_visible=False, oracle_calls=0)
    if resume:
        manifest = json.loads((output / 'manifest.json').read_text())
        verify_provenance(manifest['provenance'])
        if manifest != contract:
            raise ValueError('evaluation resume contract mismatch')
        state = restore(output)
        if state['contract'] != contract:
            raise ValueError('snapshot contract mismatch')
    else:
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('nonempty output requires explicit --resume')
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / 'manifest.json', contract)
        state = dict(contract=contract, rows=[], env=None, total_policy_steps=0)
    stopping = [False]
    def handle_stop(signum, frame):
        stopping[0] = True
    signal.signal(signal.SIGTERM, handle_stop)
    signal.signal(signal.SIGINT, handle_stop)
    model = EstimateModel(**frozen['model'])
    previous_steps, last_checkpoint = state['total_policy_steps'], time.monotonic()

    def save(status):
        ref = snapshot(output, state)
        write_json(output / 'status.json', dict(status=status, completed_runs=len(state['rows']),
                   planned_runs=len(jobs), total_policy_steps=state['total_policy_steps'],
                   snapshot=ref, checkpoint_written_unix=time.time(), training_updates=0))

    try:
        while len(state['rows']) < len(jobs):
            job = jobs[len(state['rows'])]
            config = Config(**all_regimes[job['regime']]['config'])
            if state['env'] is None:
                nav = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit, layout=frozen['layout'])
                tasks = workload(job['seed'], config, frozen['layout'])
                state['env'] = PersistentUAVThroughput(config, nav, tasks)
                # Private evaluator artifact, never supplied to Scheduler.choose.
                write_json(output / f"stream_r{job['regime']:02d}_s{job['seed']}.json", [asdict(t) for t in tasks])
            env = state['env']
            scheduler = Scheduler(job['method'], model, job['threshold'])
            action = scheduler.choose(env.observe()) if env.decision_required else None
            before = env.nav.policy_steps
            env.step(action)
            state['total_policy_steps'] += env.nav.policy_steps - before
            if env.done:
                row = dict(**job, **env.summary())
                state['rows'].append(row)
                write_json(output / f"run_{len(state['rows'])-1:05d}.json", dict(summary=row, events=env.events))
                env.nav.close()
                state['env'] = None
            if (state['total_policy_steps'] - previous_steps >= checkpoint_policy_steps
                    or time.monotonic() - last_checkpoint >= 60 or stopping[0]):
                save('paused' if stopping[0] or stop_after_checkpoint else 'running')
                print(json.dumps(dict(event='checkpoint', completed_runs=len(state['rows']),
                                      policy_steps=state['total_policy_steps'])), flush=True)
                if stopping[0] or stop_after_checkpoint:
                    return
                previous_steps, last_checkpoint = state['total_policy_steps'], time.monotonic()
        write_json(output / 'results.json', state['rows'])
        write_json(output / 'summary.json', aggregate(state['rows']))
        if split == 'validation' and 'threshold_sjf' in selected_methods:
            write_json(output / 'thresholds.json', dict(
                calibration_sha256=sha(frozen_path), validation_seeds=list(VALIDATION_SEEDS),
                selected=choose_thresholds(state['rows'], ids),
                caution='Empirical validation budget only, not a safety guarantee. Null means no admissible threshold.'))
        save('complete')
    except Exception as error:
        record_failure(output, error)
        raise
