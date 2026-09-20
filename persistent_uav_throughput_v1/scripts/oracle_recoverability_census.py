"""Bounded, privileged state census; not a deployable policy or a viability proof."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import shlex
import signal
import subprocess
import sys
import tarfile
import time

PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
WORK = ROOT.parent
os.environ['PERSISTENT_UAV_LEGACY_ROOT'] = str(ROOT)
sys.path.insert(0, str(PROJECT))
import numpy as np
import torch
from persistent_uav.baselines import Action, Scheduler
from persistent_uav.config import Config
from persistent_uav.diagnostics import audited_step
from persistent_uav.environment import PersistentUAVThroughput
from persistent_uav.estimates import EstimateModel
from persistent_uav.navigation import FrozenNavigator
from persistent_uav.provenance import provenance, verify_provenance
from persistent_uav.storage import exclusive_run, restore, sha, snapshot, write_json
from persistent_uav.streams import stream_hash, workload

THREADS = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
THREADS.update(CUDA_VISIBLE_DEVICES='', ASCEND_RT_VISIBLE_DEVICES='')
CATEGORIES = {'fallback_predicted_infeasible_before_task': 'A',
              'post_task_waiting_margin_loss': 'B', 'waiting_depletion': 'B',
              'task_underestimation_sufficient_for_margin_flip': 'C',
              'task_prediction_false_safe': 'C'}


def normalized(value):
    return json.loads(json.dumps(value, allow_nan=False))


def assert_equal(actual, expected, context):
    if normalized(actual) != expected:
        raise AssertionError('Exact historical replay mismatch: ' + context)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state())


def set_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


def branch_specs(observation, group):
    specs = [] if group == 'B' else [dict(kind='task_return', task_id=t['id'])
                                     for t in sorted(observation['queue'], key=lambda t: t['id'])]
    if observation['can_recharge']:
        specs.append(dict(kind='direct_return', task_id=None))
    return specs


def select_targets(cases, runs):
    targets = []
    for case in cases:
        category = case['category']
        if category not in CATEGORIES:
            continue
        group = CATEGORIES[category]
        run = runs[case['source_member']]
        events = run['events']
        if category in ('fallback_predicted_infeasible_before_task', 'task_underestimation_sufficient_for_margin_flip'):
            index = case['previous_task_decision_event_index']
        elif category == 'post_task_waiting_margin_loss':
            index = next(i for i in range(case['previous_task_completion_event_index'] + 1, len(events))
                         if events[i]['event'] == 'decision')
        elif category == 'waiting_depletion':
            completed = max(i for i in range(case['decision_event_index']) if events[i]['event'] == 'task_completed')
            index = next(i for i in range(completed + 1, len(events)) if events[i]['event'] == 'decision')
            assert index == case['decision_event_index']
        else:
            index = case['decision_event_index']
        event = events[index]
        assert event['event'] == 'decision'
        obs = event['observation']
        if group == 'A':
            assert not obs['can_recharge'] and event['action']['reason'] == 'full_station_infeasible_estimate_fallback'
        if group == 'B':
            assert not obs['queue'] and event['action']['kind'] == 'idle' and obs['can_recharge']
        target_id = f"r{case['regime']:02d}_s{case['seed']}_e{index:05d}"
        targets.append(dict(id=target_id, group=group, category=category, regime=case['regime'],
                            seed=case['seed'], source_member=case['source_member'], decision_event_index=index,
                            decision_time=event['time'], original_action=event['action'],
                            original_failure_phase=case['failure_phase'], observation=obs,
                            branches=branch_specs(obs, group)))
    targets.sort(key=lambda t: (t['regime'], t['seed'], t['decision_event_index']))
    assert len(targets) == len({t['id'] for t in targets}) == 205
    assert Counter(t['group'] for t in targets) == dict(A=67, B=48, C=90)
    assert sum(len(t['branches']) for t in targets) == 561
    return targets


def prepare(output, workers=16):
    if not 1 <= workers <= 27:
        raise ValueError('invalid worker count')
    published = PROJECT / 'evidence/b5_validation_complete_20260920'
    decomposition = PROJECT / 'evidence/stranding_decomposition_20260920'
    for folder in (published, decomposition):
        for line in (folder / 'SHA256SUMS').read_text().splitlines():
            expected, name = line.split('  ', 1)
            assert sha(folder / name) == expected
    compatibility = json.loads((PROJECT / 'evidence/b5_validation_startup_20260920/calibration_compatibility_b5.json').read_text())
    verify_provenance(compatibility['diagnostic_provenance'])
    assert sha(ROOT / 'runtime_support/review_bundle/safety/collision/_qp_native.so') == compatibility['native']['arm64_binary_sha256']
    frozen_path = PROJECT / 'evidence/v1_2/calibration/frozen_regimes.json'
    assert sha(frozen_path) == compatibility['calibration_sha256']
    inventory = {r['path']: r for r in json.loads((published / 'archive_inventory.json').read_text())}
    runs, raw_runs = {}, {}
    with tarfile.open(published / 'validation_raw.tar.gz') as archive:
        assert set(archive.getnames()) == set(inventory)
        for member in archive:
            raw = archive.extractfile(member).read()
            assert len(raw) == inventory[member.name]['bytes']
            assert hashlib.sha256(raw).hexdigest() == inventory[member.name]['sha256']
            if Path(member.name).name.startswith('run_') and member.name.endswith('.json'):
                runs[member.name] = json.loads(raw)
                raw_runs[member.name] = raw
    targets = select_targets(json.loads((decomposition / 'failure_cases.json').read_text()), runs)
    output.mkdir(parents=True, exist_ok=False)
    for target in targets:
        destination = output / 'inputs' / target['source_member']
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw_runs[target['source_member']])
        target['source_sha256'] = sha(destination)
    plan = dict(kind='ORACLE_RECOVERABILITY_CENSUS_NOT_POLICY', workers=workers, targets=targets,
                groups=dict(A=67, B=48, C=90), total_states=205, total_branches=561,
                evidence_archive_sha256=sha(published / 'validation_raw.tar.gz'),
                decomposition_sha256=sha(decomposition / 'failure_cases.json'),
                frozen=str(frozen_path), frozen_sha256=sha(frozen_path),
                runtime_provenance=provenance(), native_sha256=compatibility['native']['arm64_binary_sha256'],
                implementation_sha256=sha(Path(__file__)), thread_environment=THREADS,
                source_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                protocol=dict(reconstruction='Replay saved B5 actions from reset; exact event/observation equality required before branching',
                    cloning='Full simulator pickle plus Python/NumPy/Torch RNG state, no observation-only reconstruction',
                    safe='Chosen task completed then charger reached, or direct charger reached, with no depletion/timeout',
                    energy_risk_and_navigation_failures_separate=True, horizon='Original absolute cutoff, no extension; censoring is unknown',
                    branch_policy_steps_max='2 * original option_step_limit for task+return; 1 * limit for direct return',
                    replay_policy_steps_max='ceil(original cutoff / physics_dt)',
                    privileged_diagnostic_only=True, no_future_task_policy=True, no_full_oracle_policy=True,
                    inference_limit='Observed bounded continuation under the frozen controller, not maximal viability or regime risk feasibility'),
                created_unix=time.time())
    write_json(output / 'plan.json', plan)
    print(json.dumps(dict(prepared=str(output), states=205, branches=561, groups=plan['groups'])), flush=True)


def load_source(output, target):
    path = output / 'inputs' / target['source_member']
    assert sha(path) == target['source_sha256']
    return json.loads(path.read_text())


def create_replay_env(target, frozen, source):
    regime = next(r for r in frozen['regimes'] if r['id'] == target['regime'])
    config = Config(**regime['config'])
    nav = FrozenNavigator(config.capacity, option_step_limit=config.option_step_limit, layout=frozen['layout'])
    tasks = workload(target['seed'], config, frozen['layout'])
    assert stream_hash(tasks) == source['summary']['stream_sha256']
    env = PersistentUAVThroughput(config, nav, tasks)
    assert_equal(env.events, source['events'][:len(env.events)], 'reset events')
    return env


def at_target(env, target, source):
    if len(env.events) != target['decision_event_index']:
        return False
    assert env.decision_required
    assert_equal(env.observe(), target['observation'], 'target observation')
    assert_equal(env.events, source['events'][:target['decision_event_index']], 'complete prefix at branch root')
    return True


def replay_step(env, source, scheduler):
    start = len(env.events)
    action = None
    if env.decision_required:
        event = source['events'][start]
        assert event['event'] == 'decision'
        assert_equal(env.observe(), event['observation'], 'decision observation')
        action = Action(**event['action'])
        assert_equal(asdict(scheduler.choose(env.observe())), event['action'], 'original B5 action')
    audited_step(env, action, scheduler.model)
    assert_equal(env.events[start:], source['events'][start:len(env.events)], 'new events after replay step')


def new_branch(root, spec):
    env = pickle.loads(pickle.dumps(root['env'], protocol=pickle.HIGHEST_PROTOCOL))
    set_rng(root['rng'])
    return dict(env=env, spec=spec, phase='task' if spec['kind'] == 'task_return' else 'return',
                started=False, initial_completed=env.completed, start_time=env.time,
                start_energy=env.nav.energy, start_policy_steps=env.nav.policy_steps,
                start_contacts=env.nav.contacts, root_event_count=len(env.events),
                task_completion=None, oracle_policy_steps=0)


def branch_result(branch, outcome, safe):
    env = branch['env']
    end_time = env.failure_time if env.failure_time is not None else env.time
    return dict(spec=branch['spec'], safe=safe, outcome=outcome, failure=env.failure,
                failure_phase=branch['phase'] if env.failure else None,
                task_completed=env.completed > branch['initial_completed'],
                task_completion=branch['task_completion'], reached_charger=outcome == 'success',
                start_time=branch['start_time'], physical_end_time=end_time,
                duration=end_time - branch['start_time'], remaining_battery=env.nav.energy,
                consumed_energy=branch['start_energy'] - env.nav.energy,
                full_mission_energy=(branch['start_energy'] - env.nav.energy) if safe else None,
                energy_censored=safe is not True,
                navigation_failure=env.failure == 'navigation_failure', depletion=env.failure == 'energy_depletion',
                horizon_censored=outcome == 'horizon_censored', policy_steps=branch['oracle_policy_steps'],
                collision_count=env.nav.contacts - branch['start_contacts'],
                physical_resets=env.nav.reset_count, oracle_calls=1,
                events=env.events[branch['root_event_count']:])


def advance_branch(branch):
    """At most one physical policy step; never charge or choose another task."""
    env = branch['env']
    phase = branch['phase']
    action = None
    if not branch['started']:
        action = Action('serve', branch['spec']['task_id'], 'privileged_census_candidate') if phase == 'task' else Action('recharge', reason='privileged_census_immediate_return')
        branch['started'] = True
    before, events_before = env.nav.policy_steps, len(env.events)
    env.step(action)
    branch['oracle_policy_steps'] += env.nav.policy_steps - before
    cap = env.config.option_step_limit * (2 if branch['spec']['kind'] == 'task_return' else 1)
    if branch['oracle_policy_steps'] > cap:
        raise RuntimeError('Branch exceeded its predeclared policy-step bound')
    if env.failure:
        return branch_result(branch, env.failure, False)
    new_events = env.events[events_before:]
    if phase == 'task':
        completion = next((e for e in new_events if e['event'] == 'task_completed'), None)
        if completion is not None:
            branch['task_completion'] = completion
            if np.linalg.norm(env.nav.position - env.nav.station) <= env.nav.goal_radius and env.nav.energy > 0:
                return branch_result(branch, 'success', True)
            branch['phase'], branch['started'] = 'return', False
    elif any(e['event'] == 'charger_arrival' for e in new_events):
        return branch_result(branch, 'success', True)
    if env.done:
        return branch_result(branch, 'horizon_censored', None)
    return None


def target_summary(target, results, root_reference):
    candidates = [r for r in results if r['spec']['kind'] == 'task_return']
    direct = next((r for r in results if r['spec']['kind'] == 'direct_return'), None)
    safe_ids = [r['spec']['task_id'] for r in candidates if r['safe'] is True]
    unknown = any(r['safe'] is None for r in candidates)
    selected_id = target['original_action']['task_id']
    alternative = [r for r in candidates if r['spec']['task_id'] != selected_id]
    def exists(rows):
        return True if any(r['safe'] is True for r in rows) else (None if any(r['safe'] is None for r in rows) else False)
    return dict(target_id=target['id'], group=target['group'], category=target['category'],
                regime=target['regime'], seed=target['seed'], root_snapshot=root_reference,
                branches=results, safe_task_ids=safe_ids,
                any_safe_task=(True if safe_ids else (None if unknown else False)) if candidates else None,
                alternative_safe_task=exists(alternative) if candidates else None,
                original_task_safe=next((r['safe'] for r in candidates if r['spec']['task_id'] == selected_id), None),
                direct_return_legal=target['observation']['can_recharge'],
                direct_return_safe=direct['safe'] if direct else None,
                any_tested_safe_continuation=exists(results),
                warning='No successful branch is not a proof of global infeasibility; navigation failure and cutoff remain distinct.')


def verify_plan(output):
    plan = json.loads((output / 'plan.json').read_text())
    assert sha(Path(__file__)) == plan['implementation_sha256']
    verify_provenance(plan['runtime_provenance'])
    assert sha(Path(plan['frozen'])) == plan['frozen_sha256']
    assert sha(ROOT / 'runtime_support/review_bundle/safety/collision/_qp_native.so') == plan['native_sha256']
    return plan


def worker(output, worker_id, resume=False):
    plan = verify_plan(output)
    assigned = plan['targets'][worker_id::plan['workers']]
    folder = output / f'worker_{worker_id:02d}'
    contract = dict(plan_sha256=sha(output / 'plan.json'), worker=worker_id, targets=[t['id'] for t in assigned])
    with exclusive_run(folder):
        if resume:
            assert json.loads((folder / 'manifest.json').read_text()) == contract
            state = restore(folder)
            assert state['contract'] == contract
            set_rng(state['rng'])
        else:
            folder.mkdir(exist_ok=False)
            write_json(folder / 'manifest.json', contract)
            state = dict(contract=contract, target_index=0, stage='replay', env=None, root=None,
                         branch=None, branch_index=0, branch_results=[], rows=[], total_policy_steps=0,
                         replay_policy_steps=0, branch_policy_steps=0, rng=rng_state())
        stopping = [False]
        def stop(signum, frame):
            stopping[0] = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        frozen = json.loads(Path(plan['frozen']).read_text())
        scheduler = Scheduler('reserve_sjf', EstimateModel(**frozen['model']), .25)
        previous_steps, previous_time = state['total_policy_steps'], time.monotonic()
        loaded_target_id, source = None, None
        def save(status):
            state['rng'] = rng_state()
            reference = snapshot(folder, state)
            write_json(folder / 'status.json', dict(status=status, completed_states=len(state['rows']),
                       planned_states=len(assigned), completed_branches=sum(len(r['branches']) for r in state['rows']) + len(state['branch_results']),
                       stage=state['stage'], current_target=assigned[state['target_index']]['id'] if state['target_index'] < len(assigned) else None,
                       total_policy_steps=state['total_policy_steps'], replay_policy_steps=state['replay_policy_steps'],
                       branch_policy_steps=state['branch_policy_steps'], snapshot=reference,
                       checkpoint_written_unix=time.time(), training_updates=0))
        try:
            while state['target_index'] < len(assigned):
                target = assigned[state['target_index']]
                if loaded_target_id != target['id']:
                    source = load_source(output, target)
                    loaded_target_id = target['id']
                target_folder = folder / target['id']
                if state['stage'] == 'replay':
                    if state['env'] is None:
                        state['env'] = create_replay_env(target, frozen, source)
                    env = state['env']
                    if at_target(env, target, source):
                        state['root'] = dict(env=env, rng=rng_state(), target_id=target['id'],
                                             exact_event_prefix_verified=True, exact_observation_verified=True)
                        reference = snapshot(target_folder / 'root', state['root'])
                        state['root_reference'] = reference
                        write_json(target_folder / 'root_verified.json', dict(target_id=target['id'],
                                   source_sha256=target['source_sha256'], event_prefix_length=len(env.events),
                                   observation=env.observe(), exact_match=True, snapshot=reference))
                        state['stage'], state['env'] = 'branch', None
                        state['branch_index'], state['branch_results'] = 0, []
                        save('running')
                    else:
                        if len(env.events) > target['decision_event_index'] or env.done:
                            raise RuntimeError('Replay passed target without exact match')
                        before = env.nav.policy_steps
                        replay_step(env, source, scheduler)
                        delta = env.nav.policy_steps - before
                        state['total_policy_steps'] += delta
                        state['replay_policy_steps'] += delta
                        if env.nav.policy_steps > math.ceil(env.config.cutoff / env.nav.dt):
                            raise RuntimeError('Replay exceeded original physical horizon bound')
                else:
                    if state['branch'] is None:
                        spec = target['branches'][state['branch_index']]
                        state['branch'] = new_branch(state['root'], spec)
                    branch = state['branch']
                    before = branch['env'].nav.policy_steps
                    result = advance_branch(branch)
                    delta = branch['env'].nav.policy_steps - before
                    state['total_policy_steps'] += delta
                    state['branch_policy_steps'] += delta
                    if result is not None:
                        index = state['branch_index']
                        # Save the full trace separately; checkpoint stores a compact result.
                        write_json(target_folder / f'branch_{index:02d}.json', result)
                        brief = {k: v for k, v in result.items() if k != 'events'}
                        brief.update(file=f'{target["id"]}/branch_{index:02d}.json',
                                     sha256=sha(target_folder / f'branch_{index:02d}.json'))
                        state['branch_results'].append(brief)
                        branch['env'].nav.close()
                        state['branch'] = None
                        state['branch_index'] += 1
                        if state['branch_index'] == len(target['branches']):
                            row = target_summary(target, state['branch_results'], state['root_reference'])
                            write_json(target_folder / 'result.json', row)
                            state['rows'].append(row)
                            state['root']['env'].nav.close()
                            state.update(target_index=state['target_index'] + 1, stage='replay', env=None, root=None,
                                         branch_index=0, branch_results=[])
                        save('running')
                if stopping[0] or state['total_policy_steps'] - previous_steps >= 100 or time.monotonic() - previous_time >= 60:
                    save('paused' if stopping[0] else 'running')
                    if stopping[0]:
                        return
                    previous_steps, previous_time = state['total_policy_steps'], time.monotonic()
            write_json(folder / 'results.json', state['rows'])
            state['stage'] = 'complete'
            save('complete')
        except Exception as error:
            # Preserve the last valid state; never checkpoint a mismatching replay as valid.
            write_json(folder / 'error.json', dict(error=repr(error), time=time.time(), target_id=target['id']))
            if (folder / 'status.json').exists():
                status = json.loads((folder / 'status.json').read_text())
                status.update(status='error', error=repr(error))
                write_json(folder / 'status.json', status)
            raise


def launch(output):
    plan = verify_plan(output)
    if (output / 'launch.json').exists():
        raise FileExistsError('Already launched; resume only verified stopped workers')
    records = []
    for i in range(plan['workers']):
        command = [str(WORK / '.venv/bin/python'), '-u', str(Path(__file__).resolve()),
                   'worker', '--output', str(output), '--worker', str(i)]
        logfile = output / f'worker_{i:02d}.log'
        with logfile.open('xb') as stream:
            process = subprocess.Popen(command, cwd=PROJECT, env=dict(os.environ, **THREADS),
                                       stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        records.append(dict(worker=i, pid=process.pid, command=command, log=str(logfile), started_unix=time.time()))
        write_json(output / 'launch.json', dict(workers=records))
    prefix = 'env ' + shlex.join([f'{k}={v}' for k, v in THREADS.items()])
    (output / 'resume_commands.txt').write_text('\n'.join(prefix + ' ' + shlex.join(r['command'] + ['--resume']) for r in records) + '\n')
    print(f'Launched {len(records)} workers: 205 fixed states / 561 bounded branches.', flush=True)


def check(output):
    plan = verify_plan(output)
    processes = json.loads((output / 'launch.json').read_text())['workers']
    assert len(processes) == plan['workers']
    checks = {}
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        for process in processes:
            i = process['worker']
            if i in checks:
                continue
            folder = output / f'worker_{i:02d}'
            if (folder / 'error.json').exists():
                raise RuntimeError((folder / 'error.json').read_text())
            proc = Path(f"/proc/{process['pid']}")
            assert [v.decode() for v in (proc / 'cmdline').read_bytes().split(b'\0') if v] == process['command']
            if not (folder / 'status.json').exists():
                continue
            status = json.loads((folder / 'status.json').read_text())
            if status['total_policy_steps'] < 100:
                continue
            assert status['status'] == 'running'
            try:
                state = restore(folder)
            except FileNotFoundError:
                continue
            manifest = json.loads((folder / 'manifest.json').read_text())
            assert state['contract'] == manifest and manifest['plan_sha256'] == sha(output / 'plan.json')
            assigned = plan['targets'][i::plan['workers']]
            target = assigned[state['target_index']]
            assert manifest['targets'] == [t['id'] for t in assigned]
            source = load_source(output, target)
            if state['stage'] == 'replay':
                env = state['env']
                assert_equal(env.events, source['events'][:len(env.events)], 'startup replay prefix')
                exact_root = False
            else:
                root = state['root']
                assert root['exact_event_prefix_verified'] and root['exact_observation_verified']
                assert at_target(root['env'], target, source)
                env = state['branch']['env'] if state['branch'] else root['env']
                exact_root = True
            assert np.isfinite(env.nav.observation()).all() and env.nav.reset_count == 1
            assert status['training_updates'] == 0
            threads = int(next(v.split(':')[1] for v in (proc / 'status').read_text().splitlines() if v.startswith('Threads:')))
            assert threads == 1
            environ = dict(v.decode().split('=', 1) for v in (proc / 'environ').read_bytes().split(b'\0') if b'=' in v)
            assert all(environ[k] == v for k, v in THREADS.items())
            checks[i] = dict(worker=i, pid=process['pid'], stage=state['stage'],
                             target_id=target['id'], total_policy_steps=state['total_policy_steps'],
                             threads=threads, snapshot_checksum_valid=True, historical_prefix_exact=True,
                             target_root_already_verified=exact_root, finite_observation=True,
                             manifest_sha256=sha(folder / 'manifest.json'), checked_unix=time.time())
        if len(checks) == plan['workers']:
            result = dict(passed=True, workers=[checks[i] for i in sorted(checks)], states=205, branches=561,
                          note='Bounded first-checkpoint health only. Replay may still precede branch-root verification.')
            write_json(output / 'startup_health.json', result)
            print(json.dumps(result), flush=True)
            return
        time.sleep(.5)
    raise RuntimeError('First-checkpoint deadline reached; inspect worker logs')


def collect(output):
    plan = verify_plan(output)
    rows = []
    for i in range(plan['workers']):
        folder = output / f'worker_{i:02d}'
        with exclusive_run(folder):
            assert json.loads((folder / 'status.json').read_text())['status'] == 'complete'
            state = restore(folder)
            result = json.loads((folder / 'results.json').read_text())
            assert state['rows'] == result
            assigned = plan['targets'][i::plan['workers']]
            assert [r['target_id'] for r in result] == [t['id'] for t in assigned]
            for row, target in zip(result, assigned):
                assert [r['spec'] for r in row['branches']] == target['branches']
                root_folder = folder / target['id'] / 'root'
                root = restore(root_folder)
                assert at_target(root['env'], target, load_source(output, target))
                for branch in row['branches']:
                    file = folder / branch['file']
                    assert sha(file) == branch['sha256']
                    raw = json.loads(file.read_text())
                    assert {k: v for k, v in branch.items() if k not in ('file', 'sha256')} == {k: v for k, v in raw.items() if k != 'events'}
            rows.extend(result)
    assert len(rows) == len({r['target_id'] for r in rows}) == 205
    assert sum(len(r['branches']) for r in rows) == 561
    report = dict(states=205, branches=561, groups={})
    for group in ('A', 'B', 'C'):
        subset = [r for r in rows if r['group'] == group]
        def count(field):
            return {str(value): sum(r[field] is value for r in subset) for value in (True, False, None)}
        report['groups'][group] = dict(states=len(subset), any_safe_task=count('any_safe_task'),
            alternative_safe_task=count('alternative_safe_task'), direct_return_safe=count('direct_return_safe'),
            any_tested_safe_continuation=count('any_tested_safe_continuation'),
            branch_outcomes=dict(Counter(b['outcome'] for r in subset for b in r['branches'])))
    report['limits'] = 'Failure-predecessor selection; deterministic frozen-controller continuation, not maximal viability or a population 5% risk estimate. No successful branch does not prove regime infeasibility.'
    target = output / 'aggregate'
    target.mkdir(exist_ok=False)
    write_json(target / 'results.json', sorted(rows, key=lambda r: r['target_id']))
    write_json(target / 'summary.json', report)
    write_json(target / 'integrity.json', dict(passed=True, unique_states=205, unique_branches=561,
        missing_states=0, duplicate_states=0, exact_root_reconstruction=True, plan_sha256=sha(output / 'plan.json')))
    print(json.dumps(report), flush=True)


def smoke(output):
    """One real nonzero-time historical root; bounded branch/recovery engineering check."""
    plan = verify_plan(output)
    target = min((t for t in plan['targets'] if t['decision_time'] > 0), key=lambda t: (t['decision_time'], t['id']))
    source = load_source(output, target)
    frozen = json.loads(Path(plan['frozen']).read_text())
    env = create_replay_env(target, frozen, source)
    scheduler = Scheduler('reserve_sjf', EstimateModel(**frozen['model']), .25)
    while not at_target(env, target, source):
        assert not env.done and len(env.events) <= target['decision_event_index']
        replay_step(env, source, scheduler)
    root = dict(env=env, rng=rng_state(), target_id=target['id'], exact_event_prefix_verified=True,
                exact_observation_verified=True)
    before = pickle.dumps(root, protocol=pickle.HIGHEST_PROTOCOL)
    branch = new_branch(root, target['branches'][0])
    result = advance_branch(branch)
    assert result is None, 'Tiny smoke expected an in-flight branch'
    saved_rng = rng_state()
    smoke_folder = output / 'engineering_smoke'
    reference = snapshot(smoke_folder / 'midflight', dict(branch=branch, rng=saved_rng))
    recovered = restore(smoke_folder / 'midflight')
    set_rng(saved_rng)
    left = advance_branch(branch)
    set_rng(recovered['rng'])
    right = advance_branch(recovered['branch'])
    assert normalized(left) == normalized(right)
    assert normalized(branch['env'].observe()) == normalized(recovered['branch']['env'].observe())
    assert normalized(branch['env'].events) == normalized(recovered['branch']['env'].events)
    assert pickle.dumps(root, protocol=pickle.HIGHEST_PROTOCOL) == before
    write_json(smoke_folder / 'smoke.json', dict(passed=True, kind='ENGINEERING_ONLY_NOT_CENSUS_RESULT',
        target_id=target['id'], original_decision_time=target['decision_time'], exact_prefix_events=len(env.events),
        replay_policy_steps=env.nav.policy_steps, exact_target_observation=True,
        full_state_clone_parent_unchanged=True, midflight_disk_resume_equal=True,
        bounded_branch_steps=branch['oracle_policy_steps'], snapshot=reference, training_updates=0))
    env.nav.close()
    branch['env'].nav.close()
    recovered['branch']['env'].nav.close()
    print('Real nonzero-time root replay, full-state cloning and midflight resume smoke passed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'smoke', 'launch', 'worker', 'check', 'collect'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--worker', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.action == 'prepare':
        prepare(output, args.workers)
    elif args.action == 'worker':
        if args.worker is None:
            parser.error('--worker required')
        worker(output, args.worker, args.resume)
    else:
        globals()[args.action](output)


if __name__ == '__main__':
    main()
