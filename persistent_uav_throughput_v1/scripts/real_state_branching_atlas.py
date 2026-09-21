"""Frozen real-state paired branching atlas; privileged diagnostic, never training."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import shlex
import signal
import subprocess
import sys
import tarfile
import time

# Set thread controls before numerical-library imports, including direct CLI use.
THREADS = {k: '1' for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS')}
THREADS.update(CUDA_VISIBLE_DEVICES='', ASCEND_RT_VISIBLE_DEVICES='')
os.environ.update(THREADS)
import oracle_recoverability_census as census
from persistent_uav.baselines import Action, Scheduler
from persistent_uav.diagnostics import audited_step, reserve_prediction
from persistent_uav.estimates import EstimateModel
from persistent_uav.storage import exclusive_run, restore, sha, snapshot, write_json
from persistent_uav.provenance import provenance, verify_provenance
from persistent_uav.streams import stream_hash
import numpy as np

PROJECT, ROOT, WORK = census.PROJECT, census.ROOT, census.WORK
SCRIPT = Path(__file__).resolve()
WINDOW = 1308.6
WORKERS = 16
ANCHOR = 'root'  # Explicit user confirmation: common absolute window including first task.
SOURCES = {'B4_0.75': 'arm_validation_20260920', 'B5': 'b5_validation_complete_20260920'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def station(env):
    return bool(np.linalg.norm(env.nav.position - env.nav.station) <= env.nav.goal_radius)


def read_archives():
    runs, raw, archives = {}, {}, {}
    for policy, name in SOURCES.items():
        folder = PROJECT / 'evidence' / name
        for line in (folder / 'SHA256SUMS').read_text().splitlines():
            expected, relative = line.split('  ', 1)
            assert sha(folder / relative) == expected
        inventory = {r['path']: r for r in json.loads((folder / 'archive_inventory.json').read_text())
                     if r.get('archive', 'validation_raw.tar.gz') == 'validation_raw.tar.gz'}
        archive_path = folder / 'validation_raw.tar.gz'
        seen = set()
        with tarfile.open(archive_path) as archive:
            for member in archive:
                content = archive.extractfile(member).read()
                entry = inventory[member.name]
                assert len(content) == entry['bytes'] and hashlib.sha256(content).hexdigest() == entry['sha256']
                seen.add(member.name)
                if Path(member.name).name.startswith('run_') and member.name.endswith('.json'):
                    run = json.loads(content)
                    s = run['summary']
                    if policy == 'B4_0.75' and (s['method'] != 'threshold_sjf' or s['threshold'] != .75):
                        continue
                    assert s['method'] == ('threshold_sjf' if policy == 'B4_0.75' else 'reserve_sjf')
                    key = policy + '/' + member.name
                    runs[key], raw[key] = run, content
        assert seen == set(inventory)
        archives[policy] = dict(file=str(archive_path), sha256=sha(archive_path), verified_members=len(seen))
        subset = [r['summary'] for k, r in runs.items() if k.startswith(policy + '/')]
        assert len(subset) == len({(r['regime'], r['seed']) for r in subset}) == 270
    return runs, raw, archives


def select_roots(runs):
    cells = defaultdict(list)
    for member, run in sorted(runs.items()):
        policy = member.split('/')[0]
        regime, seed = run['summary']['regime'], run['summary']['seed']
        number = -1
        for event_index, event in enumerate(run['events']):
            if event['event'] != 'decision':
                continue
            number += 1
            o = event['observation']
            if len(o['queue']) < 2 or o['remaining_time'] + 1e-7 < WINDOW:
                continue
            assert o['decision_required'] and o['mode'] == 'IDLE'
            key = [policy, regime, seed, number]
            row = dict(id=f'{policy}_r{regime:02d}_s{seed}_d{number:05d}', policy=policy,
                       regime=regime, seed=seed, decision_index=number,
                       decision_event_index=event_index, decision_time=event['time'],
                       selection_sha256=digest(key), selection_key=key, source_member=member,
                       observation=o, original_action=event['action'],
                       stream_sha256=run['summary']['stream_sha256'])
            cells[(policy, regime)].append(row)
    roots, counts = [], []
    for policy in SOURCES:
        for regime in range(27):
            available = sorted(cells[(policy, regime)], key=lambda r: (r['selection_sha256'], r['id']))
            chosen = available[:4]
            roots.extend(chosen)
            counts.append(dict(policy=policy, regime=regime, eligible=len(available), selected=len(chosen)))
    roots.sort(key=lambda r: (r['policy'], r['regime'], r['selection_sha256']))
    assert len({r['id'] for r in roots}) == len(roots) <= 216
    return roots, counts


def prepare(output):
    runs, raw, archives = read_archives()
    roots, cells = select_roots(runs)
    compatibility = json.loads((PROJECT / 'evidence/b5_validation_startup_20260920/calibration_compatibility_b5.json').read_text())
    verify_provenance(compatibility['diagnostic_provenance'])
    native = ROOT / 'runtime_support/review_bundle/safety/collision/_qp_native.so'
    assert sha(native) == compatibility['native']['arm64_binary_sha256']
    frozen = PROJECT / 'evidence/v1_2/calibration/frozen_regimes.json'
    assert sha(frozen) == compatibility['calibration_sha256']
    output.mkdir(parents=True, exist_ok=False)
    for root in roots:
        path = output / 'inputs' / root['source_member']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[root['source_member']])
        root['source_sha256'] = sha(path)
    write_json(output / 'roots.json', roots)
    manifest = dict(kind='REAL_STATE_COUNTERFACTUAL_BRANCHING_ATLAS_V1', created_unix=time.time(),
        source_revision=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        implementation_sha256=sha(SCRIPT), census_helper_sha256=sha(Path(census.__file__)),
        runtime_provenance=provenance(), native_sha256=sha(native), frozen=str(frozen), frozen_sha256=sha(frozen),
        archives=archives, roots_sha256=sha(output / 'roots.json'), roots=len(roots), cells=cells, workers=WORKERS,
        window_seconds=WINDOW, window_anchor=ANCHOR, thread_environment=THREADS,
        selection='SHA256 of compact JSON [policy, integer regime, seed, zero-based decision ordinal]; lowest four per cell',
        one_step_horizon='original T; horizon censoring is unknown, not unsafe',
        continuation_oracle='all queued task+return branches; shortest frozen-estimated task time among true-safe tasks; task ID tie',
        continuation_window='observation-only cutoff; original T and stream retained for oracle calls',
        no_safe_continuation='stop diagnostic and mark N(W)/F(W) unknown; no invented wait, reset or catastrophe',
        at_station_return='already arrived, zero time/energy; never execute illegal full-station recharge',
        local_heuristics='actual task time; actual task energy; actual post-task battery minus actual return energy; task ID ties',
        no_training=True, no_environment_changes=True, outcome_based_root_selection=False)
    write_json(output / 'manifest.json', manifest)
    print(json.dumps(dict(roots=len(roots), cells=cells, output=str(output))), flush=True)


def verify(output):
    m = json.loads((output / 'manifest.json').read_text())
    assert m['implementation_sha256'] == sha(SCRIPT)
    assert m['census_helper_sha256'] == sha(Path(census.__file__))
    assert m['roots_sha256'] == sha(output / 'roots.json')
    assert m['frozen_sha256'] == sha(m['frozen'])
    assert m['native_sha256'] == sha(ROOT / 'runtime_support/review_bundle/safety/collision/_qp_native.so')
    verify_provenance(m['runtime_provenance'])
    return m, json.loads((output / 'roots.json').read_text())


def replay_step(env, root, source, model):
    start = len(env.events)
    action = None
    if env.decision_required:
        event = source['events'][start]
        assert event['event'] == 'decision'
        census.assert_equal(env.observe(), event['observation'], 'atlas replay observation')
        action = Action(**event['action'])
        scheduler = Scheduler('reserve_sjf' if root['policy'] == 'B5' else 'threshold_sjf', model,
                              .25 if root['policy'] == 'B5' else .75)
        census.assert_equal(asdict(scheduler.choose(env.observe())), event['action'], 'atlas archived action')
    if root['policy'] == 'B5':
        audited_step(env, action, model)
    else:
        env.step(action)
    census.assert_equal(env.events[start:], source['events'][start:len(env.events)], 'atlas replay prefix')


def specs(env):
    return [dict(kind='task_return', task_id=t.id) for t in sorted(env.queue, key=lambda t: t.id)] + [dict(kind='direct_return', task_id=None)]


def advance_one(branch):
    env = branch['env']
    if branch['phase'] == 'return' and not branch['started'] and station(env) and env.nav.energy > 0:
        return census.branch_result(branch, 'success', True)
    return census.advance_branch(branch)


def enriched(result, observation, model):
    completion = result['task_completion']
    candidates = reserve_prediction(observation, Action('idle'), model)['candidates']
    prediction = next((r for r in candidates if r['task_id'] == result['spec']['task_id']), None)
    result['prediction'] = prediction
    result['task_time'] = completion['time'] - result['start_time'] if completion else (result['duration'] if result['failure_phase']=='task' else None)
    result['task_energy'] = observation['battery'] - completion['battery'] if completion else (result['consumed_energy'] if result['failure_phase']=='task' else None)
    result['task_cost_complete'] = completion is not None
    result['post_task_battery'] = completion['battery'] if completion else None
    result['post_task_position'] = completion['position'] if completion else None
    result['return_time'] = (result['physical_end_time'] - completion['time']) if completion else (result['duration'] if result['spec']['kind'] == 'direct_return' else None)
    result['return_energy'] = (completion['battery'] - result['remaining_battery']) if completion else (result['consumed_energy'] if result['spec']['kind'] == 'direct_return' else None)
    result['actual_reserve'] = result['remaining_battery'] if completion and result['safe'] is True else None
    return result


def clone_root(env, rng):
    return dict(env=env, rng=rng)


def window_end(root_time, task_time=None):
    origin = root_time if ANCHOR == 'root' else task_time
    return None if origin is None else round((origin + WINDOW) / .05) * .05


def window_step(env, action, deadline):
    """Apply only an observation cutoff; the physical model is never edited.

    A temporary config copy bounds stationary leaps and last flight substeps.
    Oracle queries always receive the restored original config/horizon.
    """
    original = env.config
    if deadline is not None:
        env.config = replace(original, cutoff=min(original.cutoff, deadline))
    try:
        return env.step(action)
    finally:
        env.config = original


def new_continuation(root, row):
    env = pickle.loads(pickle.dumps(root['env'], protocol=pickle.HIGHEST_PROTOCOL))
    return dict(env=env, rng=root['rng'], first=row, stage='initial', initial_started=False,
                start_time=env.time, initial_completed=env.completed, event_start=len(env.events),
                deadline=window_end(env.time), initial_completion=None, oracle=None,
                oracle_results=[], oracle_index=0, oracle_decisions=[], oracle_physics=0,
                forced_returns=0, forced_return_reasons=Counter(), stopped_reason=None,
                start_modes=dict(env.time_by_mode), start_overflow=env.overflow)


def continuation_result(c, reason):
    env = c['env']
    deadline = c['deadline']
    failure_time = env.failure_time
    failed = bool(env.failure and (deadline is None or failure_time <= deadline + 1e-7))
    resolved = failed or (deadline is not None and env.time >= deadline - 1e-7 and reason == 'window_end')
    end = failure_time if failed else env.time
    origin = c['start_time'] if ANCHOR == 'root' else (c['initial_completion']['time'] if c['initial_completion'] else None)
    initial_in_window = 0 if ANCHOR == 'root' else int(c['initial_completion'] is not None)
    observed = env.completed - c['initial_completed'] - initial_in_window
    return dict(task_id=c['first']['spec']['task_id'], start_time=c['start_time'], window_origin=origin,
                window_end=deadline, window_seconds=WINDOW, N_W=observed if resolved else None,
                N_observed=observed, F_W=(not failed) if resolved else None,
                outcome=reason, failure=env.failure, failure_time=failure_time,
                time_alive=min(WINDOW, max(0., end-origin)) if origin is not None else None,
                failure_elapsed=(failure_time-origin) if failed and origin is not None else None,
                depletion=failed and env.failure == 'energy_depletion',
                navigation_failure=failed and env.failure == 'navigation_failure',
                no_safe_continuation=reason == 'no_safe_continuation', horizon_censored=not resolved,
                forced_return_count=c['forced_returns'], forced_return_reasons=dict(c['forced_return_reasons']),
                first_task_completion=c['initial_completion'], physical_end_time=end,
                post_battery=env.nav.energy, post_position=env.nav.position.tolist(),
                overflow=env.overflow-c['start_overflow'],
                time_by_mode={k: env.time_by_mode[k]-v for k,v in c['start_modes'].items()},
                oracle_decisions=c['oracle_decisions'], oracle_policy_steps=c['oracle_physics'],
                stream_sha256=env.workload_hash, physical_resets=env.nav.reset_count,
                events=env.events[c['event_start']:])


def advance_continuation(c, model):
    """One real or counterfactual step. Every nested branch is checkpointable."""
    env = c['env']
    if env.done:
        reason = env.failure or ('window_end' if c['deadline'] is not None and env.time >= c['deadline']-1e-7 else 'original_horizon_censored')
        return continuation_result(c, reason)
    if c['deadline'] is not None and env.time >= c['deadline']-1e-7:
        return continuation_result(c, 'window_end')
    if c['stage'] == 'initial':
        action = None if c['initial_started'] else Action('serve', c['first']['spec']['task_id'], 'atlas_first_safe_task')
        c['initial_started'] = True
        before = len(env.events)
        census.set_rng(c['rng'])
        window_step(env, action, c['deadline'])
        c['rng'] = census.rng_state()
        completion = next((e for e in env.events[before:] if e['event'] == 'task_completed'), None)
        if completion is not None:
            if c['deadline'] is None or completion['time'] < c['deadline']-1e-7:
                census.assert_equal(completion, c['first']['task_completion'], 'paired first-task trajectory')
            # At the reporting endpoint the unchanged navigator may use fewer
            # substeps; full-option endpoint equality is not asserted there.
            c['initial_completion'] = completion
            c['deadline'] = window_end(c['start_time'], completion['time'])
            c['stage'] = 'physical'
        return None
    if c['stage'] == 'oracle':
        if c['oracle'] is None:
            c['oracle'] = census.new_branch(clone_root(env, c['rng']), c['oracle_specs'][c['oracle_index']])
        branch = c['oracle']
        before = branch['env'].nav.policy_steps
        result = advance_one(branch)
        c['oracle_physics'] += branch['env'].nav.policy_steps - before
        if result is None:
            return None
        result = enriched(result, env.observe(), model)
        # Oracle queries are evaluated with original T, not the reporting window.
        c['oracle_results'].append(result)
        branch['env'].nav.close()
        c['oracle'], c['oracle_index'] = None, c['oracle_index']+1
        if c['oracle_index'] < len(c['oracle_specs']):
            return None
        safe = [r for r in c['oracle_results'] if r['spec']['kind']=='task_return' and r['safe'] is True]
        direct = next(r for r in c['oracle_results'] if r['spec']['kind']=='direct_return')
        if safe:
            chosen = min(safe, key=lambda r:(r['prediction']['predicted_task_time'],r['spec']['task_id']))
            action = Action('serve', chosen['spec']['task_id'], 'atlas_oracle_safe_frozen_sjf')
        elif env.can_recharge and direct['safe'] is True:
            why = 'empty_queue_return' if not env.queue else 'no_safe_task_return'
            action = Action('recharge', reason=why)
            c['forced_returns'] += 1
            c['forced_return_reasons'][why] += 1
        else:
            action = None
        c['oracle_decisions'].append(dict(time=env.time, observation=env.observe(),
            action=asdict(action) if action else None, branches=c['oracle_results']))
        c['oracle_results'] = []
        if action is None:
            return continuation_result(c, 'no_safe_continuation')
        c['pending_action'], c['stage'] = action, 'physical'
        return None
    if env.decision_required and 'pending_action' not in c:
        if not env.queue and station(env):
            c['pending_action'] = Action('idle', reason='atlas_empty_queue_at_station_wait')
        else:
            c['oracle_specs'] = specs(env) if env.queue else [dict(kind='direct_return',task_id=None)]
            c['oracle_index'], c['oracle_results'], c['stage'] = 0, [], 'oracle'
            return None
    action = c.pop('pending_action', None)
    census.set_rng(c['rng'])
    window_step(env, action, c['deadline'])
    c['rng'] = census.rng_state()
    return None


def compact(row):
    return {k:v for k,v in row.items() if k not in ('events','oracle_decisions')}


def physical_steps(state):
    total = state['replay_steps'] + state['a_steps'] + state['b_finished_steps']
    c = state.get('continuation')
    if c:
        total += c['env'].nav.policy_steps - state['root']['env'].nav.policy_steps + c['oracle_physics']
    return total


def worker(output, worker_id, resume=False):
    manifest, roots = verify(output)
    assigned = roots[worker_id::manifest['workers']]
    folder = output / f'worker_{worker_id:02d}'
    contract = dict(manifest_sha256=sha(output/'manifest.json'), worker=worker_id, roots=[r['id'] for r in assigned])
    with exclusive_run(folder):
        if resume:
            assert json.loads((folder/'manifest.json').read_text()) == contract
            state = restore(folder)
            assert state['contract'] == contract
            census.set_rng(state['rng'])
        else:
            folder.mkdir(exist_ok=False)
            write_json(folder/'manifest.json', contract)
            state = dict(contract=contract, index=0, stage='replay', env=None, root=None, branch=None,
                         a_index=0, a_rows=[], b_index=0, b_rows=[], continuation=None, rows=[],
                         replay_steps=0,a_steps=0,b_finished_steps=0,rng=census.rng_state())
        stopping = [False]
        def stop(*unused): stopping[0] = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        frozen = json.loads(Path(manifest['frozen']).read_text())
        model = EstimateModel(**frozen['model'])
        previous_steps, previous_time = physical_steps(state), time.monotonic()
        loaded, source = None, None
        def save(status):
            state['rng'] = census.rng_state()
            ref = snapshot(folder,state)
            write_json(folder/'status.json',dict(status=status,stage=state['stage'], completed_roots=len(state['rows']),
                planned_roots=len(assigned),current_root=assigned[state['index']]['id'] if state['index']<len(assigned) else None,
                total_policy_steps=physical_steps(state),one_step_complete=len(state['a_rows']),
                continuation_complete=len(state['b_rows']),snapshot=ref,checkpoint_unix=time.time(),training_updates=0))
        try:
            while state['index'] < len(assigned):
                target = assigned[state['index']]
                path = folder/target['id']
                if loaded != target['id']:
                    source = census.load_source(output,target)
                    loaded = target['id']
                if state['stage'] == 'replay':
                    if state['env'] is None:
                        state['env'] = census.create_replay_env(target,frozen,source)
                    env = state['env']
                    if census.at_target(env,target,source):
                        assert stream_hash(env._tasks) == target['stream_sha256']
                        state['root'] = clone_root(env,census.rng_state())
                        ref = snapshot(path/'root',state['root'])
                        write_json(path/'root_verified.json',dict(root_id=target['id'],exact_prefix=True,exact_observation=True,
                                   root_snapshot=ref,stream_sha256=env.workload_hash,observation=env.observe()))
                        state.update(stage='A',env=None,a_index=0,a_rows=[],b_index=0,b_rows=[])
                        state['a_specs'] = specs(env)
                        save('running')
                    else:
                        assert not env.done and len(env.events)<target['decision_event_index']
                        before = env.nav.policy_steps
                        replay_step(env,target,source,model)
                        state['replay_steps'] += env.nav.policy_steps-before
                elif state['stage'] == 'A':
                    if state['branch'] is None:
                        state['branch'] = census.new_branch(state['root'],state['a_specs'][state['a_index']])
                    branch = state['branch']
                    before = branch['env'].nav.policy_steps
                    result = advance_one(branch)
                    state['a_steps'] += branch['env'].nav.policy_steps-before
                    if result is not None:
                        result = enriched(result,target['observation'],model)
                        result.update(root_id=target['id'],policy=target['policy'],regime=target['regime'])
                        file = path/f'A_{state["a_index"]:02d}.json'
                        write_json(file,result)
                        row = compact(result); row.update(file=str(file.relative_to(folder)),sha256=sha(file))
                        state['a_rows'].append(row)
                        branch['env'].nav.close()
                        state['branch'], state['a_index'] = None,state['a_index']+1
                        if state['a_index']==len(state['a_specs']):
                            state['safe_rows'] = [r for r in state['a_rows'] if r['spec']['kind']=='task_return' and r['safe'] is True]
                            state['stage'] = 'B' if len(state['safe_rows'])>=2 else 'finish'
                        save('running')
                elif state['stage'] == 'B':
                    if state['continuation'] is None:
                        state['continuation'] = new_continuation(state['root'],state['safe_rows'][state['b_index']])
                    c = state['continuation']
                    result = advance_continuation(c,model)
                    if result is not None:
                        assert stream_hash(c['env']._tasks) == target['stream_sha256']
                        result.update(root_id=target['id'],policy=target['policy'],regime=target['regime'])
                        file = path/f'B_{state["b_index"]:02d}.json'
                        write_json(file,result)
                        row = compact(result); row.update(file=str(file.relative_to(folder)),sha256=sha(file))
                        state['b_rows'].append(row)
                        state['b_finished_steps'] += c['env'].nav.policy_steps-state['root']['env'].nav.policy_steps+c['oracle_physics']
                        c['env'].nav.close()
                        state['continuation'], state['b_index'] = None,state['b_index']+1
                        if state['b_index']==len(state['safe_rows']):
                            state['stage']='finish'
                        save('running')
                else:
                    assert state['stage']=='finish'
                    row=dict(root_id=target['id'],one_step=state['a_rows'],continuations=state['b_rows'])
                    write_json(path/'result.json',row)
                    state['rows'].append(row)
                    state['root']['env'].nav.close()
                    state.update(index=state['index']+1,stage='replay',env=None,root=None,branch=None,
                                 continuation=None,a_rows=[],b_rows=[],a_index=0,b_index=0)
                    save('running')
                if stopping[0] or physical_steps(state)-previous_steps>=100 or time.monotonic()-previous_time>=60:
                    save('paused' if stopping[0] else 'running')
                    if stopping[0]: return
                    previous_steps,previous_time=physical_steps(state),time.monotonic()
            state['stage']='complete'
            write_json(folder/'results.json',state['rows'])
            save('complete')
        except Exception as error:
            write_json(folder/'error.json',dict(error=repr(error),root=loaded,time=time.time()))
            raise


def launch(output):
    manifest,_=verify(output)
    if (output/'launch.json').exists(): raise FileExistsError('already launched; use recorded resume commands')
    records=[]
    for i in range(manifest['workers']):
        command=[str(WORK/'.venv/bin/python'),'-u',str(SCRIPT),'worker','--output',str(output),'--worker',str(i)]
        with (output/f'worker_{i:02d}.log').open('xb') as log:
            p=subprocess.Popen(command,cwd=PROJECT,env=dict(os.environ,**THREADS),stdin=subprocess.DEVNULL,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        records.append(dict(worker=i,pid=p.pid,command=command))
        write_json(output/'launch.json',dict(workers=records))
    (output/'resume_commands.txt').write_text('\n'.join(shlex.join(r['command']+['--resume']) for r in records)+'\n')
    print(json.dumps(dict(launched=len(records),output=str(output))),flush=True)


def check(output):
    manifest,roots=verify(output)
    checks=[]
    for p in json.loads((output/'launch.json').read_text())['workers']:
        folder=output/f'worker_{p["worker"]:02d}'
        if (folder/'error.json').exists(): raise RuntimeError((folder/'error.json').read_text())
        status=json.loads((folder/'status.json').read_text())
        state=restore(folder)
        if status['status']!='complete' and physical_steps(state)<100:
            raise RuntimeError('First physical checkpoint not ready yet')
        assert state['contract']['manifest_sha256']==sha(output/'manifest.json')
        if status['status']!='complete':
            proc=Path(f'/proc/{p["pid"]}')
            assert [v.decode() for v in (proc/'cmdline').read_bytes().split(b'\0') if v]==p['command']
            threads=int(next(v.split(':')[1] for v in (proc/'status').read_text().splitlines() if v.startswith('Threads:')))
            assert threads==1
            target=roots[p['worker']::manifest['workers']][state['index']]
            env=state['env'] if state['stage']=='replay' else state['root']['env']
            source=census.load_source(output,target)
            census.assert_equal(env.events,source['events'][:len(env.events)],'startup exact prefix')
            assert env.nav.reset_count==1 and np.isfinite(env.nav.observation()).all()
            assert stream_hash(env._tasks)==target['stream_sha256']
        else:
            threads=0
        checks.append(dict(worker=p['worker'],pid=p['pid'],stage=state['stage'],threads=threads,
                           checkpoint_sha_verified=True,total_policy_steps=physical_steps(state),training_updates=0))
    write_json(output/'startup_health.json',dict(passed=True,workers=checks,roots=manifest['roots']))
    print(json.dumps(dict(passed=True,workers=len(checks))),flush=True)


def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def analyze_root(target,row):
    tasks=[r for r in row['one_step'] if r['spec']['kind']=='task_return']
    safe=[r for r in tasks if r['safe'] is True]
    scores={r['task_id']:r for r in row['continuations']}
    known={i:r['N_W'] for i,r in scores.items() if r['N_W'] is not None}
    complete=len(safe)>=2 and len(known)==len(safe)
    answer=dict(root_id=target['id'],policy=target['policy'],regime=target['regime'],safe_tasks=len(safe),
                unknown_tasks=sum(r['safe'] is None for r in tasks),total_tasks=len(tasks),
                multi_safe=len(safe)>=2,fully_resolved=complete,delta_N=None,heuristics={},reversals={})
    original=target.get('original_action',{})
    answer['original_selected_safe']=next((r['safe'] for r in tasks if r['spec']['task_id']==original.get('task_id')),None)
    if 'observation' in target:
        frozen=json.loads((PROJECT/'evidence/v1_2/calibration/frozen_regimes.json').read_text())
        b5=Scheduler('reserve_sjf',EstimateModel(**frozen['model']),.25).choose(target['observation'])
        selected=next((r for r in safe if r['spec']['task_id']==b5.task_id),None)
        answer['actual_B5_choice']=dict(action=asdict(b5),true_safe=next((r['safe'] for r in tasks if r['spec']['task_id']==b5.task_id),None),
            ranking_regret=max(known.values())-known[b5.task_id] if complete and selected else None)
    if len(safe)>=2:
        scalar={'Safe-SJF':lambda r:r['task_time'],'Safe-Energy':lambda r:r['task_energy'],
                'Safe-Reserve':lambda r:-r['actual_reserve'],
                'Safe-predicted-SJF':lambda r:r['prediction']['predicted_task_time']}
        for name,key in scalar.items():
            chosen=min(safe,key=lambda r:(key(r),r['spec']['task_id']))['spec']['task_id']
            answer['heuristics'][name]=dict(task_id=chosen,N_W=known.get(chosen),
                regret=(max(known.values())-known[chosen]) if complete else None)
        predicted=[r for r in safe if r['prediction']['estimated_feasible']]
        answer['predicted_and_true_safe_count']=len(predicted)
        if predicted:
            selected=min(predicted,key=lambda r:(r['prediction']['predicted_task_time'],r['spec']['task_id']))['spec']['task_id']
            answer['B5_safe_ranking']=dict(task_id=selected,N_W=known.get(selected),
                regret=(max(known.values())-known[selected]) if complete else None)
        if complete:
            answer['delta_N']=max(known.values())-min(known.values())
            answer['oracle_first_action']=min(known,key=lambda i:(-known[i],i))
            for name,field,direction in [('shorter','task_time',1),('less_energy','task_energy',1),('more_post_battery','post_task_battery',-1)]:
                pairs=[(a,b) for a in safe for b in safe if direction*a[field] < direction*b[field]-1e-7]
                answer['reversals'][name]=dict(ordered_comparable_pairs=len(pairs),
                    reversed_pairs=sum(known[a['spec']['task_id']]<known[b['spec']['task_id']] for a,b in pairs))
    return answer


def summarize(roots,rows):
    atlas=[analyze_root(t,r) for t,r in zip(roots,rows)]
    out=dict(roots=len(roots),one_step_branches=sum(len(r['one_step']) for r in rows),
             continuation_branches=sum(len(r['continuations']) for r in rows),groups={},root_comparisons=atlas)
    frozen=json.loads((PROJECT/'evidence/v1_2/calibration/frozen_regimes.json').read_text())
    batteries={r['id']:r['battery_tilde'] for r in frozen['regimes']}
    groups={'all':list(range(len(roots)))}
    for p in SOURCES: groups[p]=[i for i,t in enumerate(roots) if t['policy']==p]
    for b in (2,4,6): groups[f'B={b}']=[i for i,t in enumerate(roots) if batteries[t['regime']]==b]
    for p in SOURCES:
        for b in (2,4,6): groups[f'{p}/B={b}']=[i for i,t in enumerate(roots) if t['policy']==p and batteries[t['regime']]==b]
    for name,indices in groups.items():
        sub=[atlas[i] for i in indices]
        tasks=[a for i in indices for a in rows[i]['one_step'] if a['spec']['kind']=='task_return']
        pred=[a for a in tasks if a['prediction']['estimated_feasible']]
        known=[a for a in pred if a['safe'] is not None]
        continuations=[c for i in indices for c in rows[i]['continuations']]
        counts=Counter('>=2' if r['safe_tasks']>=2 else ('unknown' if r['unknown_tasks'] else str(r['safe_tasks'])) for r in sub)
        out['groups'][name]=dict(roots=len(sub),safe_task_count={k:counts[k] for k in ('0','1','>=2','unknown')},
            safe_task_fraction={k:counts[k]/len(sub) if sub else None for k in ('0','1','>=2','unknown')},
            predicted_feasible_tasks=len(pred),predicted_feasible_unknown=len(pred)-len(known),
            true_safe_given_predicted_feasible=mean([int(r['safe']) for r in known]),
            false_safe_rate=mean([int(not r['safe']) for r in known]),
            continuation_outcomes=dict(Counter(c['outcome'] for c in continuations)),
            N_W_mean=mean([c['N_W'] for c in continuations]),survival_mean=mean([c['F_W'] for c in continuations]),
            fully_resolved_multi_safe_roots=sum(r['fully_resolved'] for r in sub),
            unresolved_multi_safe_roots=sum(r['multi_safe'] and not r['fully_resolved'] for r in sub),
            delta_N_mean=mean([r['delta_N'] for r in sub]),
            delta_N_positive=sum(r['delta_N'] is not None and r['delta_N']>0 for r in sub),
            heuristic_regret={h:mean([r['heuristics'].get(h,{}).get('regret') for r in sub])
                for h in ('Safe-SJF','Safe-Energy','Safe-Reserve','Safe-predicted-SJF')},
            reversal_roots={k:sum(r['reversals'].get(k,{}).get('reversed_pairs',0)>0 for r in sub)
                for k in ('shorter','less_energy','more_post_battery')})
    out['limits']='Archived-policy occupancy, not stationary population sampling; equal cell allocation; paired deterministic branches. Oracle is privileged, not deployable. Unresolved diagnostic stops are unknown, not deaths or zero-throughput outcomes. No novelty or planning necessity follows automatically.'
    return out


def collect(output):
    manifest,roots=verify(output)
    by_id={}
    for i in range(manifest['workers']):
        folder=output/f'worker_{i:02d}'
        with exclusive_run(folder):
            assert json.loads((folder/'status.json').read_text())['status']=='complete'
            state=restore(folder)
            rows=json.loads((folder/'results.json').read_text())
            assert state['rows']==rows
            assigned=roots[i::manifest['workers']]
            assert [r['root_id'] for r in rows]==[t['id'] for t in assigned]
            for t,row in zip(assigned,rows):
                root=restore(folder/t['id']/'root')
                assert census.at_target(root['env'],t,census.load_source(output,t))
                assert stream_hash(root['env']._tasks)==t['stream_sha256']
                assert [r['spec'] for r in row['one_step']]==specs(root['env'])
                safe=[r['spec']['task_id'] for r in row['one_step'] if r['spec']['kind']=='task_return' and r['safe'] is True]
                assert [r['task_id'] for r in row['continuations']]==(safe if len(safe)>=2 else [])
                for r in row['one_step']+row['continuations']:
                    file=folder/r['file']; assert sha(file)==r['sha256']
                    assert {k:v for k,v in r.items() if k not in ('file','sha256')}==compact(json.loads(file.read_text()))
                assert t['id'] not in by_id
                by_id[t['id']]=row
    rows=[by_id[t['id']] for t in roots]
    for name,field in [('one_step_branches','one_step'),('continuation_branches','continuations')]:
        file=output/(name+'.jsonl')
        temporary=file.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for row in rows for r in row[field]))
        os.replace(temporary,file)
    write_json(output/'summary.json',summarize(roots,rows))
    write_json(output/'integrity.json',dict(passed=True,roots=len(roots),missing=0,duplicate=0,
        manifest_sha256=sha(output/'manifest.json'),one_step_sha256=sha(output/'one_step_branches.jsonl'),
        continuation_sha256=sha(output/'continuation_branches.jsonl'),summary_sha256=sha(output/'summary.json')))
    print(json.dumps(dict(complete=True,roots=len(roots))),flush=True)


def supervise(output):
    """Complete-only collection; never inspect scientific outcomes to change a run."""
    manifest,_=verify(output)
    while True:
        states=[]
        for i in range(manifest['workers']):
            folder=output/f'worker_{i:02d}'
            if (folder/'error.json').exists():
                write_json(output/'collection_blocked.json',dict(worker=i,error=json.loads((folder/'error.json').read_text())))
                return
            states.append(json.loads((folder/'status.json').read_text()).get('status') if (folder/'status.json').exists() else None)
        if all(s=='complete' for s in states):
            collect(output)
            return
        time.sleep(30)


def smoke(output):
    manifest,roots=verify(output)
    frozen=json.loads(Path(manifest['frozen']).read_text())
    model=EstimateModel(**frozen['model'])
    targets=[min((r for r in roots if r['policy']==p and r['decision_time']>0),key=lambda r:(r['decision_time'],r['id'])) for p in SOURCES]
    records=[]
    for t in targets:
        source=census.load_source(output,t)
        env=census.create_replay_env(t,frozen,source)
        while not census.at_target(env,t,source):
            assert not env.done and len(env.events)<t['decision_event_index']
            replay_step(env,t,source,model)
        root=clone_root(env,census.rng_state())
        before=pickle.dumps(root)
        branch=census.new_branch(root,specs(env)[0])
        advance_one(branch)
        saved=dict(branch=branch,rng=census.rng_state())
        sf=output/'engineering_smoke'/t['policy']
        snapshot(sf,saved); recovered=restore(sf)
        census.set_rng(saved['rng']); left=advance_one(branch)
        census.set_rng(recovered['rng']); right=advance_one(recovered['branch'])
        census.assert_equal(census.normalized(left),census.normalized(right),'atlas branch resume result')
        census.assert_equal(branch['env'].observe(),census.normalized(recovered['branch']['env'].observe()),'atlas branch resume observation')
        assert pickle.dumps(root)==before
        assert stream_hash(branch['env']._tasks)==t['stream_sha256']
        # Exercise the new nested-query state machine at this genuine decision
        # state, without pretending this engineering call followed a first task.
        c=new_continuation(root,dict(spec=specs(env)[0]))
        c['stage']='physical'
        advance_continuation(c,model)
        advance_continuation(c,model)
        assert c['stage']=='oracle' and c['oracle'] is not None
        snapshot(sf/'nested',dict(c=c,rng=census.rng_state()))
        recovered_c=restore(sf/'nested')
        parent_before=pickle.dumps(c['env'])
        left_c=advance_continuation(c,model)
        census.set_rng(recovered_c['rng'])
        right_c=advance_continuation(recovered_c['c'],model)
        census.assert_equal(census.normalized(left_c),census.normalized(right_c),'nested real resume result')
        census.assert_equal(c['oracle']['env'].observe(),census.normalized(recovered_c['c']['oracle']['env'].observe()),'nested real resume observation')
        assert pickle.dumps(c['env'])==parent_before and pickle.dumps(root)==before
        records.append(dict(root=t['id'],policy=t['policy'],exact_replay=True,clone_parent_unchanged=True,
                            resume_equal=True,nested_resume_equal=True,oracle_parent_unchanged=True,
                            replay_policy_steps=env.nav.policy_steps))
        c['oracle']['env'].nav.close();c['env'].nav.close()
        recovered_c['c']['oracle']['env'].nav.close();recovered_c['c']['env'].nav.close()
        env.nav.close();branch['env'].nav.close();recovered['branch']['env'].nav.close()
    write_json(output/'engineering_smoke.json',dict(passed=True,records=records,training_updates=0))
    print(json.dumps(dict(passed=True,real_replays=len(records))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','smoke','launch','worker','check','collect','supervise'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',type=int)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); output=args.output.resolve()
    if args.action=='worker':
        if args.worker is None or not 0<=args.worker<WORKERS: parser.error('valid --worker required')
        worker(output,args.worker,args.resume)
    else: globals()[args.action](output)


if __name__=='__main__': main()
