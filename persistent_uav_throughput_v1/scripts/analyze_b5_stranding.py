"""Offline B5 stranding decomposition from published JSON telemetry only.

Standard library only: no navigator import, pickle load, simulator, or new policy.
Failure-leg energy is censored consumed energy, never a completed-route label.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import re
import statistics
import tarfile

TOL = 1e-7


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def close(a, b):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a - b) > TOL:
        raise AssertionError((a, b))


def predict_energy(model, start, goal):
    if math.dist(start, goal) <= model['goal_radius']:
        return 0.
    c = model['energy_coefficients']
    return max(0., c[0] + c[1] * math.hypot(goal[0] - start[0], goal[1] - start[1])
               + c[2] * abs(goal[2] - start[2]))


def decompose(e_before, predicted_task, predicted_return_goal, e_post, predicted_return_endpoint,
              e_return, predicted_return_at_departure):
    """Exact accounting identity, not a claim about counterfactual true feasibility."""
    actual_task = e_before - e_post
    m_pre = e_before - predicted_task - predicted_return_goal
    task_error = actual_task - predicted_task
    endpoint_shift = predicted_return_endpoint - predicted_return_goal
    m_post = e_post - predicted_return_endpoint
    between_energy = e_post - e_return
    later_estimate_shift = predicted_return_at_departure - predicted_return_endpoint
    m_return = e_return - predicted_return_at_departure
    task_identity_error = m_pre - task_error - endpoint_shift - m_post
    return_identity_error = m_post - between_energy - later_estimate_shift - m_return
    close(task_identity_error, 0.)
    close(return_identity_error, 0.)
    return dict(e_before=e_before, predicted_task_energy=predicted_task,
                predicted_return_goal=predicted_return_goal, actual_task_energy=actual_task,
                e_post=e_post, predicted_return_actual_endpoint=predicted_return_endpoint,
                m_pre=m_pre, task_energy_error=task_error, endpoint_return_shift=endpoint_shift,
                m_post=m_post, e_return=e_return, predicted_return_at_departure=predicted_return_at_departure,
                post_task_battery_loss=between_energy, later_return_estimate_shift=later_estimate_shift,
                m_return=m_return, task_identity_error=task_identity_error,
                return_identity_error=return_identity_error,
                m_post_if_task_energy_prediction_exact=m_pre - endpoint_shift,
                m_post_if_endpoint_estimate_unchanged=m_pre - task_error)


def return_category(d, fallback):
    """Mutually exclusive descriptive ordering; ancillary flags preserve overlap."""
    if fallback:
        return 'fallback_predicted_infeasible_before_task'
    if d['m_return'] > 0:
        return 'return_prediction_false_safe'
    if d['m_post'] > 0:
        return 'post_task_waiting_margin_loss'
    if d['m_pre'] > 0:
        if d['m_post_if_endpoint_estimate_unchanged'] <= 0:
            return 'task_underestimation_sufficient_for_margin_flip'
        if d['m_post_if_task_energy_prediction_exact'] <= 0:
            return 'endpoint_shift_sufficient_for_margin_flip'
        return 'combined_task_and_endpoint_margin_flip'
    return 'unresolved_nonpositive_pre_margin_without_fallback'


def failed_energy(e_before, residual, prediction):
    """Only energy spent before absorption is observed; full-route cost is unknown."""
    consumed = e_before - residual
    return dict(consumed_energy=consumed, full_leg_energy=None,
                full_leg_energy_censored=True, completion_energy_lower_bound=consumed,
                prediction_error_lower_bound=consumed - prediction)


def quantile(values, p):
    values = sorted(values)
    x = (len(values) - 1) * p
    lo, hi = math.floor(x), math.ceil(x)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def stats(values):
    values = list(values)
    if not values:
        return dict(n=0)
    return dict(n=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), q95=quantile(values, .95), max=max(values))


def analyze_run(run, source, model):
    summary, events = run['summary'], run['events']
    base = {k: summary[k] for k in ('regime', 'method', 'threshold', 'seed')}
    base.update(source_member=source, run_failure=summary['failure'])
    decision_indices = [i for i, e in enumerate(events) if e['event'] == 'decision']
    assert len(decision_indices) == len(run['decision_diagnostics'])
    completed, returns, failures = [], [], []
    for number, index in enumerate(decision_indices):
        decision = events[index]
        action, obs, prediction = decision['action'], decision['observation'], decision['reserve_prediction']
        saved = run['decision_diagnostics'][number]
        assert saved['action'] == action and saved['decision_index'] == number
        assert saved['selected'] == prediction['selected']
        close(obs['battery'], prediction['starting_battery'])
        close(saved['starting_battery'], obs['battery'])
        direct = predict_energy(model, obs['position'], obs['charger_position'])
        close(direct, prediction['predicted_direct_return_energy'])
        close(obs['battery'] - direct, prediction['direct_return_margin'])
        by_id = {t['id']: t for t in obs['queue']}
        assert len(by_id) == len(prediction['candidates'])
        for c in prediction['candidates']:
            task = by_id[c['task_id']]
            close(c['predicted_task_energy'], predict_energy(model, obs['position'], task['position']))
            close(c['predicted_return_energy'], predict_energy(model, task['position'], obs['charger_position']))
            close(c['predicted_reserve_margin'], obs['battery'] - c['predicted_task_energy'] - c['predicted_return_energy'])
            assert c['estimated_feasible'] == (c['predicted_task_energy'] + c['predicted_return_energy'] < obs['battery'])
        stop = decision_indices[number + 1] if number + 1 < len(decision_indices) else len(events)
        kind = {'serve': 'task_completed', 'recharge': 'charger_arrival'}.get(action['kind'])
        terminal = next(((j, events[j]) for j in range(index + 1, stop)
                         if events[j]['event'] in (kind, 'failure', 'evaluation_cutoff')), None)
        if terminal is None:
            assert action['kind'] == 'idle'
            continue
        end_index, end = terminal
        anchor = dict(base, decision_index=number, decision_event_index=index, terminal_event_index=end_index,
                      action=action, start_time=decision['time'], end_time=end['time'])
        if action['kind'] == 'serve' and end['event'] == 'task_completed':
            assert end['task_id'] == action['task_id']
            selected = prediction['selected']
            close(saved['actual_leg_energy'], obs['battery'] - end['battery'])
            at_endpoint = predict_energy(model, end['position'], obs['charger_position'])
            d = decompose(obs['battery'], selected['predicted_task_energy'], selected['predicted_return_energy'],
                          end['battery'], at_endpoint, end['battery'], at_endpoint)
            close(d['m_pre'], selected['predicted_reserve_margin'])
            completed.append(dict(anchor, **d, task_id=action['task_id'], selected_estimated_feasible=selected['estimated_feasible'],
                                  fallback=action['reason'] == 'full_station_infeasible_estimate_fallback',
                                  endpoint_position=end['position'], charger_position=obs['charger_position'],
                                  goal_position=by_id[action['task_id']]['position'],
                                  initial_direct_return_margin=obs['battery'] - direct,
                                  endpoint_distance_from_goal=math.dist(end['position'], by_id[action['task_id']]['position'])))
            following = (run['decision_diagnostics'][number + 1]
                         if number + 1 < len(run['decision_diagnostics']) else None)
            completed[-1].update(next_action=None if following is None else following['action']['kind'],
                                 next_leg_outcome=None if following is None else following['leg_outcome'],
                                 next_leg_failure=None if following is None else following.get('leg_failure'))
        if action['kind'] == 'recharge':
            returned = dict(anchor, predicted_energy=direct, starting_battery=obs['battery'],
                            predicted_margin=obs['battery'] - direct, outcome=end['event'],
                            failure_reason=end.get('reason'), actual_completed_energy=None)
            if end['event'] == 'charger_arrival':
                returned['actual_completed_energy'] = obs['battery'] - end['battery']
                close(saved['actual_leg_energy'], returned['actual_completed_energy'])
                returned['energy_prediction_error'] = returned['actual_completed_energy'] - direct
            elif end['event'] == 'failure':
                returned.update(failed_energy(obs['battery'], end['residual_battery'], direct))
            returns.append(returned)
        if end['event'] != 'failure':
            continue
        assert end['reason'] == summary['failure']
        phase = end['failure_phase']
        record = dict(anchor, failure_phase=phase, depletion=end['reason'] == 'energy_depletion',
                      navigation_failure=end['reason'] == 'navigation_failure',
                      starting_battery=obs['battery'], starting_direct_return_margin=obs['battery'] - direct,
                      duration=end['time'] - decision['time'])
        if phase == 'task':
            selected = prediction['selected']
            record.update(selected=selected, task_id=action['task_id'],
                          fallback=action['reason'] == 'full_station_infeasible_estimate_fallback',
                          **failed_energy(obs['battery'], end['residual_battery'], selected['predicted_task_energy']))
            record['category'] = ('task_prediction_false_safe' if selected['estimated_feasible']
                                  else 'task_depletion_after_infeasible_fallback')
            close(saved['actual_leg_energy'], record['consumed_energy'])
        elif phase == 'return':
            assert completed, 'Return failure without a preceding completed task requires separate handling'
            previous = completed[-1]
            middle = [e for e in events[previous['terminal_event_index'] + 1:index] if e['event'] == 'decision']
            assert all(e['action']['kind'] == 'idle' for e in middle)
            close(math.dist(previous['endpoint_position'], obs['position']), 0.)
            d = decompose(previous['e_before'], previous['predicted_task_energy'], previous['predicted_return_goal'],
                          previous['e_post'], previous['predicted_return_actual_endpoint'], obs['battery'], direct)
            close(d['m_pre'], previous['m_pre'])
            close(d['m_return'], prediction['direct_return_margin'])
            record.update(d, previous_task_id=previous['task_id'], previous_task_decision_event_index=previous['decision_event_index'],
                          previous_task_completion_event_index=previous['terminal_event_index'],
                          previous_task_action_reason=previous['action']['reason'], fallback=previous['fallback'],
                          selected_estimated_feasible=previous['selected_estimated_feasible'],
                          intervening_idle_decisions=len(middle), post_task_delay=decision['time'] - previous['end_time'],
                          positive_post_to_nonpositive_return=d['m_post'] > 0 and d['m_return'] <= 0,
                          **failed_energy(obs['battery'], end['residual_battery'], direct))
            record['category'] = return_category(d, previous['fallback'])
            close(saved['actual_leg_energy'], record['consumed_energy'])
            # Battery loss between task completion and return is solely from recorded idle intervals.
            if not middle:
                close(d['post_task_battery_loss'], 0.)
            else:
                close(d['post_task_battery_loss'], previous['e_post'] - obs['battery'])
        elif phase == 'waiting':
            assert action['kind'] == 'idle' and not obs['queue']
            record.update(category='waiting_depletion', consumed_energy=obs['battery'] - end['residual_battery'],
                          starting_soc=obs['battery'] / obs['capacity'],
                          distance_to_charger=math.dist(obs['position'], obs['charger_position']),
                          queue_empty=True)
        else:
            raise AssertionError(('Unexpected phase', phase))
        if not record['depletion']:
            record['category'] = 'navigation_failure_not_depletion'
        failures.append(record)
    assert len(completed) == summary['completed']
    assert len(failures) == int(summary['depletion'] or summary['navigation_failure'])
    return completed, returns, failures


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def analyze(evidence, frozen_path, output):
    # Read published archives without extracting or unpickling any checkpoint.
    package_hashes = {}
    for line in (evidence / 'SHA256SUMS').read_text().splitlines():
        expected, name = line.split('  ', 1)
        actual = digest((evidence / name).read_bytes())
        assert actual == expected, name
        package_hashes[name] = actual
    frozen = json.loads(frozen_path.read_text())
    rows = json.loads((evidence / 'results.json').read_text())
    inventory = {r['path']: r for r in json.loads((evidence / 'archive_inventory.json').read_text())}
    runs, plan = [], None
    with tarfile.open(evidence / 'validation_raw.tar.gz', 'r:gz') as archive:
        assert set(archive.getnames()) == set(inventory)
        for member in archive:
            assert member.isfile()
            raw = archive.extractfile(member).read()
            item = inventory[member.name]
            assert len(raw) == item['bytes'] and digest(raw) == item['sha256'], member.name
            if re.fullmatch(r'worker_\d{2}/run_\d{5}\.json', member.name):
                runs.append((member.name, json.loads(raw)))
            elif member.name == 'plan.json':
                plan = json.loads(raw)
    assert plan['frozen_sha256'] == digest(frozen_path.read_bytes())
    key = lambda r: (r['regime'], r['method'], r['threshold'], r['seed'])
    expected = set(itertools.product(range(27), ['reserve_sjf'], [.25], range(920260920, 920260930)))
    assert len(runs) == len(rows) == 270
    assert {key(r) for r in rows} == expected
    assert sorted((r['summary'] for _, r in runs), key=key) == sorted(rows, key=key)
    completed, returns, failures = [], [], []
    for source, run in runs:
        c, r, f = analyze_run(run, source, frozen['model'])
        completed.extend(c)
        returns.extend(r)
        failures.extend(f)
    depleted = [r for r in failures if r['depletion']]
    return_failures = [r for r in depleted if r['failure_phase'] == 'return']
    flight = [r for r in depleted if r['failure_phase'] in ('task', 'return')]
    flip = [r for r in return_failures if r['category'] == 'task_underestimation_sufficient_for_margin_flip']
    wait_flip = [r for r in return_failures if r['positive_post_to_nonpositive_return']]
    direct_false = [r for r in return_failures if r['m_return'] > 0]
    task_false = [r for r in depleted if r['category'] == 'task_prediction_false_safe']
    categories = dict(sorted(Counter(r['category'] for r in depleted).items()))
    battery = {r['id']: r['battery_tilde'] for r in frozen['regimes']}
    metrics = dict(runs=270, completed_tasks=len(completed), total_return_decisions=len(returns),
                   depletion=len(depleted), navigation_failure=sum(r['navigation_failure'] for r in rows),
                   horizon_survivors=sum(not r['depletion'] and not r['navigation_failure'] for r in rows),
                   flight_depletion=len(flight), depletion_phase_counts=dict(Counter(r['failure_phase'] for r in depleted)),
                   mutually_exclusive_depletion_categories=categories,
                   denominator_notes='Categories partition observed depletions; not causal fractions or full-policy counterfactuals.',
                   return_decomposition=dict(n=len(return_failures),
                       preceding_task_fallback=sum(r['fallback'] for r in return_failures),
                       accepted_task_positive_pre_margin=sum(r['m_pre'] > 0 for r in return_failures),
                       task_error_sufficient_positive_to_nonpositive_post=len(flip),
                       endpoint_only_or_joint_flip=sum(r['category'] in ('endpoint_shift_sufficient_for_margin_flip', 'combined_task_and_endpoint_margin_flip') for r in return_failures),
                       positive_post_to_nonpositive_departure=len(wait_flip),
                       wait_flip_preceded_by_fallback=sum(r['fallback'] for r in wait_flip),
                       nonpositive_return_departure=sum(r['m_return'] <= 0 for r in return_failures),
                       positive_return_departure_false_safe=len(direct_false),
                       idle_between_task_and_return=sum(r['intervening_idle_decisions'] > 0 for r in return_failures),
                       max_abs_task_identity_error=max(abs(r['task_identity_error']) for r in return_failures),
                       max_abs_return_identity_error=max(abs(r['return_identity_error']) for r in return_failures)),
                   subgroup_statistics=dict(
                       all_completed_task_energy_error=stats(r['task_energy_error'] for r in completed),
                       preceding_task_error_return_failures=stats(r['task_energy_error'] for r in return_failures),
                       endpoint_shift_return_failures=stats(r['endpoint_return_shift'] for r in return_failures),
                       task_flip_pre_margin=stats(r['m_pre'] for r in flip),
                       task_flip_task_error=stats(r['task_energy_error'] for r in flip),
                       task_flip_endpoint_shift=stats(r['endpoint_return_shift'] for r in flip),
                       wait_flip_energy_loss=stats(r['post_task_battery_loss'] for r in wait_flip),
                       wait_flip_task_error=stats(r['task_energy_error'] for r in wait_flip),
                       positive_return_false_safe_margin=stats(r['m_return'] for r in direct_false),
                       task_depletion_prediction_error_lower_bound=stats(r['prediction_error_lower_bound'] for r in task_false),
                       positive_return_false_safe_previous_task_error=stats(r['task_energy_error'] for r in direct_false)),
                   all_completed_task_transitions=dict(
                       estimated_feasible=sum(r['selected_estimated_feasible'] for r in completed),
                       fallback=sum(r['fallback'] for r in completed),
                       positive_pre_to_nonpositive_post=sum(r['m_pre'] > 0 and r['m_post'] <= 0 for r in completed),
                       nonpositive_post=sum(r['m_post'] <= 0 for r in completed),
                       positive_to_nonpositive_then_successful_return=sum(
                           r['m_pre'] > 0 and r['m_post'] <= 0 and r['next_action'] == 'recharge'
                           and r['next_leg_outcome'] == 'charger_arrival' for r in completed),
                       positive_to_nonpositive_then_return_depletion=sum(
                           r['m_pre'] > 0 and r['m_post'] <= 0 and r['next_action'] == 'recharge'
                           and r['next_leg_failure'] == 'energy_depletion' for r in completed)),
                   observed_return_prediction_table={
                       'positive_margin_arrival': sum(r['predicted_margin'] > 0 and r['outcome'] == 'charger_arrival' for r in returns),
                       'positive_margin_depletion': sum(r['predicted_margin'] > 0 and r['failure_reason'] == 'energy_depletion' for r in returns),
                       'nonpositive_margin_arrival': sum(r['predicted_margin'] <= 0 and r['outcome'] == 'charger_arrival' for r in returns),
                       'nonpositive_margin_depletion': sum(r['predicted_margin'] <= 0 and r['failure_reason'] == 'energy_depletion' for r in returns)},
                   by_battery_tilde={str(b):dict(runs=sum(battery[r['regime']] == b for r in rows),
                       depletion_categories=dict(Counter(r['category'] for r in depleted if battery[r['regime']] == b)))
                       for b in sorted(set(battery.values()))})
    assert sum(categories.values()) == len(depleted)
    # Directly reconcile the previously published failure audit.
    published = json.loads((evidence / 'initial_failure_audit.json').read_text())['counts']
    assert len(direct_false) == published['return_depletion_positive_predicted_margin']
    assert len(return_failures) - len(direct_false) == published['return_depletion_nonpositive_predicted_margin']
    assert len(task_false) == published['task_depletion_predicted_feasible']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'summary.json', metrics)
    write_json(output / 'failure_cases.json', failures)
    for name, data in [('completed_task_legs.json.gz', completed), ('return_legs.json.gz', returns)]:
        # Stable gzip metadata permits exact artifact regeneration.
        (output / name).write_bytes(gzip.compress((json.dumps(data, allow_nan=False) + '\n').encode(), mtime=0))
    write_json(output / 'input_integrity.json', dict(passed=True, published_runs=270,
        raw_archive_members_verified=len(inventory), package_hashes=package_hashes,
        frozen_sha256=digest(frozen_path.read_bytes()), archive_rows_match_summary=True,
        all_candidate_energy_predictions_recomputed=True, completed_task_energy_matches_saved_telemetry=True,
        prior_failure_audit_reconciled=True, identity_tolerance=TOL,
        analyzer_sha256=digest(Path(__file__).read_bytes()), python_version=platform.python_version(),
        simulations_executed=0, checkpoints_unpickled=0, policy_or_model_changes=False))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--frozen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.evidence, args.frozen, args.output)


if __name__ == '__main__':
    main()
