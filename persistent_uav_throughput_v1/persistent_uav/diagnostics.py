"""Passive B5 decision/leg telemetry; never changes a scheduler input or action."""


def reserve_prediction(observation, action, model):
    o = observation
    candidates = []
    for task in o['queue']:
        duration, task_energy = model.predict(o['position'], task['position'])
        _, return_energy = model.predict(task['position'], o['charger_position'])
        total = task_energy + return_energy
        candidates.append(dict(task_id=task['id'], predicted_task_time=duration,
                               predicted_task_energy=task_energy,
                               predicted_return_energy=return_energy,
                               predicted_reserve_margin=o['battery'] - total,
                               estimated_feasible=total < o['battery']))
    selected = next((c for c in candidates if c['task_id'] == action.task_id), None)
    _, direct_return = model.predict(o['position'], o['charger_position'])
    return dict(schema='reserve_decision_v1', starting_battery=o['battery'],
                candidates=candidates, selected=selected,
                predicted_direct_return_energy=direct_return,
                direct_return_margin=o['battery'] - direct_return,
                margin_added=0., strict_feasibility=True)


def audited_step(env, action, model):
    """Enrich existing events after the unchanged environment step returns."""
    prediction = reserve_prediction(env.observe(), action, model) if action is not None else None
    phase = {'serve': 'task', 'recharge': 'return', 'idle': 'waiting'}.get(
        action.kind if action is not None else None,
        {'SERVING': 'task', 'RETURNING': 'return', 'CHARGING': 'charging',
         'WAITING': 'waiting', 'IDLE': 'waiting'}.get(env.mode))
    start = len(env.events)
    result = env.step(action)
    for event in env.events[start:]:
        if event['event'] == 'decision' and prediction is not None:
            event['reserve_prediction'] = prediction
        elif event['event'] == 'failure':
            event['failure_phase'] = phase
    return result


def run_diagnostics(events, summary):
    """Per-decision outcomes plus run-level flags; predicted return is hypothetical.

    A served task's predicted return is not an executed return unless the next
    action chooses recharge. Actual energy below refers only to the chosen leg.
    """
    failure = next((e for e in events if e['event'] == 'failure'), None)
    phase = failure.get('failure_phase') if failure else None
    flags = dict(depletion=summary['depletion'], navigation_failure=summary['navigation_failure'],
                 recharge_failure=summary['recharge_failure'], failure_phase=phase,
                 depletion_phase=phase if summary['depletion'] else None)
    decisions = []
    indices = [i for i, e in enumerate(events) if e['event'] == 'decision']
    for number, index in enumerate(indices):
        event = events[index]
        stop = indices[number + 1] if number + 1 < len(indices) else len(events)
        action = event['action']
        terminal_kind = {'serve': 'task_completed', 'recharge': 'charger_arrival'}.get(action['kind'])
        terminal = next((e for e in events[index + 1:stop]
                         if e['event'] in (terminal_kind, 'failure', 'evaluation_cutoff')), None)
        audit = dict(decision_index=number, time=event['time'], action=action,
                     **event['reserve_prediction'], run_outcome=dict(flags),
                     leg_outcome=None, actual_leg_energy=None)
        if terminal is not None:
            audit['leg_outcome'] = terminal['event']
            audit['leg_failure'] = terminal.get('reason')
            audit['leg_failure_phase'] = terminal.get('failure_phase')
            battery = terminal.get('battery', terminal.get('residual_battery'))
            if action['kind'] in ('serve', 'recharge') and battery is not None:
                audit['actual_leg_energy'] = audit['starting_battery'] - battery
        decisions.append(audit)
    return flags, decisions
