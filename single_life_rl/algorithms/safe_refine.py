"""One engine; baselines differ only in frozen menu / stopping / knowledge.

LR boundaries have simultaneous lifetime coverage via Ville plus per-bit budget.
There is no reset after catastrophe. Independent evaluation runs are distinct lives.
"""
import math
import numpy as np
from single_life_rl.core.model import VersionSet
from single_life_rl.core.refinement import safe_experiments, safe_refinement_action
from single_life_rl.core.information import choose
from single_life_rl.envs.families import experiments, FactoredModels


def run_toy(job, protocol):
    method, truth = job['method'], tuple(job['truth'])
    exps, decision_dims, dims = experiments(job)
    T, delta = job['horizon'], protocol['delta']
    v = VersionSet(dims)
    rngs = {e.name: np.random.default_rng(np.random.SeedSequence([job['seed'], i, 771])) for i,e in enumerate(exps)}
    buffers, offsets = {}, {}
    def observe(e):
        pos = offsets.get(e.name, 0)
        if e.name not in buffers or pos == len(buffers[e.name]):
            p = .5+(2*truth[e.bit]-1)*e.kappa
            buffers[e.name] = rngs[e.name].binomial(1, p, size=256)
            pos = 0
        offsets[e.name] = pos+1
        return int(buffers[e.name][pos])
    t = reward = explore = safe_explore = 0.
    cycles = probes = safe_probes = stages = 0
    catastrophic = eliminated = False
    cert = None
    model_identified = None
    counts, path_gammas, events = {}, [], []
    models = FactoredModels(job, counts)
    previous_menu = None
    initial_safe = {e.name for e in safe_experiments(exps, v)}
    unlocked = set(initial_safe)
    status = 'horizon_censored'
    if method == 'Oracle':
        v.known = dict(enumerate(truth)); cert = 0.; model_identified = list(truth)
    elif method in ('RobustOnly',):
        cycles = int(T); reward = protocol['baseline_reward']*cycles; t = T
        status = 'robust_only'
    elif method == 'UnsafeMLE':
        # Greedy point-estimate action before any evidence; lexicographic MLE tie.
        candidate = (0,)*decision_dims
        catastrophic = candidate != truth[:decision_dims]
        cycles = 1 if catastrophic else int(T)
        reward = 0. if catastrophic else float(cycles)
        t = 1. if catastrophic else T
        status = 'catastrophe' if catastrophic else 'uncertified_exploitation'
    else:
        while t < T:
            policy = v.common_optimal_policy(decision_dims)
            if cert is None and policy is not None:
                cert = t
            if len(v.known) == dims and model_identified is None:
                model_identified = [v.known[i] for i in range(dims)]
            kind, action, gamma, candidates = safe_refinement_action(models, v, full_identification=method=='FullModelID', static=method=='StaticSafeID')
            if kind == 'deploy':
                status = 'certified'; break
            safe = safe_experiments(exps, v, static=method == 'StaticSafeID')
            unlocked.update(e.name for e in safe)
            # Same design framework and ordering; FullModelID alone continues
            # past common-policy certification to resolve nuisance parameters.
            e = action
            if e is None:
                status = 'unidentifiable_with_available_menu'
                rest = int(T-t); cycles += rest; reward += rest*protocol['baseline_reward']; t = T
                break
            menu = tuple(e.name for e in candidates)
            if menu != previous_menu:
                path_gammas.append({'time': t, 'gamma': gamma, 'menu': menu})
                previous_menu = menu
            if t+e.duration > T:
                explore += T-t; t = T; break
            t += e.duration; explore += e.duration; cycles += 1; probes += 1
            counts[e.name] = counts.get(e.name,0)+1
            if not e.actually_safe(truth):
                catastrophic = True; status = 'catastrophe'; break
            safe_probes += 1
            safe_explore += e.duration
            reward += protocol['probe_reward']
            budget = delta if dims == 1 else (delta/(4*(dims-2)) if job['suite']=='nuisance' and e.bit>=2 else delta*2.**(-(e.bit+1)))
            changed = v.update(e, observe(e), budget)
            eliminated |= not v.includes(truth)
            if changed:
                stages += 1
                events.append({'time': t, 'bit': e.bit, 'value': v.known[e.bit], 'cardinality': v.cardinality(), 'true_model_in_set': v.includes(truth)})
        # Record certification even when the last observation lands on horizon.
        policy = v.common_optimal_policy(decision_dims)
        if cert is None and policy is not None and not catastrophic:
            cert = t
        if len(v.known) == dims and model_identified is None:
            model_identified = [v.known[i] for i in range(dims)]
    if method == 'Oracle' or status == 'certified':
        policy = v.common_optimal_policy(decision_dims)
        remaining = int(T-t)
        if remaining and policy != truth[:decision_dims]:
            catastrophic = True; t += 1; cycles += 1; status = 'catastrophe'
        else:
            reward += remaining; cycles += remaining; t = T
    # Task horizon is one continuing life; terminal reward is held after failure.
    relevant_path = []
    for i in range(decision_dims):
        e = next(e for e in exps if e.bit == i and e.actually_safe(truth))
        relevant_path.append(e.duration/e.information() if e.information() else None)
    return dict(job_id=job['job_id'], suite=job['suite'], method=method, spec=job,
                catastrophe=catastrophic, reward=reward, regret=T-reward, oracle_rate=1.,
                certification_time=cert, certification_censored=cert is None,
                correct_certificate=None if cert is None else v.common_optimal_policy(decision_dims)==truth[:decision_dims],
                exploration_time=explore, safe_exploration_time=safe_explore, cycles=cycles, safe_probe_count=safe_probes,
                attempted_probes=probes, refinement_stages=stages, model_set_cardinality=v.cardinality(),
                policy_answer_count=2**sum(i not in v.known for i in range(decision_dims)),
                unlocked_experiments=len(unlocked-initial_safe), true_model_eliminated=eliminated,
                identified_model=model_identified, full_identification_time=(events[-1]['time'] if model_identified is not None and events else (0. if method=='Oracle' else None)),
                log_odds=v.log_odds, probe_counts=counts, expected_log_likelihood_information=sum(counts.get(e.name,0)*e.information() for e in exps), refinement_events=events,
                information_path=path_gammas, sum_inverse_stage_information=None if None in relevant_path else sum(relevant_path), status=status)
