"""Structural refinement and answer overlap, distinct from confidence updates."""
def observational_partition(models, experiments, law):
    blocks = {}
    for model in models:
        signature = tuple(law(model, e) for e in experiments)
        blocks.setdefault(signature, []).append(model)
    return list(blocks.values())


def common_answers(answer_sets):
    sets = list(map(set, answer_sets))
    return set.intersection(*sets) if sets else set()


def safe_experiments(experiments, version, static=False):
    return [e for e in experiments if (not e.prerequisites if static else e.safe(version))]


def safe_refinement_action(model, version, *, full_identification=False, static=False):
    """Shared decision rule for factored synthetic and finite UAV model APIs."""
    policy = model.common_optimal_policy(version)
    if policy is not None and (not full_identification or model.singleton(version)):
        return 'deploy', policy, None, []
    safe = model.safe_experiments(version, static=static)
    informative = model.informative_experiments(version, safe, decision_only=policy is None)
    experiment, gamma = model.information_design(version, informative)
    return ('unidentifiable' if experiment is None else 'experiment'), experiment, gamma, informative
