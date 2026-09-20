"""Exact design for independent single-bit probes and symmetric Bernoulli laws.

Cross-block hardest pairs differ in one currently testable bit. Optimum cycle
weights are proportional to 1/KL; Gamma_stage = 1/sum_e(duration_e/KL_e).
This is a local menu complexity, not the rejected all-pair root Gamma.
"""
def design(experiments):
    informative = [e for e in experiments if e.information() > 0]
    if not informative:
        return {}, 0.
    # Current suite has one active probe per unresolved bit.
    assert len({e.bit for e in informative}) == len(informative)
    inv = {e.name: 1/e.information() for e in informative}
    norm = sum(inv.values())
    gamma = 1/sum(e.duration*inv[e.name] for e in informative)
    return {k: v/norm for k, v in inv.items()}, gamma


def choose(experiments, counts):
    weights, gamma = design(experiments)
    if not weights:
        return None, gamma
    # Greedy deficit tracking of the optimal proportions for this menu.
    total = sum(counts.get(e.name, 0) for e in experiments)
    chosen = max(experiments, key=lambda e: (weights.get(e.name, 0)*(total+1)-counts.get(e.name, 0), e.name))
    return chosen, gamma
