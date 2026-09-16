"""Implementation-diverse audit of the existing T=12 theorem anchor.

Normalized Bayesian state recursion and explicit reward-vector evaluation.
Does not import original verifier/learner, optimize a minimax mixture, sample
trajectories, or evaluate new horizons. This is not independent human review.
"""
import argparse
from fractions import Fraction as F
from functools import lru_cache
import hashlib
from itertools import product
import json
from pathlib import Path

MEANS = ((F(1, 20), F(1, 20)), (F(3, 10), F(1, 20)),
         (F(3, 10), F(19, 20)))
GAINS = (F(1, 20), F(1, 10), F(19, 60))
ZERO = (F(0),) * 3
ROUTES = (("dock", 1, ()), ("A", 3, (0,)), ("B", 3, (1,)),
          ("AB", 4, (0, 1)), ("BA", 4, (1, 0)), ("empty", 2, ()))


def add(*vectors):
    return tuple(sum(v[i] for v in vectors) for i in range(3))


def information_state(prior):
    """Normalized beliefs, unlike the original unnormalized value recursion."""
    prior = tuple(map(F, prior))
    assert sum(prior) == 1 and min(prior) >= 0

    @lru_cache(None)
    def belief(counts):
        a, sa, b, sb = counts
        weights = tuple(prior[i] * x**sa * (1-x)**(a-sa)
                        * y**sb * (1-y)**(b-sb)
                        for i, (x, y) in enumerate(MEANS))
        total = sum(weights)
        assert total > 0
        return tuple(w/total for w in weights)

    def outcomes(counts, channels):
        for bits in product((0, 1), repeat=len(channels)):
            updated = list(counts)
            probability = [F(1)] * 3
            for channel, bit in zip(channels, bits):
                updated[2*channel] += 1
                updated[2*channel+1] += bit
                for i in range(3):
                    p = MEANS[i][channel]
                    probability[i] *= p if bit else 1-p
            yield tuple(updated), tuple(probability)

    def choose(counts, candidates):
        p = belief(counts)
        return max(candidates, key=lambda vector: sum(p[i]*vector[i] for i in range(3)))

    return belief, outcomes, choose


def primitive_low(T, prior):
    """All safe low-capacity actions; arbitrary legal terminal positions."""
    belief, outcomes, choose = information_state(prior)

    @lru_cache(None)
    def value(t, phase, battery, counts):
        if t == 0:
            return ZERO
        if phase == "reset":
            assert battery == 3
            return choose(counts, [add((F(1, 20),)*3,
                                      value(t-1, "reset", 3, counts)),
                                   value(t-1, "prepared", 2, counts)])
        assert battery >= 1
        candidates = [value(t-1, "reset", 3, counts)]
        if phase == "prepared" and battery >= 2:
            for channel in (0, 1):
                reward = tuple(MEANS[i][channel] for i in range(3))
                future = list(ZERO)
                for updated, probabilities in outcomes(counts, (channel,)):
                    vector = value(t-1, "measured", battery-1, updated)
                    for i in range(3):
                        future[i] += probabilities[i]*vector[i]
                candidates.append(add(reward, tuple(future)))
        return choose(counts, candidates)

    rewards = value(T, "reset", 3, (0, 0, 0, 0))
    risk = tuple(T*g-y for g, y in zip(GAINS, rewards))
    lower = sum(p*r for p, r in zip(map(F, prior), risk))
    return risk, lower, value.cache_info().currsize


def committed_terminal(T, prior, capacity):
    """Actual feedback-vector evaluation of a normalized Bayes route policy.

    Empty and BA routes included, not omitted by transcription. Docking feedback
    is independent of the label and can be ignored by a Bayesian best response.
    """
    _, outcomes, choose = information_state(prior)
    routes = tuple(route for route in ROUTES if route[1] <= capacity)

    @lru_cache(None)
    def value(t, counts):
        if t == 0:
            return ZERO
        candidates = []
        for name, length, channels in routes:
            if length > t:
                continue
            reward = ((F(1, 20),)*3 if name == "dock" else
                      tuple(sum((MEANS[i][c] for c in channels), F(0)) for i in range(3)))
            future = list(ZERO)
            for updated, probabilities in outcomes(counts, channels):
                vector = value(t-length, updated)
                for i in range(3):
                    future[i] += probabilities[i]*vector[i]
            candidates.append(add(reward, tuple(future)))
        return choose(counts, candidates)

    reward = value(T, (0, 0, 0, 0))
    return tuple(T*g-y for g, y in zip(GAINS, reward))


def static_terminal_certificate():
    prior = (F(21, 23), F(0), F(2, 23))
    price = F(8, 345)
    for name, length, channels in ROUTES:
        reward = ((F(1, 20),)*3 if name == "dock" else
                  tuple(sum((MEANS[i][c] for c in channels), F(0)) for i in range(3)))
        costs = tuple(g*length-y for g, y in zip(GAINS, reward))
        assert sum(p*c for p, c in zip(prior, costs)) >= price*length
    def schedule(n_ab, n_dock):
        assert 4*n_ab+n_dock == 12
        return tuple(12*g-n_ab*(a+b)-n_dock*F(1, 20)
                     for g, (a, b) in zip(GAINS, MEANS))
    risks = tuple(F(5, 46)*x+F(41, 46)*y
                  for x, y in zip(schedule(1, 8), schedule(3, 0)))
    assert risks == (F(32, 115), F(21, 115), F(32, 115))
    assert max(risks) == 12*price
    return risks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.certificate.read_bytes()
    row = next(x for x in json.loads(raw)["results"] if x["T"] == 12)
    low_risk, lower, states = primitive_low(12, row["low_prior"])
    assert lower == F(row["lower"])
    assert low_risk == tuple(map(F, row["low_Bayes_optimal_risks"]))
    mixed = list(ZERO)
    weights = [F(w["weight"]) for w in row["high_witnesses"]]
    assert min(weights) >= 0 and sum(weights) == 1
    for witness, weight in zip(row["high_witnesses"], weights):
        risk = committed_terminal(12, witness["prior"], 4)
        assert risk == tuple(map(F, witness["risks"]))
        for i in range(3):
            mixed[i] += weight*risk[i]
    assert tuple(mixed) == tuple(map(F, row["high_combined_risks"]))
    upper = max(mixed)
    assert upper == F(row["upper"])
    gap = lower-upper
    assert gap == F(row["gap"])
    for T in (12, 24):
        for capacity in (3, 4):
            for i in range(3):
                p = tuple(F(int(j == i)) for j in range(3))
                assert committed_terminal(T, p, capacity)[i] == 0
    static_risks = static_terminal_certificate()
    robust_gap = gap-180*F(1, 10000)
    assert robust_gap == F("0.038509475")
    report = {"status": "PASS: implementation-diverse anchor and control audit",
              "independent_human_or_cross_model_review": False,
              "formal_proof": False, "original_verifier_imported": False,
              "sampled_trajectories": 0, "new_horizons": [],
              "certificate_sha256": hashlib.sha256(raw).hexdigest(),
              "audit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "low_primitive_states": states, "lower": str(lower),
              "upper": str(upper), "gap": str(gap),
              "mixed_risks": list(map(str, mixed)),
              "robust_gap_floor": str(robust_gap),
              "known_model_terminal_checks": "T=12,24; capacities=3,4; all three labels",
              "static_terminal_risks": list(map(str, static_risks))}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
