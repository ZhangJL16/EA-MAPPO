"""Deterministic finite-game calculation; no sampling, plant, or learner run.

Independent literal transcription of the public three-hypothesis route class.
Bayesian Bellman best responses and finite zero-sum column generation produce
lower/upper certificates. Floating output is exploratory, not a rigorous proof.
"""
from functools import lru_cache
from fractions import Fraction
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

THETA = ((.05, .05), (.30, .05), (.30, .95))
GAIN = (.05, .10, .95 / 3)
ROUTES = (("dock", 1, ()), ("A", 3, (0,)), ("B", 3, (1,)),
          ("AB", 4, (0, 1)))


def exact_response(T, prior, bundled, end_at_dock=False):
    """Second Bellman implementation: unnormalized rational likelihood masses.

    NO floating posterior, optimization tolerance, random rewards or simulation.
    The maximum is evaluated exactly at every finite information state.
    """
    theta = ((Fraction(1, 20), Fraction(1, 20)),
             (Fraction(3, 10), Fraction(1, 20)),
             (Fraction(3, 10), Fraction(19, 20)))
    gain = (Fraction(1, 20), Fraction(1, 10), Fraction(19, 60))
    routes = ROUTES if bundled else ROUTES[:3]
    prior = tuple(Fraction(p) for p in prior)
    assert sum(prior) == 1 and min(prior) >= 0
    policy = {}

    @lru_cache(None)
    def masses(na, sa, nb, sb):
        return tuple(prior[v]*a**sa*(1-a)**(na-sa)*b**sb*(1-b)**(nb-sb)
                     for v, (a, b) in enumerate(theta))

    def next_counts(counts, channels, bits):
        result = list(counts)
        for k, c in enumerate(channels):
            result[2*c] += 1
            result[2*c+1] += (bits >> k) & 1
        return tuple(result)

    @lru_cache(None)
    def weighted_value(t, na, sa, nb, sb):
        if not t:
            return Fraction(0)
        counts = (na, sa, nb, sb)
        weights = masses(*counts)
        scores = []
        for name, duration, channels in routes:
            if end_at_dock and duration > t:
                continue
            if name == "dock":
                score = sum(weights)/20+weighted_value(t-1, *counts)
            elif duration > t:
                observed = channels[:max(0, t-1)]
                score = sum(weights[v]*sum(theta[v][c] for c in observed)
                            for v in range(3))
            else:
                score = sum(weights[v]*sum(theta[v][c] for c in channels)
                            for v in range(3))
                score += sum(weighted_value(t-duration,
                                            *next_counts(counts, channels, bits))
                             for bits in range(1 << len(channels)))
            scores.append((score, name))
        score, name = max(scores, key=lambda x: x[0])
        policy[(t, *counts)] = name
        return score

    weighted_reward = weighted_value(T, 0, 0, 0, 0)
    route_map = {r[0]: r for r in routes}

    @lru_cache(None)
    def evaluation(t, na, sa, nb, sb):
        if not t:
            return (Fraction(0),)*3
        counts = (na, sa, nb, sb)
        name, duration, channels = route_map[policy[(t, *counts)]]
        if name == "dock":
            return tuple(Fraction(1, 20)+x for x in evaluation(t-1, *counts))
        if duration > t:
            observed = channels[:max(0, t-1)]
            return tuple(sum((theta[v][c] for c in observed), Fraction(0))
                         for v in range(3))
        rewards = [sum((theta[v][c] for c in channels), Fraction(0)) for v in range(3)]
        for bits in range(1 << len(channels)):
            future = evaluation(t-duration, *next_counts(counts, channels, bits))
            for v in range(3):
                prob = math.prod(theta[v][c] if (bits >> k) & 1 else 1-theta[v][c]
                                 for k, c in enumerate(channels))
                rewards[v] += prob*future[v]
        return tuple(rewards)

    rewards = evaluation(T, 0, 0, 0, 0)
    risks = tuple(T*g-r for g, r in zip(gain, rewards))
    lower = T*sum(p*g for p, g in zip(prior, gain))-weighted_reward
    assert sum(p*r for p, r in zip(prior, risks)) == lower
    return risks, lower, weighted_value.cache_info().currsize, policy


def normalized_rational(values, denominator=1000000):
    items = [Fraction(float(v)).limit_denominator(denominator) for v in values]
    total = sum(items)
    return tuple(x/total for x in items)


def primitive_low_bayes(T, prior):
    """Independent primitive-action Bellman check, INCLUDING feedback adaptivity.

    Low capacity permits no second inspection before return. There is no
    path commitment in this recursion. All supported outcomes are enumerated.
    """
    theta = ((Fraction(1, 20), Fraction(1, 20)),
             (Fraction(3, 10), Fraction(1, 20)),
             (Fraction(3, 10), Fraction(19, 20)))
    gains = (Fraction(1, 20), Fraction(1, 10), Fraction(19, 60))
    prior = tuple(Fraction(p) for p in prior)

    @lru_cache(None)
    def value(t, battery, departed, weights):
        if not t:
            return Fraction(0)
        if not departed:
            return max(sum(weights)/20+value(t-1, 3, False, weights),
                       value(t-1, 2, True, weights))
        scores = [value(t-1, 3, False, weights)] if battery>=1 else []
        if battery>=2:
            for c in (0, 1):
                score = sum(weights[v]*theta[v][c] for v in range(3))
                score += value(t-1, battery-1, True,
                               tuple(weights[v]*theta[v][c] for v in range(3)))
                score += value(t-1, battery-1, True,
                               tuple(weights[v]*(1-theta[v][c]) for v in range(3)))
                scores.append(score)
        assert scores
        return max(scores)

    return T*sum(p*g for p, g in zip(prior, gains))-value(T, 3, False, prior)


def fixed_point_relaxation(T, bundled):
    """User-specified F minimax envelope; exploratory floating LP evaluation."""
    routes = ROUTES if bundled else ROUTES[:3]
    L = 4 if bundled else 3
    opt = ("dock", "A", "B")
    costs = [[GAIN[v]*ell-(.05 if name=="dock" else sum(THETA[v][c] for c in ch))
              for name, ell, ch in routes] for v in range(3)]

    def KL(p, q):
        return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))

    def rhs(R):
        maximum = 0.
        for v in range(3):
            delta = min(costs[v][p]/route[1] for p, route in enumerate(routes)
                        if route[0] != opt[v])
            e = (R+L)/(delta*(T/2-L))
            A, requirements = [], []
            for alt in range(3):
                if alt==v:
                    continue
                optimal_p = next(p for p, r in enumerate(routes) if r[0]==opt[v])
                alternative_gap = costs[alt][optimal_p]/routes[optimal_p][1]
                q = 2*(R+L)/(alternative_gap*T)
                k = max(0., (1-e)*math.log(1/q)-math.log(2)) if e<1 and q<1 else 0.
                credit = L*max(KL(THETA[v][c], THETA[alt][c]) for c in (0, 1))
                requirements.append(max(0., k-credit))
                A.append([-sum(KL(THETA[v][c], THETA[alt][c]) for c in route[2])
                          for route in routes])
            answer = linprog(costs[v], A_ub=A+[[r[1] for r in routes]],
                             b_ub=[-x for x in requirements]+[T],
                             bounds=(0, None), method="highs")
            if not answer.success:
                return math.inf
            maximum = max(maximum, float(answer.fun)-L)
        return max(0., maximum)

    if T<=2*L:
        raise ValueError("the F lemma requires T>2L")
    if rhs(0.)<=0:
        return 0.
    low, high = 0., float(T)
    for _ in range(65):
        middle = (low+high)/2
        if middle >= rhs(middle):
            high = middle
        else:
            low = middle
    return high


def exact_certificate(T):
    if T == 12:
        low_prior = (Fraction(1, 2), Fraction(9, 20), Fraction(1, 20))
        high_priors = [(Fraction(1, 3),)*3,
                       (Fraction(568, 1000), Fraction(421, 1000), Fraction(11, 1000)),
                       (Fraction(567, 1000), Fraction(421, 1000), Fraction(12, 1000))]
        mixing = (Fraction(12, 25), Fraction(1, 10), Fraction(21, 50))
    else:
        # Float discovery proposes priors/mixtures ONLY; exact recurrence verifies
        # the resulting bound independently and may reject the candidate.
        low_game = solve_game(T, False)
        high_game = solve_game(T, True, True)
        low_prior = normalized_rational(low_game["prior"])
        high_priors = [normalized_rational(row["policy_prior"])
                       for row in high_game["mixture"]]
        mixing = normalized_rational([row["weight"] for row in high_game["mixture"]])
    low_risks, lower, low_states, _ = exact_response(T, low_prior, False)
    primitive_lower = primitive_low_bayes(T, low_prior)
    assert primitive_lower == lower, "primitive adaptive and route Bellman values differ"
    witnesses, high_states = [], []
    for prior in high_priors:
        risks, _, states, _ = exact_response(T, prior, True, True)
        witnesses.append(risks)
        high_states.append(states)
    combined = tuple(sum(w*r[v] for w, r in zip(mixing, witnesses)) for v in range(3))
    upper = max(combined)
    gap = lower-upper
    assert gap > 0, "proposed finite-budget strictness certificate did not verify"
    return {"T": T, "method": "exact rational Bellman and policy evaluation",
            "low_all_safe_terminal_states_allowed": True,
            "high_witness_finishes_at_charger": True,
            "low_prior": [str(p) for p in low_prior],
            "low_Bayes_optimal_risks": [str(r) for r in low_risks],
            "independent_primitive_adaptive_lower": str(primitive_lower),
            "lower": str(lower), "upper": str(upper), "gap": str(gap),
            "lower_decimal": float(lower), "upper_decimal": float(upper),
            "gap_decimal": float(gap), "low_Bellman_states": low_states,
            "gap_gt_one_twentieth": gap > Fraction(1, 20),
            "high_Bellman_states": high_states,
            "high_witnesses": [{"prior": [str(p) for p in prior],
                                "weight": str(weight), "risks": [str(r) for r in risks]}
                               for prior, weight, risks in zip(high_priors, mixing, witnesses)],
            "high_combined_risks": [str(r) for r in combined]}


def verify_saved_certificate(path):
    """Verify encoded rational witnesses WITHOUT any floating LP/discovery."""
    document = json.loads(Path(path).read_text())
    assert document["sampled_trajectories"] == 0
    results = []
    for row in document["results"]:
        T = row["T"]
        prior = tuple(Fraction(x) for x in row["low_prior"])
        risks, lower, _, _ = exact_response(T, prior, False)
        assert lower == Fraction(row["lower"])
        assert risks == tuple(Fraction(x) for x in row["low_Bayes_optimal_risks"])
        assert primitive_low_bayes(T, prior) == lower
        high_rows, weights = [], []
        for witness in row["high_witnesses"]:
            prior = tuple(Fraction(x) for x in witness["prior"])
            risks, _, _, _ = exact_response(T, prior, True, True)
            assert risks == tuple(Fraction(x) for x in witness["risks"])
            high_rows.append(risks)
            weights.append(Fraction(witness["weight"]))
        assert min(weights)>=0 and sum(weights)==1
        combined = tuple(sum(w*r[v] for w, r in zip(weights, high_rows)) for v in range(3))
        upper = max(combined)
        assert combined == tuple(Fraction(x) for x in row["high_combined_risks"])
        assert upper == Fraction(row["upper"])
        assert lower-upper == Fraction(row["gap"])
        assert lower-upper > Fraction(1, 20)
        results.append({"T": T, "exact_gap": str(lower-upper)})
    return {"method": "rational witnesses only; NO floating optimization",
            "checked_horizons": [r["T"] for r in results], "results": results,
            "sampled_trajectories": 0, "formal_proof": False}


def best_response(T, prior, bundled, end_at_dock=False):
    routes = ROUTES if bundled else ROUTES[:3]
    prior = tuple(float(x) for x in prior)
    policy = {}

    @lru_cache(None)
    def belief(na, sa, nb, sb):
        logs = [math.log(p) + sa*math.log(a) + (na-sa)*math.log1p(-a)
                + sb*math.log(b) + (nb-sb)*math.log1p(-b)
                if p else -math.inf for p, (a, b) in zip(prior, THETA)]
        peak = max(logs)
        w = [math.exp(v-peak) for v in logs]
        z = sum(w)
        return tuple(x/z for x in w)

    def outcomes(channels):
        return range(1 << len(channels))

    def next_counts(counts, channels, bits):
        na, sa, nb, sb = counts
        for k, channel in enumerate(channels):
            y = (bits >> k) & 1
            if channel == 0:
                na, sa = na+1, sa+y
            else:
                nb, sb = nb+1, sb+y
        return na, sa, nb, sb

    def probability(v, channels, bits):
        return math.prod(THETA[v][channel] if (bits >> k) & 1
                         else 1-THETA[v][channel]
                         for k, channel in enumerate(channels))

    @lru_cache(None)
    def value(t, na, sa, nb, sb):
        if not t:
            return 0.
        counts = (na, sa, nb, sb)
        posterior = belief(*counts)
        scores = []
        for name, duration, channels in routes:
            if end_at_dock and duration > t:
                continue
            if name == "dock":
                score = .05+value(t-1, *counts)
            elif duration > t:
                # Inspection A/B occurs on primitive step 2; second on step 3.
                observed = channels[:max(0, t-1)]
                score = sum(posterior[v]*sum(THETA[v][c] for c in observed)
                            for v in range(3))
            else:
                score = sum(posterior[v]*sum(THETA[v][c] for c in channels)
                            for v in range(3))
                for bits in outcomes(channels):
                    prob = sum(posterior[v]*probability(v, channels, bits)
                               for v in range(3))
                    score += prob*value(t-duration,
                                        *next_counts(counts, channels, bits))
            scores.append((score, name))
        score, name = max(scores, key=lambda x: x[0])
        policy[(t, *counts)] = name
        return score

    bayes_reward = value(T, 0, 0, 0, 0)
    route_map = {r[0]: r for r in routes}

    @lru_cache(None)
    def evaluate(t, na, sa, nb, sb):
        if not t:
            return (0.,)*3
        counts = (na, sa, nb, sb)
        name, duration, channels = route_map[policy[(t, *counts)]]
        if name == "dock":
            return tuple(.05+x for x in evaluate(t-1, *counts))
        if duration > t:
            observed = channels[:max(0, t-1)]
            return tuple(sum(THETA[v][c] for c in observed) for v in range(3))
        rewards = [sum(THETA[v][c] for c in channels) for v in range(3)]
        for bits in outcomes(channels):
            future = evaluate(t-duration, *next_counts(counts, channels, bits))
            for v in range(3):
                rewards[v] += probability(v, channels, bits)*future[v]
        return tuple(rewards)

    rewards = evaluate(T, 0, 0, 0, 0)
    risks = tuple(T*g-r for g, r in zip(GAIN, rewards))
    dual_value = sum(p*r for p, r in zip(prior, risks))
    assert abs(dual_value-(T*sum(p*g for p, g in zip(prior, GAIN))-bayes_reward)) < 1e-8
    return risks, dual_value, value.cache_info().currsize, policy


def solve_game(T, bundled, end_at_dock=False, tol=1e-9):
    risks, _, states, _ = best_response(T, (1/3,)*3, bundled, end_at_dock)
    rows = [risks]
    row_priors = [(1/3,)*3]
    for iteration in range(200):
        # Adversary on a restricted catalogue of full decision policies.
        dual = linprog([0., 0., 0., -1.],
                       A_ub=[[-r[0], -r[1], -r[2], 1.] for r in rows],
                       b_ub=np.zeros(len(rows)), A_eq=[[1., 1., 1., 0.]],
                       b_eq=[1.], bounds=[(0., None)]*3+[(None, None)],
                       method="highs")
        if not dual.success:
            raise RuntimeError(dual.message)
        prior = tuple(dual.x[:3])
        risks, lower, states, _ = best_response(T, prior, bundled, end_at_dock)
        upper = float(dual.x[3])
        if upper-lower <= tol:
            mixture = linprog([0.]*len(rows)+[1.],
                              A_ub=[[*[r[v] for r in rows], -1.] for v in range(3)],
                              b_ub=[0.]*3,
                              A_eq=[[1.]*len(rows)+[0.]], b_eq=[1.],
                              bounds=[(0., None)]*len(rows)+[(None, None)],
                              method="highs")
            assert mixture.success
            return {"T": T, "bundled": bundled, "end_at_dock": end_at_dock,
                    "lower": lower, "upper": float(mixture.x[-1]),
                    "prior": prior, "iterations": iteration+1,
                    "Bellman_states": states,
                    "mixture": [{"weight": float(w), "risks": r,
                                 "policy_prior": row_prior}
                                for w, r, row_prior in zip(mixture.x[:-1], rows, row_priors)
                                if w>1e-9]}
        rows.append(risks)
        row_priors.append(prior)
    raise RuntimeError("column generation did not close")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--exact-certificate", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixed-point", action="store_true")
    parser.add_argument("--verify-receipt", type=Path)
    args = parser.parse_args()
    if args.verify_receipt:
        if args.horizons or args.output or args.fixed_point or args.exact_certificate:
            parser.error("receipt verification is a separate exact-only command")
        print(json.dumps(verify_saved_certificate(args.verify_receipt)), flush=True)
        return
    if not args.horizons:
        parser.error("specify --horizons or --verify-receipt")
    if args.fixed_point and args.exact_certificate:
        parser.error("choose fixed-point relaxation or exact certificate, not both")
    if args.output and not args.exact_certificate:
        parser.error("only exact certificates may be exported")
    if not args.fixed_point and any(T<0 or T>36 for T in args.horizons):
        parser.error("finite-game verification is restricted to 0<=T<=36")
    if args.output and args.output.exists():
        raise FileExistsError("do not overwrite a previous mathematical certificate")
    results = []
    for T in args.horizons:
        if args.fixed_point:
            print(json.dumps({"T": T, "fixed_point_low": fixed_point_relaxation(T, False),
                              "fixed_point_high": fixed_point_relaxation(T, True),
                              "status": "floating relaxation only"}), flush=True)
            continue
        if args.exact_certificate:
            result = exact_certificate(T)
            results.append(result)
            print(json.dumps(result), flush=True)
            continue
        low = solve_game(T, False)
        high = solve_game(T, True, end_at_dock=True)
        print(json.dumps({"low_unrestricted_terminal": low,
                          "high_dock_terminal_witness": high,
                          "certified_candidate_gap": low["lower"]-high["upper"]}), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"status": "EXACT RATIONAL CHECKS; NOT FORMAL PROOF",
                                           "sampled_trajectories": 0,
                                           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                           "public_hypothesis_order": ["theta_q", "theta_A", "theta_B"],
                                           "discovery": "floating LP proposes witnesses; exact arithmetic alone certifies gaps",
                                           "results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
