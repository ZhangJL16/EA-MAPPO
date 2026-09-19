"""Exact rational finite policy-tree game; deliberately tiny-instance only."""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import lru_cache
from itertools import product, combinations
from time import monotonic
from ..protocols import enumerate_protocols, PlanningLimit
from ..belief import outcomes
from ..utility import protocol_vector
from .exact_bayes import known_model_value
from ..budget import SearchBudget


@dataclass(frozen=True)
class Tree:
    route: object = None
    branches: tuple = ()  # (feedback tuple, continuation Tree)


class ExactMinimax:
    def __init__(self,problem,**limits):
        self.problem,self.limits = problem,limits

    def solve(self):
        return solve(self.problem,**self.limits)


class UnresolvedCertificate(PlanningLimit):
    """No reported optimum: a computational cap or certificate failure."""


CertifiedMinimaxSmall = ExactMinimax


def solve_linear(matrix, rhs):
    a = [list(map(F,row))+[F(b)] for row,b in zip(matrix,rhs)]
    n = len(a)
    for i in range(n):
        pivot = next((j for j in range(i,n) if a[j][i]),None)
        if pivot is None:
            return None
        a[i],a[pivot] = a[pivot],a[i]
        scale = a[i][i]
        a[i] = [x/scale for x in a[i]]
        for j in range(n):
            if j != i:
                scale = a[j][i]
                a[j] = [x-scale*y for x,y in zip(a[j],a[i])]
    return tuple(row[-1] for row in a)


def certify_matrix(risks, max_candidates=100000, max_seconds=10, check=None):
    """Rational minimax LP vertices, with INDEPENDENT primal/dual supports.

    Columns are policies; coordinates are hypotheses. On a support of size s,
    a basic vertex has s independent active risk inequalities (normalization
    supplies the last equation). Enumerating supports and active sets covers
    degenerate vertices too. Dual vertices are enumerated independently: their
    support need not equal the primal support. Only matching feasible rational
    witnesses are returned. Limits never certify nonexistence.
    """
    risks = tuple(tuple(map(F,c)) for c in risks)
    if not risks or not risks[0] or any(len(c)!=len(risks[0]) for c in risks):
        raise ValueError("nonempty rectangular risk matrix required")
    n,m = len(risks[0]),len(risks)
    started,candidates = monotonic(),0
    bestp,bestd = None,None
    for k in range(1,min(n,m)+1):
        for cols in combinations(range(m),k):
            for rows in combinations(range(n),k):
                if check:
                    check()
                candidates += 1
                if candidates>max_candidates or monotonic()-started>max_seconds:
                    raise UnresolvedCertificate("rational basis search limit")
                p = solve_linear([[risks[c][r] for c in cols]+[F(-1)] for r in rows]
                                 +[[F(1)]*k+[F(0)]],[F(0)]*k+[F(1)])
                if p is not None and min(p[:-1])>=0:
                    mixed = tuple(sum(w*risks[c][r] for w,c in zip(p[:-1],cols)) for r in range(n))
                    if max(mixed)<=p[-1] and (bestp is None or p[-1]<bestp[0]):
                        bestp = (p[-1],tuple((c,w) for c,w in zip(cols,p[:-1]) if w),mixed)
                d = solve_linear([[risks[c][r] for r in rows]+[F(-1)] for c in cols]
                                 +[[F(1)]*k+[F(0)]],[F(0)]*k+[F(1)])
                if d is not None and min(d[:-1])>=0:
                    prior = tuple(d[rows.index(r)] if r in rows else F(0) for r in range(n))
                    if (all(sum(w*x for w,x in zip(prior,c))>=d[-1] for c in risks)
                        and (bestd is None or d[-1]>bestd[0])):
                        bestd = (d[-1],prior)
                if bestp is not None and bestd is not None and bestp[0]==bestd[0]:
                    return dict(value=bestp[0],weights=bestp[1],risks=bestp[2],
                                least_favorable_prior=bestd[1],lp_candidates=candidates)
    raise UnresolvedCertificate("basis enumeration exhausted without matching witnesses")


def solve(problem, max_trees=1000, max_combinations=100000, max_seconds=10,budget=None):
    started = monotonic()
    budget = budget if budget is not None else SearchBudget(max_expansions=10000000,max_model_calls=10000000,max_seconds=max_seconds)
    n = len(problem.hypotheses)
    uniform = (F(1,n),)*n  # Enumerate feedback support, not a Bayes objective.
    stats = {"tree_combinations":0,"lp_candidates":0}

    def check():
        budget.consume()
        if monotonic()-started>max_seconds:
            raise PlanningLimit("minimax time limit; unresolved")

    @lru_cache(None)
    def trees(t):
        check()
        pool = {(F(0),)*n:Tree()}
        for route in enumerate_protocols(problem,t,budget=budget):
            future = trees(t-route.duration)
            branches = list(outcomes(problem,uniform,route.channels,budget=budget))
            probabilities = []
            for feedback,_,_ in branches:
                ps = []
                for row in problem.hypotheses:
                    budget.consume(model_calls=max(1,len(feedback)))
                    mass = F(1)
                    for c,y in feedback:
                        p = row[problem.channels.index(c)]
                        mass *= p if y else 1-p
                    ps.append(mass)
                probabilities.append(ps)
            immediate = protocol_vector(problem,route,budget)
            for choices in product(future,repeat=len(branches)):
                check()
                stats["tree_combinations"] += 1
                budget.consume(expansions=1)
                if stats["tree_combinations"]>max_combinations:
                    raise PlanningLimit("minimax policy product limit; unresolved")
                vector = tuple(immediate[i]+sum(probabilities[j][i]*choices[j][0][i]
                                                for j in range(len(branches))) for i in range(n))
                pool.setdefault(vector,Tree(route,tuple((branches[j][0],choices[j][1])
                                                       for j in range(len(branches)))))
                if len(pool)>max_trees:
                    raise PlanningLimit("minimax unique-vector limit; unresolved")
        return tuple(pool.items())

    policies = trees(problem.budget)
    oracle = tuple(known_model_value(problem,i,budget=budget) for i in range(n))
    risks = tuple(tuple(oracle[i]-v[i] for i in range(n)) for v,_ in policies)
    cert = certify_matrix(risks,max_candidates=max_combinations,max_seconds=max_seconds,check=check)
    stats["lp_candidates"] = cert["lp_candidates"]
    stats.update(budget.snapshot())
    return dict(status="exact_rational_primal_dual",value=cert["value"],risks=cert["risks"],
                least_favorable_prior=cert["least_favorable_prior"],
                mixture=tuple((w,policies[c][1]) for c,w in cert["weights"]),
                deterministic_vectors=len(policies),stats=stats)
