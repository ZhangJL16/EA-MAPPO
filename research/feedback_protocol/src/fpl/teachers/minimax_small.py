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


@dataclass(frozen=True)
class Tree:
    route: object = None
    branches: tuple = ()  # (feedback tuple, continuation Tree)


class ExactMinimax:
    def __init__(self,problem,**limits):
        self.problem,self.limits = problem,limits

    def solve(self):
        return solve(self.problem,**self.limits)


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


def solve(problem, max_trees=1000, max_combinations=100000, max_seconds=10):
    started = monotonic()
    n = len(problem.hypotheses)
    uniform = (F(1,n),)*n  # Enumerate feedback support, not a Bayes objective.
    stats = {"tree_combinations":0,"lp_candidates":0}

    def check():
        if monotonic()-started>max_seconds:
            raise PlanningLimit("minimax time limit; unresolved")

    @lru_cache(None)
    def trees(t):
        check()
        pool = {(F(0),)*n:Tree()}
        for route in enumerate_protocols(problem,t):
            future = trees(t-route.duration)
            branches = list(outcomes(problem,uniform,route.channels))
            probabilities = []
            for feedback,_,_ in branches:
                ps = []
                for row in problem.hypotheses:
                    mass = F(1)
                    for c,y in feedback:
                        p = row[problem.channels.index(c)]
                        mass *= p if y else 1-p
                    ps.append(mass)
                probabilities.append(ps)
            immediate = protocol_vector(problem,route)
            for choices in product(future,repeat=len(branches)):
                check()
                stats["tree_combinations"] += 1
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
    oracle = tuple(known_model_value(problem,i) for i in range(n))
    risks = tuple(tuple(oracle[i]-v[i] for i in range(n)) for v,_ in policies)
    # Randomized policies are mixtures of deterministic contingent policy trees.
    for k in range(1,min(n,len(policies))+1):
        for cols in combinations(range(len(policies)),k):
            for rows in combinations(range(n),k):
                check()
                stats["lp_candidates"] += 1
                if stats["lp_candidates"]>max_combinations:
                    raise PlanningLimit("minimax LP vertex limit; unresolved")
                a = [[risks[c][r] for c in cols]+[F(-1)] for r in rows]+[[F(1)]*k+[F(0)]]
                primal = solve_linear(a,[F(0)]*k+[F(1)])
                dual = solve_linear([[risks[c][r] for r in rows]+[F(-1)] for c in cols]
                                    +[[F(1)]*k+[F(0)]],[F(0)]*k+[F(1)])
                if primal is None or dual is None or min(primal[:-1])<0 or min(dual[:-1])<0:
                    continue
                v = primal[-1]
                if dual[-1] != v:
                    continue
                mixture_risks = tuple(sum(w*risks[c][i] for w,c in zip(primal[:-1],cols)) for i in range(n))
                prior = tuple(dual[rows.index(i)] if i in rows else F(0) for i in range(n))
                if max(mixture_risks)<=v and all(sum(w*r for w,r in zip(prior,col))>=v for col in risks):
                    return dict(status="exact_rational_primal_dual",value=v,risks=mixture_risks,
                                least_favorable_prior=prior,
                                mixture=tuple((w,policies[c][1]) for w,c in zip(primal[:-1],cols) if w),
                                deterministic_vectors=len(policies),stats=stats)
    raise RuntimeError("no exact saddle certificate found")
