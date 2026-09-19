"""Belief UCT with progressive-widened, return-safe prefix proposals.

This is a classical baseline, not an implementation-equivalence claim to POMCP.
Root samples a public hypothesis; feedback is available only after a whole
committed protocol. Rollout decisions use the posterior, never sampled truth.
All family nodes have a direct return; this guard restricts the general graph
policy class and is explicit rather than pretending to solve arbitrary routes.
"""
from collections import Counter
from fractions import Fraction as F
from math import log,sqrt
import random
from ..belief import update
from ..protocols import Protocol,PlanningLimit
from ..utility import protocol_value,protocol_vector
from ..budget import reset_fallback
from ..work import PlanningWorkBudget


class BeliefMCTS:
    def __init__(self,problem,seed=0,exploration=1.0,max_simulations=100000):
        self.problem,self.rng = problem,random.Random(seed)
        self.exploration,self.max_simulations = exploration,max_simulations

    def select(self,state,budget=None):
        p = self.problem
        if state.resource!=p.capacity:
            raise ValueError("MCTS selects at reset only")
        budget = budget if budget is not None else PlanningWorkBudget()
        root = dict(n=0,arms={})
        best = None
        self.stats = dict(simulations=0,truncated=False)

        def propose(t):
            node,battery,names,channels,duration = p.reset,p.capacity,(),(),0
            while True:
                legal = []
                for op in p.operations:
                    if op.source!=node:
                        continue
                    budget.consume(expansions=1)
                    cs = channels+op.channels
                    if op.energy>battery or duration+op.duration>t or len(cs)>p.max_measurements or any(v>p.per_channel_limit for v in Counter(cs).values()):
                        continue
                    returns = op.target==p.reset
                    if not returns:
                        for back in p.operations:
                            if back.source!=op.target or back.target!=p.reset:
                                continue
                            budget.consume(expansions=1)
                            # Family return operations have no observations.
                            if not back.channels and back.energy<=battery-op.energy and duration+op.duration+back.duration<=t:
                                returns = True
                                break
                    if returns:
                        legal.append(op)
                if not legal:
                    return None
                op = self.rng.choice(legal)
                names += (op.name,)
                channels += op.channels
                duration += op.duration
                battery -= op.energy
                node = op.target
                if node==p.reset:
                    return Protocol(names,duration,channels)

        def sample(route,b,theta):
            bits = []
            for c in route.channels:
                budget.consume(model_calls=1)
                mu = p.hypotheses[theta][p.channels.index(c)]
                bits.append((c,int(self.rng.random()<mu)))
            budget.consume(model_calls=len(b)*max(1,len(bits)))
            post = update(p,b,bits)
            # Conditional observed utility and unobserved task utility; no
            # sampled truth is ever supplied to a decision rule.
            from ..utility import operation_score
            cursor,reward = 0,F(0)
            for name in route.operations:
                op = next(o for o in p.operations if o.name==name)
                budget.consume(model_calls=1)
                ys = [y for _,y in bits[cursor:cursor+len(op.channels)]]
                cursor += len(op.channels)
                reward += operation_score(op,theta,ys)
            return tuple(bits),post,float(reward)

        def rollout(t,b,theta):
            options = []
            for o in p.operations:
                if o.source!=p.reset or o.target!=p.reset:
                    continue
                budget.consume(expansions=1)
                if o.channels or o.energy>p.capacity or o.duration>t:
                    continue
                r = Protocol((o.name,),o.duration,())
                options.append((protocol_value(p,b,r,budget)/r.duration,r))
            if not options:
                return 0.
            _,route = max(options,key=lambda x:x[0])
            return float((t//route.duration)*protocol_vector(p,route,budget)[theta])

        def simulate(node,t,b,theta):
            if t==0:
                return 0.
            budget.consume(expansions=1)
            if len(node["arms"])<max(1,int(sqrt(node["n"]+1))):
                r = propose(t)
                if r is not None:
                    node["arms"].setdefault(r.operations,dict(route=r,n=0,total=0.,children={}))
            if not node["arms"]:
                return 0.
            arms = list(node["arms"].values())
            fresh = next((a for a in arms if not a["n"]),None)
            arm = fresh if fresh is not None else max(arms,key=lambda a:a["total"]/a["n"]+
                        self.exploration*p.budget*sqrt(log(node["n"]+1)/a["n"]))
            route = arm["route"]
            feedback,post,reward = sample(route,b,theta)
            remaining = t-route.duration
            if feedback not in arm["children"]:
                arm["children"][feedback] = dict(n=0,arms={})
                reward += rollout(remaining,post,theta)
            else:
                reward += simulate(arm["children"][feedback],remaining,post,theta)
            node["n"] += 1
            arm["n"] += 1
            arm["total"] += reward
            return reward

        try:
            best = reset_fallback(p,state.remaining,budget)
            for _ in range(self.max_simulations):
                budget.consume(model_calls=len(state.posterior))
                theta = self.rng.choices(range(len(state.posterior)),weights=list(map(float,state.posterior)))[0]
                simulate(root,state.remaining,state.posterior,theta)
                self.stats["simulations"] += 1
                eligible = [a for a in root["arms"].values() if a["n"]]
                if eligible:
                    best = max(eligible,key=lambda a:(a["total"]/a["n"],a["n"]))["route"]
        except PlanningLimit as exc:
            self.stats.update(truncated=True,limit_reason=str(exc))
        self.stats.update(budget.snapshot())
        return best
