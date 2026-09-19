from fractions import Fraction as F
from time import monotonic
from ..belief import expected_reward, outcomes
from ..protocols import enumerate_protocols, PlanningLimit


class ExactBayes:
    """Rational Bellman recursion for committed protocols and reset terminals.

    Limits fail explicitly; no timeout result is labeled optimal.
    """
    def __init__(self, problem, max_states=20000, max_seconds=15,
                 max_protocol_nodes=10000, max_outcomes=4096):
        self.problem = problem
        self.max_states, self.max_seconds = max_states, max_seconds
        self.max_protocol_nodes, self.max_outcomes = max_protocol_nodes, max_outcomes
        self.reset_cache()

    def reset_cache(self):
        self.cache, self.route_cache, self.actions = {}, {}, {}
        self.states, self.likelihood_branches = 0, 0
        self.started = monotonic()

    def value(self, remaining, prior):
        if type(remaining) is not int or not 0 <= remaining <= self.problem.budget:
            raise ValueError("invalid remaining budget")
        if len(prior) != len(self.problem.hypotheses) or min(prior) < 0 or sum(prior) != 1:
            raise ValueError("invalid belief")
        key = remaining, tuple(prior)
        if key in self.cache:
            return self.cache[key]
        if monotonic()-self.started > self.max_seconds or self.states >= self.max_states:
            raise PlanningLimit("Bayes solver state/time limit; optimum unresolved")
        self.states += 1
        if remaining not in self.route_cache:
            self.route_cache[remaining] = enumerate_protocols(self.problem, remaining, self.max_protocol_nodes)
        best, action = F(0), None  # Stop at reset; unused time earns zero.
        for route in self.route_cache[remaining]:
            candidate = expected_reward(self.problem, prior, route.channels)
            for _, mass, posterior in outcomes(self.problem, prior, route.channels, self.max_outcomes):
                self.likelihood_branches += 1
                candidate += mass*self.value(remaining-route.duration, posterior)
            if candidate > best:
                best, action = candidate, route
        self.cache[key], self.actions[key] = best, action
        return best

    def select(self, state):
        if state.resource != self.problem.capacity:
            raise ValueError("planner may act only after reset")
        self.reset_cache()
        self.value(state.remaining, state.posterior)
        return self.actions[(state.remaining, tuple(state.posterior))]


def known_model_value(problem, hypothesis_index, remaining=None):
    """Evaluator-only deterministic finite-budget DP, with the same terminals."""
    if not 0 <= hypothesis_index < len(problem.hypotheses):
        raise ValueError("invalid hypothesis")
    horizon = problem.budget if remaining is None else remaining
    routes = enumerate_protocols(problem, horizon)
    prior = tuple(F(int(i == hypothesis_index)) for i in range(len(problem.hypotheses)))
    values = [F(0)]*(horizon+1)
    for t in range(1, horizon+1):
        values[t] = max([F(0)]+[expected_reward(problem, prior, route.channels)+values[t-route.duration]
                                for route in routes if route.duration <= t])
    return values[horizon]
