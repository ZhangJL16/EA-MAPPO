"""Isolated, fixed-fixture route calibration; no legacy environment imports."""
from dataclasses import dataclass
import math
from scipy.optimize import linprog

HYPOTHESES = ((.05, .05), (.30, .05), (.30, .95))
CELLS = ((3, True), (4, True), (3, False), (4, False))
POSES = {"q": (0., 0.), "A": (1., 0.), "B": (.5, math.sqrt(3)/2)}
METHODS = ("resource_path", "cost_aware_reference", "cost_blind", "independent_ucb", "oracle_allocation")


def kl(p, q):
    return p*math.log(p/q)+(1-p)*math.log((1-p)/(1-q))


@dataclass(frozen=True)
class Route:
    name: str
    actions: tuple
    channels: tuple

    @property
    def length(self):
        return len(self.actions)

    def reward(self, theta):
        return .05 if self.name == "dock" else sum(theta[0 if c == "A" else 1] for c in self.channels)

    def information(self, theta, alternative):
        return sum(kl(theta[0 if c == "A" else 1], alternative[0 if c == "A" else 1]) for c in self.channels)


def catalogue(capacity, bundling):
    if capacity not in (3, 4) or not isinstance(bundling, bool):
        raise ValueError("calibration fixture only")
    routes = [Route("dock", ("dock",), ()), Route("empty", ("undock", "return"), ()),
              Route("A", ("undock", "A", "return"), ("A",)),
              Route("B", ("undock", "B", "return"), ("B",))]
    if capacity == 4 and bundling:
        routes.append(Route("AB", ("undock", "A", "B", "return"), ("A", "B")))
    return tuple(routes)  # BA is a fixed, publicly declared equivalent path type.


def solve_models(routes, rate_objective=False):
    models = []
    for theta in HYPOTHESES:
        rates = [p.reward(theta)/p.length for p in routes]
        opt = max(range(len(routes)), key=lambda i: rates[i])
        if sum(abs(v-rates[opt]) < 1e-12 for v in rates) != 1:
            raise ValueError("nonunique optimum")
        costs = [rates[opt]*p.length-p.reward(theta) for p in routes]
        confusing, free = [], []
        for alt in HYPOTHESES:
            alt_opt = max(range(len(routes)), key=lambda i: routes[i].reward(alt)/routes[i].length)
            if alt_opt != opt:
                info = routes[opt].information(theta, alt)
                (confusing if info < 1e-12 else free).append(alt)
        allocation = [0.] * len(routes)
        candidates = [i for i in range(len(routes)) if i != opt]
        if confusing:
            answer = linprog([costs[i]/(routes[i].length if rate_objective else 1) for i in candidates],
                             A_ub=[[-routes[i].information(theta, alt) for i in candidates] for alt in confusing],
                             b_ub=[-1.] * len(confusing), bounds=(0, None), method="highs")
            if not answer.success:
                raise ValueError(answer.message)
            for i, value in zip(candidates, answer.x):
                allocation[i] = float(value)
        models.append({"theta": theta, "opt": opt, "gain": rates[opt], "costs": costs,
                       "confusing": confusing, "allocation": allocation,
                       "C": sum(a*c for a, c in zip(allocation, costs)),
                       "kappa": 2/min(routes[opt].information(theta, alt) for alt in free) if free else 0.})
    return models


class Plant:
    """Every action advances pose, integer clock and debit/reload simultaneously."""
    def __init__(self, capacity, bundling):
        catalogue(capacity, bundling)
        self.capacity, self.bundling = capacity, bundling
        self.state = ("q", (), capacity, False)
        self.time = 0

    @property
    def at_dock(self):
        return not self.state[3]

    def legal(self):
        site, mask, battery, departed = self.state
        if not departed:
            return ("dock", "undock")
        actions = ["return"] if battery >= 1 else []
        if battery >= 2 and (self.bundling or not mask):
            actions.extend(c for c in ("A", "B") if c not in mask and c != site)
        return tuple(actions)

    def step(self, action):
        if action not in self.legal():
            raise ValueError("unsafe or protocol-illegal action")
        site, mask, battery, departed = self.state
        battery -= 1  # Must be nonnegative BEFORE reload, not merely after.
        assert battery >= 0
        channel = None
        if action in ("return", "dock"):
            site, mask, battery, departed = "q", (), self.capacity, False
            channel = "dock" if action == "dock" else None
        elif action == "undock":
            departed = True  # q coordinate is NOT a reload after undocking.
        else:
            site, mask, channel = action, tuple(sorted(mask+(action,))), action
        self.state = site, mask, battery, departed
        self.time += 1
        return channel


def bias_audit(capacity, bundling, gain, theta):
    """Independent reachable primitive-state AROE audit, not catalogue constants."""
    root = ("q", (), capacity, False)
    edges, pending = {}, [root]
    while pending:
        state = pending.pop()
        if state in edges:
            continue
        plant = Plant(capacity, bundling)
        plant.state = state
        outgoing = []
        for action in plant.legal():
            branch = Plant(capacity, bundling)
            branch.state = state
            channel = branch.step(action)
            reward = .05 if channel == "dock" else theta[0 if channel == "A" else 1] if channel else 0.
            outgoing.append((branch.state, reward, action))
            if branch.state not in edges:
                pending.append(branch.state)
        edges[state] = outgoing
    values = {root: 0.}
    active = set()
    def value(state):
        if state in values:
            return values[state]
        if state in active:
            raise AssertionError("nonreload support cycle")
        active.add(state)
        values[state] = max(reward-gain+value(nxt) for nxt, reward, _ in edges[state])
        active.remove(state)
        return values[state]
    for state in edges:
        value(state)
    residual = max(abs(values[state]-max(reward-gain+values[nxt] for nxt, reward, _ in out))
                   for state, out in edges.items())
    if residual > 1e-10:
        raise AssertionError("AROE audit failed")
    return {"reachable_states": len(edges), "span": max(values.values())-min(values.values()),
            "AROE_residual": residual}


def audits():
    output = []
    for capacity, bundling in CELLS:
        routes = catalogue(capacity, bundling)
        models = solve_models(routes)
        for model in models:
            output.append({"capacity": capacity, "bundling": bundling,
                           "theta": model["theta"], "optimal_path": routes[model["opt"]].name,
                           "gain": model["gain"], "confusing": model["confusing"], "C": model["C"],
                           "allocation": dict(zip((r.name for r in routes), model["allocation"])),
                           "acquisition_costs": dict(zip((r.name for r in routes), model["costs"])),
                           **bias_audit(capacity, bundling, model["gain"], model["theta"])})
    for index in range(len(HYPOTHESES)):
        group = output[index::len(HYPOTHESES)]
        assert len({(r["gain"], r["optimal_path"], tuple(r["confusing"])) for r in group}) == 1
    return output


class Learner:
    """Capability-limited API: public catalogue + observed feedback only; no IO.

    This is interface isolation, NOT an OS sandbox for adversarial code.
    """
    def __init__(self, routes, method):
        if method not in METHODS[:4]:
            raise ValueError("privileged reference cannot enter learner")
        self.routes, self.method = routes, method
        self.models = solve_models(routes, method == "cost_blind")
        self.Q = [0]*len(routes)
        self.likelihood = [0.]*len(HYPOTHESES)
        self.plays, self.totals = [0]*len(routes), [0.]*len(routes)

    def select(self, time):
        if 0 in self.Q:
            return self.Q.index(0), True, "initialization"
        if self.method == "independent_ucb":
            scores = [self.totals[i]/self.plays[i]/p.length+math.sqrt(2*math.log(time+3)/self.plays[i])
                      for i, p in enumerate(self.routes)]
            return max(range(len(scores)), key=scores.__getitem__), True, "rate_UCB"
        estimate = max(range(len(HYPOTHESES)), key=self.likelihood.__getitem__)
        threshold = math.log(time+3)+2*math.log(math.log(time+3))+math.log(len(HYPOTHESES)+1)
        plausible = [i for i, ll in enumerate(self.likelihood) if self.likelihood[estimate]-ll <= threshold]
        decisions = {self.models[i]["opt"] for i in plausible}
        if len(decisions) == 1:
            return next(iter(decisions)), False, "certificate"
        model, j = self.models[estimate], sum(self.Q)
        if self.method == "cost_aware_reference":
            # Independent scalar scan, rather than resource learner's list decisions.
            least = min(range(len(self.Q)), key=lambda k: (self.Q[k], k))
            if self.Q[least] < math.sqrt(j):
                return least, True, "forced"
            for k in range(len(self.routes)):
                target = (1+(j+1)**(-1/8))*model["allocation"][k]*threshold
                if k != model["opt"] and self.Q[k] < target:
                    return k, True, "allocation"
            if self.Q[model["opt"]] < model["kappa"]*threshold:
                return model["opt"], True, "free_information"
            return least, True, "fallback"
        least = min(range(len(self.Q)), key=self.Q.__getitem__)
        if min(self.Q) < math.sqrt(j):
            return least, True, "forced"
        deficient = [k for k in range(len(self.routes)) if k != model["opt"] and
                     self.Q[k] < (1+(j+1)**(-1/8))*model["allocation"][k]*threshold]
        if deficient:
            return deficient[0], True, "allocation"
        if self.Q[model["opt"]] < model["kappa"]*threshold:
            return model["opt"], True, "free_information"
        return least, True, "fallback"

    def observe(self, path_index, feedback, exploration):
        reward = sum(y for _, y in feedback)
        self.plays[path_index] += 1
        self.totals[path_index] += reward
        if exploration:
            self.Q[path_index] += 1
        for channel, y in feedback:
            if channel == "dock":
                continue
            for i, theta in enumerate(HYPOTHESES):
                p = theta[0 if channel == "A" else 1]
                self.likelihood[i] += math.log(p if y else 1-p)

