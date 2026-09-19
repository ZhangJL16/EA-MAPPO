from fractions import Fraction as F


def operation_score(op, hypothesis_index, bits):
    if op.utility is None:
        return sum(bits, F(0))
    spec = op.utility
    base = spec.by_hypothesis[hypothesis_index] if spec.by_hypothesis else F(0)
    return base + sum((w*y for w,y in zip(spec.observation_weights, bits)), F(0))


def protocol_vector(problem, route):
    """Expected task utility under each public model, before observing outcomes."""
    ops = {op.name: op for op in problem.operations}
    return tuple(sum((operation_score(ops[name], i,
                    [row[problem.channels.index(c)] for c in ops[name].channels])
                    for name in route.operations), F(0))
                 for i,row in enumerate(problem.hypotheses))


def protocol_value(problem, prior, route):
    return sum((w*v for w,v in zip(prior, protocol_vector(problem,route))), F(0))
