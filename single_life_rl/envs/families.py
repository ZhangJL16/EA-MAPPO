from single_life_rl.core.experiment import Experiment


def experiments(spec):
    family = spec['suite']
    if family == 'binary':
        return [Experiment('p0', 0, spec['kappa'])], 1, 1
    if family in ('recursive', 'nuisance'):
        k1, k2 = spec.get('kappas', [spec.get('kappa', .08)]*2)
        result = [Experiment('p0', 0, k1)]
        result += [Experiment(f'p1_{b}', 1, k2, prerequisites=((0,b),)) for b in (0,1)]
        for i in range(spec.get('m', 0)):
            result.append(Experiment(f'u{i}', i+2, spec['kappa']))
        return result, 2, 2+spec.get('m', 0)
    if family == 'random':
        result = []
        for node in spec['tree']['nodes']:
            prefix = tuple(node['prefix'])
            result.append(Experiment(node['name'], len(prefix), node['kappa'], node['duration'], tuple(enumerate(prefix))))
        return result, 3, 3
    raise ValueError(family)


class FactoredModels:
    """Model/Experiment API shared with the mission library."""
    def __init__(self, spec, counts):
        self.experiments, self.decision_dims, self.dimensions = experiments(spec)
        self.counts = counts

    def common_optimal_policy(self, version):
        return version.common_optimal_policy(self.decision_dims)

    def singleton(self, version):
        return len(version.known) == self.dimensions

    def safe_experiments(self, version, static=False):
        from single_life_rl.core.refinement import safe_experiments
        return safe_experiments(self.experiments, version, static)

    def informative_experiments(self, version, safe, decision_only=True):
        target = range(self.decision_dims if decision_only else self.dimensions)
        return [e for e in safe if e.bit in target and e.bit not in version.known and e.information() > 0]

    def information_design(self, version, informative):
        from single_life_rl.core.information import choose
        return choose(informative, self.counts)

    def observation_likelihood(self, model, experiment, observation):
        return experiment.observation_likelihood(model[experiment.bit], observation)

    def cycle_duration(self, model, experiment):
        return experiment.duration
