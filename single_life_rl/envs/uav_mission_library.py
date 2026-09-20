"""Finite semi-empirical model. Full observed cycle clock is not suppressed."""
import math
import numpy as np
from single_life_rl.core.refinement import safe_refinement_action


class MissionModels:
    def __init__(self, library, thetas):
        self.arms = [r for r in library['missions'] if r['success']]
        self.thetas = thetas
        self.B = library['battery']
        self.r = library['recharge_rate']

    def safe_experiments(self, version, static=False):
        return [a for a in self.arms if all(self.thetas[m]*a['energy'] < self.B for m in version)]

    def singleton(self, version):
        return len(version) == 1

    def informative_experiments(self, version, safe, decision_only=True):
        return safe if len(version) > 1 else []

    def information_design(self, version, informative):
        if not informative:
            return None, 0.
        arm = min(informative, key=lambda a: (max(self.cycle_duration(m,a) for m in version), a['mission_id']))
        return arm, 'singular_exact_clock'

    def cycle_duration(self, model, arm):
        return arm['duration'] + self.thetas[model]*arm['energy']/self.r

    def optimal(self, model):
        safe = self.safe_experiments([model])
        if not safe:
            return None
        return min(safe, key=lambda a: (self.cycle_duration(model,a), a['mission_id']))

    def common_optimal_policy(self, version):
        best = [self.optimal(m) for m in version]
        if not best or any(a is None for a in best):
            return None
        return best[0] if len({a['mission_id'] for a in best}) == 1 else None

    def observation_likelihood(self, model, arm, observed_clock, energy_sensor):
        # Exact discrete support in the known deterministic clock channel.
        if observed_clock != self.cycle_duration(model, arm):
            return 0.
        sigma = .03*arm['energy']
        z = (energy_sensor-self.thetas[model]*arm['energy'])/sigma
        return math.exp(-z*z/2)/(math.sqrt(2*math.pi)*sigma)


def run_uav(job, protocol, library):
    models = MissionModels(library, protocol['uav']['theta'])
    truth = job['truth_index']; method = job['method']; T = job['horizon']
    rng = np.random.default_rng(job['seed'])
    version = list(range(len(models.thetas)))
    safe_initial = len(models.safe_experiments(version))
    oracle = models.optimal(truth)
    rate = 0. if oracle is None else 1/models.cycle_duration(truth, oracle)
    t = reward = explore = 0.
    cycles = 0; cert = None; catastrophic = False; eliminated = False
    events = []; status = 'horizon_censored'
    initial_common = models.common_optimal_policy(version)
    if initial_common is not None:
        cert = 0.
    while t < T:
        common = models.common_optimal_policy(version)
        learning = False
        if method == 'Oracle':
            arm = oracle; cert = 0.
        elif method == 'WorstCaseRobust':
            arm = models.optimal(len(models.thetas)-1)
        elif method == 'CertaintyEquivalent':
            # Equal prior / lexicographically smallest MLE tie before first sample.
            arm = models.optimal(version[0])
        else:
            kind, arm, _, _ = safe_refinement_action(models, version)
            learning = kind == 'experiment'
            if kind == 'deploy' and cert is None: cert = t
        if arm is None:
            status = 'no_available_arm'; break
        duration = models.cycle_duration(truth, arm)
        if models.thetas[truth]*arm['energy'] >= models.B:
            # Library has no within-mission energy profile. This semi-Markov
            # abstraction places catastrophic outcome at scheduled cycle end.
            if t+duration > T:
                if learning: explore += T-t
                t=T; break
            t += duration; cycles += 1; catastrophic = True; status='catastrophe'; break
        if t+duration > T:
            if learning: explore += T-t
            t=T; break
        if not learning and method in ('Oracle','WorstCaseRobust','SafeRefine'):
            n = int((T-t)/duration)
            reward += n; cycles += n; t=T; status='exploitation'; break
        t += duration; reward += 1; cycles += 1
        if learning: explore += duration
        y = models.thetas[truth]*arm['energy'] + rng.normal(0, .03*arm['energy'])
        # Exact support filtering; do not let Gaussian underflow exclude models.
        version = [m for m in version if models.cycle_duration(m,arm) == duration]
        if not version: raise RuntimeError('empty UAV support set')
        eliminated |= truth not in version
        events.append(dict(time=t, arm=arm['mission_id'], clock=duration, energy_sensor=y, version=version.copy()))
        if method=='SafeRefine' and models.common_optimal_policy(version) is not None:
            cert=t
    return dict(job_id=job['job_id'], suite='uav', method=method, spec=job, catastrophe=catastrophic,
                reward=reward, oracle_rate=rate, regret=T*rate-reward, certification_time=cert if method in ('SafeRefine','Oracle') else None,
                certification_censored=cert is None or method not in ('SafeRefine','Oracle'),
                exploration_time=explore, cycles=cycles, true_model_eliminated=eliminated,
                model_set_cardinality=len(version), unlocked_experiments=len(models.safe_experiments(version))-safe_initial,
                observations=events, status=status,
                limitation='Deterministic full cycle clock reveals theta; no finite Gaussian-only information complexity claim.')
