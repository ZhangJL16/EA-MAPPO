from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Experiment:
    name: str
    bit: int
    kappa: float
    duration: float = 1.
    prerequisites: tuple = ()

    def safe(self, version):
        return all(version.known.get(i) == b for i, b in self.prerequisites)

    def actually_safe(self, truth):
        return all(truth[i] == b for i, b in self.prerequisites)

    def observation_likelihood(self, bit_value, observation):
        p = .5 + (2 * bit_value - 1) * self.kappa
        return p if observation else 1 - p

    def cycle_duration(self, model=None):
        return self.duration

    def information(self):
        k = self.kappa
        return 2*k*math.log((.5+k)/(.5-k)) if k else 0.
