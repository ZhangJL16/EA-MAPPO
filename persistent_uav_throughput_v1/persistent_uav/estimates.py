"""Deployable linear estimates fitted once to physical calibration, no neural nets."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import nnls


def features(start, goal):
    delta = np.asarray(goal) - np.asarray(start)
    return np.array([1., np.linalg.norm(delta[:2]), abs(delta[2])], dtype=float)


@dataclass(frozen=True)
class EstimateModel:
    time_coefficients: tuple[float, float, float]
    energy_coefficients: tuple[float, float, float]
    goal_radius: float = 5.

    def predict(self, start, goal):
        if np.linalg.norm(np.asarray(goal) - np.asarray(start)) <= self.goal_radius:
            return 0., 0.
        x = features(start, goal)
        return max(0., float(x @ self.time_coefficients)), max(0., float(x @ self.energy_coefficients))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def fit(cls, rows):
        successful = [r for r in rows if r['success']]
        if not successful:
            raise ValueError('no successful calibration jobs; scales and model undefined')
        x = np.stack([features(r['start'], r['goal']) for r in successful])
        time = np.asarray([r['duration'] for r in successful])
        energy = np.asarray([r['energy_used'] for r in successful])
        return cls(tuple(nnls(x, time)[0]), tuple(nnls(x, energy)[0]))

