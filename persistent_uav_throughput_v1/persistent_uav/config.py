from dataclasses import dataclass
import math

PHYSICS_DT = 0.05
MAP_SEED = 720260920
CALIBRATION_SEED = 820260920
VALIDATION_SEEDS = tuple(range(920260920, 920260930))
EVALUATION_SEEDS = tuple(range(1020260920, 1020260940))
THRESHOLDS = (0.1, 0.25, 0.5, 0.75)
METHODS = ('fifo', 'nearest', 'shortest_time', 'energy_greedy', 'threshold_sjf', 'reserve_sjf')


def ceil_grid(value, dt=PHYSICS_DT):
    return math.ceil(value / dt - 1e-9) * dt


@dataclass(frozen=True)
class Config:
    capacity: float
    recharge_rate: float
    arrival_rate: float
    cutoff: float
    queue_capacity: int = 5
    initial_tasks: int = 3
    option_step_limit: int = 4000
    delta: float = 0.05

    def __post_init__(self):
        for field in ('capacity', 'recharge_rate', 'arrival_rate', 'cutoff'):
            if not math.isfinite(getattr(self, field)) or getattr(self, field) <= 0:
                raise ValueError(f'{field} must be finite and positive')
        if not 0 <= self.initial_tasks <= self.queue_capacity or self.queue_capacity < 1:
            raise ValueError('invalid queue configuration')
        if not isinstance(self.option_step_limit, int) or self.option_step_limit < 1:
            raise ValueError('invalid option step limit')
        if not 0 <= self.delta < 1:
            raise ValueError('invalid risk budget')
        if abs(self.cutoff - ceil_grid(self.cutoff)) > 1e-7:
            raise ValueError('cutoff must be on the physical time grid')

