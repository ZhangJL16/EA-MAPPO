"""Fresh, sensor-conditioned HOCBF supervision; no cached projection Jacobians.

The SAC action remains the nominal action of the shielded environment. The
auxiliary loss uses the SAME current stochastic action and its fresh projection
in physical acceleration coordinates. It is not a safety certificate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import numpy as np
import torch

from experiments.directional_navigation.standard_baselines import ContinuousRecovery
from review_bundle.safety.collision.filter import SafetyFilterConfig, UAVSafetyActionFilter
from review_bundle.safety.collision.geometry3d import Lidar3DConfig, Lidar3DModel
from review_bundle.safety.collision.hocbf import HOCBFConfig

VERSION = "recovery_sac_fresh_hocbf_correction_v1"
OBSERVATION_DIM = 2056


@dataclass(frozen=True)
class TeacherConfig:
    horizontal_v_max: float = 20.0
    vertical_v_max: float = 5.0
    horizontal_a_max: float = 5.0
    vertical_a_max: float = 3.0
    lidar_max_range: float = 100.0
    lidar_horizontal_fov: float = 360.0
    lidar_vertical_fov: float = 60.0
    safe_radius: float = 0.5
    physics_dt: float = 0.05
    hocbf_k1: float = 1.0
    hocbf_k2: float = 1.0
    hocbf_uncertainty_margin: float = 1.0
    # Preserve existing HOCBF. Robust sampled-data is a separate experiment.
    hocbf_sampled_data_robust: bool = False

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "hocbf_sampled_data_robust":
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"positive finite teacher parameter required: {name}")

    def make_filter(self) -> UAVSafetyActionFilter:
        return UAVSafetyActionFilter(
            SafetyFilterConfig(
                horizontal_acceleration_limit=self.horizontal_a_max,
                vertical_acceleration_limit=self.vertical_a_max,
                top_k=None, safety_dt=self.physics_dt,
                deadline_seconds=self.physics_dt,
                sampled_data_robust=self.hocbf_sampled_data_robust,
            ),
            HOCBFConfig(k1=self.hocbf_k1, k2=self.hocbf_k2,
                        uav_radius=self.safe_radius,
                        uncertainty_margin=self.hocbf_uncertainty_margin),
        )


def physical_actions(actions: torch.Tensor, config: TeacherConfig) -> torch.Tensor:
    """Differentiable copy of the environment's actuator coordinate map."""
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("actions must be [batch,3]")
    clipped = actions.clamp(-1.0, 1.0)
    horizontal = clipped[:, :2]
    horizontal = horizontal / torch.linalg.vector_norm(
        horizontal, dim=1, keepdim=True
    ).clamp_min(1.0)
    return torch.cat((horizontal * config.horizontal_a_max,
                      clipped[:, 2:] * config.vertical_a_max), dim=1)


def correction_loss(physical: torch.Tensor, targets: torch.Tensor,
                    valid: torch.Tensor) -> torch.Tensor:
    """Mean over all queried samples; invalid teachers contribute no gradient.

No renormalization by valid/intervention count: declining validity must not
silently amplify the remaining teachers. Finite placeholders are mandatory.
"""
    if physical.shape != targets.shape or physical.ndim != 2 or physical.shape[1] != 3:
        raise ValueError("aligned [batch,3] actions and targets required")
    if valid.shape != (physical.shape[0],):
        raise ValueError("one validity flag per sample required")
    if not torch.isfinite(targets).all():
        raise ValueError("teacher targets must be finite, even if masked")
    per_sample = 0.5 * (physical - targets.detach()).square().sum(dim=1)
    return (per_sample * valid.to(per_sample.dtype)).mean()


class FreshHOCBFTeacher:
    def __init__(self, config: TeacherConfig):
        self.config = config
        self.filter = config.make_filter()
        self.directions = Lidar3DModel(Lidar3DConfig(
            horizontal_sectors=128, vertical_sectors=8,
            max_range=config.lidar_max_range,
            horizontal_fov_degrees=config.lidar_horizontal_fov,
            vertical_fov_degrees=config.lidar_vertical_fov,
        )).directions

    def project(self, observation: np.ndarray,
                acceleration: np.ndarray) -> tuple[np.ndarray, bool, str]:
        obs = np.asarray(observation, dtype=np.float64)
        nominal = np.asarray(acceleration, dtype=np.float64)
        if obs.shape != (OBSERVATION_DIM,) or nominal.shape != (3,):
            raise ValueError("teacher requires legacy2056 observation and physical3 action")
        if not np.isfinite(obs).all() or not np.isfinite(nominal).all():
            raise ValueError("nonfinite teacher input")
        ranges, flags = obs[7:1031], obs[1031:2055]
        if np.any((ranges < 0) | (ranges > 1)) or np.any((flags != 0) & (flags != 1)):
            raise ValueError("invalid sensor encoding")
        config = self.config
        if (np.linalg.norm(nominal[:2]) > config.horizontal_a_max + 1e-5
                or abs(nominal[2]) > config.vertical_a_max + 1e-5):
            raise ValueError("teacher action exceeds actuator bounds")
        hit = flags.astype(bool)
        if not np.any(hit):
            return nominal.copy(), True, "no_hits"
        # Translational invariance: all points and velocities use the sensor
        # frame; no simulator center, radius, position, or hidden map is needed.
        points = self.directions[hit] * (ranges[hit, None] * config.lidar_max_range)
        velocity = obs[:3] * [config.horizontal_v_max, config.horizontal_v_max,
                              config.vertical_v_max]
        output = self.filter.filter_lidar_points(np.zeros(3), velocity, nominal, points)
        diag, problem = output.diagnostics, output.projection_problem
        if (not diag.feasible or diag.fallback_used or problem is None
                or not problem.converged):
            return nominal.copy(), False, "solver_invalid"
        if ((diag.minimum_h is not None and diag.minimum_h < 0)
                or (diag.minimum_psi1 is not None and diag.minimum_psi1 < 0)):
            return nominal.copy(), False, "emergency_region"
        target = np.asarray(output.acceleration, dtype=np.float64)
        if (not np.isfinite(target).all()
                or np.any(problem.rows @ target < problem.lower_bounds - 1e-6)
                or np.linalg.norm(target[:2]) > config.horizontal_a_max + 1e-6
                or abs(target[2]) > config.vertical_a_max + 1e-6):
            return nominal.copy(), False, "constraint_invalid"
        # Offline labels have no execution deadline; wall-clock timing does not
        # change validity or RNG-resumability. Runtime latency is logged separately.
        return target.copy(), True, "projected"

    def batch(self, observations: torch.Tensor, physical: torch.Tensor
              ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
        started = perf_counter()
        obs = observations.detach().cpu().numpy()
        nominal = physical.detach().cpu().numpy()
        results = [self.project(o, a) for o, a in zip(obs, nominal, strict=True)]
        targets = np.stack([r[0] for r in results])
        valid = np.asarray([r[1] for r in results], dtype=bool)
        changed = valid & (np.linalg.norm(targets - nominal, axis=1) > 1e-6)
        metrics = {"queries": float(len(results)), "valid": float(valid.sum()),
                   "corrected": float(changed.sum()),
                   "solver_invalid": float(sum(r[2] == "solver_invalid" for r in results)),
                   "emergency_region": float(sum(r[2] == "emergency_region" for r in results)),
                   "constraint_invalid": float(sum(r[2] == "constraint_invalid" for r in results)),
                   "teacher_seconds": perf_counter() - started}
        return (torch.as_tensor(targets, dtype=physical.dtype, device=physical.device),
                torch.as_tensor(valid, device=physical.device), metrics)


class CorrectionRecovery(ContinuousRecovery):
    """Old2056 observations and locked recovery, with optional substep HOCBF."""
    def __init__(self, *, seed_start: int, seed_stride: int,
                 horizon: int = 4000, obstacles: int = 24, hocbf: bool = True):
        super().__init__(seed_start=seed_start, seed_stride=seed_stride,
                         horizon=horizon, obstacles=obstacles)
        self.teacher_config = TeacherConfig()
        for key, value in asdict(self.teacher_config).items():
            if getattr(self.base, key) != value:
                raise ValueError(f"teacher/environment contract mismatch: {key}")
        if self.base.hocbf_top_k is not None:
            raise ValueError("all LiDAR constraints must remain enabled")
        self.base.cbf_enabled = bool(hocbf)
        self.base.safety_filter = self.teacher_config.make_filter() if hocbf else None
        # This term was dormant in raw SAC. Do not silently activate an extra
        # reward penalty when adding the filter; correction enters actor loss only.
        self.base.safety_intervention_penalty = 0.0
        self.hocbf_steps = self.hocbf_fallback_steps = self.hocbf_emergency_steps = 0

    def reset(self, *, seed=None, options=None):
        self.hocbf_steps = self.hocbf_fallback_steps = self.hocbf_emergency_steps = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        before = (self.base.safety_interventions, self.base.safety_fallbacks,
                  self.base.safety_emergency_brakes)
        obs, reward, done, truncated, info = super().step(action)
        flags = tuple(int(new > old) for new, old in zip(
            (self.base.safety_interventions, self.base.safety_fallbacks,
             self.base.safety_emergency_brakes), before, strict=True))
        self.hocbf_steps += flags[0]
        self.hocbf_fallback_steps += flags[1]
        self.hocbf_emergency_steps += flags[2]
        info.update(hocbf_intervened=bool(flags[0]), hocbf_fallback=bool(flags[1]),
                    hocbf_emergency=bool(flags[2]))
        if "navigation_episode" in info:
            info["navigation_episode"].update(
                hocbf_intervention_steps=self.hocbf_steps,
                hocbf_fallback_steps=self.hocbf_fallback_steps,
                hocbf_emergency_steps=self.hocbf_emergency_steps)
        return obs, reward, done, truncated, info
