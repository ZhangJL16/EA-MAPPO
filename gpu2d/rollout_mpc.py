"""Batched GPU proposal rollout with exact CPU certification of chosen actions.

The Torch rollout is a ranking proxy, never a return-energy or collision
certificate.  The existing exact shield validates each shortlisted action and
the existing environment validates its execution.  This keeps safety semantics
unchanged while moving broad candidate search to the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from time import perf_counter

import numpy as np
import torch

from dual_constraint_2d.action_adapter import ActionAdapter
from dual_constraint_2d.shield import certify_policy_block


@dataclass(frozen=True)
class GPUStats:
    decisions: int
    gpu_candidates: int
    exact_candidates: int
    gpu_wall_seconds: float
    exact_wall_seconds: float


class BatchedRolloutMPC:
    def __init__(self, case, *, horizon_blocks: int = 4,
                 exact_shortlist: int = 3, device: str = "cuda") -> None:
        if horizon_blocks not in (4, 8) or not 1 <= exact_shortlist <= 9:
            raise ValueError("unsupported rollout horizon or shortlist")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.case = case
        self.horizon_blocks = horizon_blocks
        self.exact_shortlist = exact_shortlist
        self.adapter = ActionAdapter(case)
        self.discs = torch.tensor([
            (float(o.center_xy[0]), float(o.center_xy[1]), float(o.radius))
            for o in case.world.obstacles
        ], dtype=torch.float32, device=self.device)
        self.station = torch.tensor(case.station_xy, dtype=torch.float32, device=self.device)
        self.decisions = 0
        self.gpu_candidates = 0
        self.exact_candidates = 0
        self.gpu_wall_seconds = 0.0
        self.exact_wall_seconds = 0.0
        # 9 initial actions x 9 continuation actions x 4 switch patterns.
        self.first_ids = torch.arange(9, device=self.device).repeat_interleave(36)
        self.later_ids = torch.arange(9, device=self.device).repeat_interleave(4).repeat(9)
        self.switch_ids = torch.arange(4, device=self.device).repeat(81)

    @property
    def stats(self) -> GPUStats:
        return GPUStats(self.decisions, self.gpu_candidates,
                        self.exact_candidates, self.gpu_wall_seconds,
                        self.exact_wall_seconds)

    def _route_proxy(self, position: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        direct = goal[None, :] - position
        length = torch.linalg.vector_norm(direct, dim=1).clamp_min(1e-6)
        unit = direct / length[:, None]
        relative = self.discs[None, :, :2] - position[:, None, :]
        projection = (relative * unit[:, None, :]).sum(dim=2)
        clamped = projection.clamp_min(0.0).minimum(length[:, None])
        closest = position[:, None, :] + clamped[:, :, None] * unit[:, None, :]
        clearance = torch.linalg.vector_norm(
            closest - self.discs[None, :, :2], dim=2
        ) - self.discs[None, :, 2]
        blocked = (clearance < 1.1) & (projection > 0.0) & (projection < length[:, None])
        first = torch.where(blocked, projection, torch.full_like(projection, 1e6)).argmin(dim=1)
        selected = self.discs[first]
        has_blocker = blocked.any(dim=1)
        cross = unit[:, 0] * relative[torch.arange(len(position), device=self.device), first, 1] - unit[:, 1] * relative[torch.arange(len(position), device=self.device), first, 0]
        side = torch.where(cross >= 0.0, -1.0, 1.0)
        perpendicular = torch.stack((-unit[:, 1], unit[:, 0]), dim=1)
        waypoint = selected[:, :2] + side[:, None] * perpendicular * (selected[:, 2:3] + 1.5)
        direction = torch.where(has_blocker[:, None], waypoint - position, direct)
        return direction / torch.linalg.vector_norm(direction, dim=1).clamp_min(1e-6)[:, None]

    def _proposal_scores(self, env) -> np.ndarray:
        n = len(self.first_ids)
        device = self.device
        p = torch.as_tensor(env.state.position[:2], dtype=torch.float32, device=device).repeat(n, 1)
        v = torch.as_tensor(env.state.velocity[:2], dtype=torch.float32, device=device).repeat(n, 1)
        energy = torch.full((n,), float(env.energy), device=device)
        contact_count = torch.zeros(n, dtype=torch.int32, device=device)
        previous_contact = torch.zeros(n, dtype=torch.bool, device=device)
        goal = torch.as_tensor(env.target.position_xy, dtype=torch.float32, device=device)
        cfg = env.case.config
        dt = cfg.physics_dt_s
        obs = env.observe()
        initial_commands = torch.tensor([
            self.adapter.decode(i, obs).desired_velocity_xy for i in range(9)
        ], dtype=torch.float32, device=device)
        offsets = torch.tensor((-pi / 6, 0.0, pi / 6), device=device)
        speeds = torch.tensor((3.5, 5.0, 7.0), device=device)
        first_dist = torch.linalg.vector_norm(goal[None, :] - p, dim=1)
        for block in range(self.horizon_blocks):
            later = torch.where(
                block < (self.switch_ids + 1) * self.horizon_blocks // 4,
                self.first_ids, self.later_ids,
            )
            heading = self._route_proxy(p, goal)
            theta = offsets[later // 3]
            c, s = torch.cos(theta), torch.sin(theta)
            rotated = torch.stack((c * heading[:, 0] - s * heading[:, 1],
                                   s * heading[:, 0] + c * heading[:, 1]), dim=1)
            desired = rotated * speeds[later % 3, None]
            distance = torch.linalg.vector_norm(goal[None, :] - p, dim=1)
            desired *= torch.minimum(torch.ones_like(distance),
                                     0.8 * distance / speeds[later % 3])[:, None]
            if block == 0:
                desired = initial_commands[self.first_ids]
            for _ in range(cfg.policy_hold_steps):
                nominal = 2.0 * (desired - v)
                norm = torch.linalg.vector_norm(nominal, dim=1).clamp_min(1e-6)
                nominal *= torch.minimum(torch.ones_like(norm),
                                         torch.full_like(norm, cfg.max_acceleration_mps2) / norm)[:, None]
                candidate_v = v + dt * nominal
                speed = torch.linalg.vector_norm(candidate_v, dim=1).clamp_min(1e-6)
                candidate_v *= torch.minimum(torch.ones_like(speed),
                                             torch.full_like(speed, cfg.max_speed_mps) / speed)[:, None]
                realized = (candidate_v - v) / dt
                candidate_p = p + dt * v + 0.5 * dt * dt * realized
                segment = candidate_p - p
                rel = self.discs[None, :, :2] - p[:, None, :]
                denom = (segment * segment).sum(dim=1).clamp_min(1e-12)
                closest_t = ((rel * segment[:, None, :]).sum(dim=2) / denom[:, None]).clamp(0.0, 1.0)
                closest = p[:, None, :] + closest_t[:, :, None] * segment[:, None, :]
                obstacle = (torch.linalg.vector_norm(
                    closest - self.discs[None, :, :2], dim=2
                ) <= self.discs[None, :, 2] + cfg.body_radius_m).any(dim=1)
                boundary = ((candidate_p < cfg.body_radius_m) |
                            (candidate_p > cfg.side_m - cfg.body_radius_m)).any(dim=1)
                contact = obstacle | boundary
                energy -= dt * (env.energy_model.idle_rate
                                + env.energy_model.speed_squared_rate * (v * v).sum(dim=1)
                                + env.energy_model.acceleration_squared_rate * (realized * realized).sum(dim=1))
                p = torch.where(contact[:, None], p, candidate_p)
                v = torch.where(contact[:, None], torch.zeros_like(v), candidate_v)
                contact_count += contact.int()
                previous_contact = contact
        goal_distance = torch.linalg.vector_norm(goal[None, :] - p, dim=1)
        station_distance = torch.linalg.vector_norm(self.station[None, :] - p, dim=1)
        # The return term ranks proposals only.  Exact return witnesses are
        # always checked by certify_policy_block and DualConstraintEnv.
        reserve_gap = energy - (station_distance * 0.30 + 0.05 * env.charger.capacity)
        scores = (first_dist - goal_distance) / 5.0 - 0.03 * (env.energy - energy)
        scores -= 100.0 * contact_count.float()
        scores -= 100.0 * torch.relu(-reserve_gap)
        self.gpu_candidates += n
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return scores.detach().cpu().numpy()

    def act(self, env) -> int:
        if env.mode == "return":
            return 9
        if env._flight_step_limit() == 0:
            return 9
        if env.mode == "docked":
            needed = min(env.charger.capacity,
                         env.target.reference_round_trip_energy + 0.05 * env.charger.capacity)
            if env.energy + 1e-8 < needed:
                for action, fraction in ((10, 0.5), (11, 0.75), (12, 1.0)):
                    if env.charger.capacity * fraction + 1e-8 >= needed:
                        return action
            return 4
        self.decisions += 1
        began = perf_counter()
        scores = self._proposal_scores(env)
        self.gpu_wall_seconds += perf_counter() - began
        order = np.argsort(-scores, kind="stable")
        unique: list[int] = []
        for index in order:
            action = int(self.first_ids[index].item())
            if action not in unique:
                unique.append(action)
            if len(unique) == self.exact_shortlist:
                break
        began = perf_counter()
        best = None
        for action in unique:
            desired = self.adapter.decode(action, env.observe()).desired_velocity_xy
            certified = certify_policy_block(
                env.case, env.state, env.energy, desired,
                router=env.router, current_backup=env.backup,
                energy_model=env.energy_model, hold_steps=env._flight_step_limit(),
            )
            self.exact_candidates += 1
            if certified.accepted_policy:
                best = action
                break
        self.exact_wall_seconds += perf_counter() - began
        return 9 if best is None else best
