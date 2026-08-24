from __future__ import annotations

from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .dataset import SafetyBridgeTrajectoryWriter
from .safety_buffer import SafetyBridgeReplay


class SafetyBridgeCollectionCallback(BaseCallback):
    def __init__(
        self,
        *,
        replay: SafetyBridgeReplay,
        trajectory_writer: SafetyBridgeTrajectoryWriter,
        metrics_path: str | Path | None = None,
        log_frequency_transitions: int = 10_000,
    ) -> None:
        super().__init__(verbose=0)
        if log_frequency_transitions <= 0:
            raise ValueError("log_frequency_transitions must be positive")
        self.replay = replay
        self.trajectory_writer = trajectory_writer
        self.metrics_path = None if metrics_path is None else Path(metrics_path)
        self.log_frequency_transitions = int(log_frequency_transitions)
        self._next_log = int(log_frequency_transitions)
        self._interventions = 0
        self._emergencies = 0
        self._fallbacks = 0
        self._valid = 0
        self._total = 0
        self._authorities: list[float] = []
        self._normal_fractions: list[float] = []
        self._intervention_norms: list[float] = []
        self._ranks: list[int] = []
        self._valid_authorities: list[float] = []
        self._constraint_build_seconds: list[float] = []
        self._solver_seconds: list[float] = []
        self._filter_total_seconds: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals["infos"]
        dones = np.asarray(self.locals["dones"], dtype=bool)
        for env_index, info in enumerate(infos):
            self.replay.add_from_info(info)
            self.trajectory_writer.observe(env_index, info, bool(dones[env_index]))
            geometry = info["projection_geometry"]
            valid = bool(
                geometry.get("valid", False)
                and geometry.get("active_set_stable", False)
                and geometry.get("coordinate_map_stable", False)
                and not info.get("hocbf_fallback_used", False)
                and not info.get("hocbf_emergency_brake", False)
            )
            self._total += 1
            self._valid += int(valid)
            self._interventions += int(bool(info.get("hocbf_intervened", False)))
            self._emergencies += int(bool(info.get("hocbf_emergency_brake", False)))
            self._fallbacks += int(bool(info.get("hocbf_fallback_used", False)))
            self._authorities.append(float(geometry.get("action_authority", 0.0)))
            if valid:
                self._valid_authorities.append(
                    float(geometry.get("action_authority", 0.0))
                )
            self._normal_fractions.append(float(geometry.get("normal_action_fraction", 0.0)))
            self._intervention_norms.append(float(info.get("hocbf_intervention_norm", 0.0)))
            self._ranks.append(int(geometry.get("rank", 0)))
            self._constraint_build_seconds.append(
                float(info.get("hocbf_constraint_build_seconds", 0.0))
            )
            self._solver_seconds.append(float(info.get("hocbf_solver_seconds", 0.0)))
            self._filter_total_seconds.append(
                float(info.get("hocbf_filter_total_seconds", 0.0))
            )
        if self.metrics_path is not None and self.num_timesteps >= self._next_log:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            record = self.metrics()
            record["global_env_transitions"] = int(self.num_timesteps)
            import json

            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            self._next_log += self.log_frequency_transitions
        return True

    def metrics(self) -> dict[str, object]:
        total = max(self._total, 1)
        ranks = np.asarray(self._ranks, dtype=np.int64)
        return {
            "transitions": self._total,
            "valid_jacobian_rate": float(self._valid / total),
            "nominal_safe_action_rate": float(
                1.0 - self._interventions / total
            ),
            "hocbf_intervention_rate": float(self._interventions / total),
            "hocbf_emergency_rate": float(self._emergencies / total),
            "hocbf_fallback_rate": float(self._fallbacks / total),
            "mean_authority": float(np.mean(self._authorities)) if self._authorities else 0.0,
            "mean_valid_authority": (
                float(np.mean(self._valid_authorities))
                if self._valid_authorities
                else None
            ),
            "mean_normal_action_fraction": (
                float(np.mean(self._normal_fractions)) if self._normal_fractions else 0.0
            ),
            "mean_intervention_norm": (
                float(np.mean(self._intervention_norms)) if self._intervention_norms else 0.0
            ),
            "p90_intervention_norm": (
                float(np.quantile(self._intervention_norms, 0.9))
                if self._intervention_norms
                else 0.0
            ),
            "rank_histogram": {
                str(rank): int(np.sum(ranks == rank)) for rank in range(4)
            },
            "mean_constraint_build_seconds": (
                float(np.mean(self._constraint_build_seconds))
                if self._constraint_build_seconds
                else 0.0
            ),
            "mean_solver_seconds": (
                float(np.mean(self._solver_seconds)) if self._solver_seconds else 0.0
            ),
            "mean_filter_total_seconds": (
                float(np.mean(self._filter_total_seconds))
                if self._filter_total_seconds
                else 0.0
            ),
            "p95_filter_total_seconds": (
                float(np.quantile(self._filter_total_seconds, 0.95))
                if self._filter_total_seconds
                else 0.0
            ),
            "replay": self.replay.metadata(),
            "trajectory_dataset": self.trajectory_writer.metadata(),
        }
