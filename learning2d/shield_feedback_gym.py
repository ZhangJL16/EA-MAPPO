"""Frozen shielded plant with an optional safety-takeover learning signal."""

from __future__ import annotations

from .finite_horizon_gym import FiniteHorizonGym


def intervention_penalty(info: dict, *, feedback: bool) -> tuple[float, int, int]:
    """Extra reward only; the parent already charges 0.2 per takeover."""
    events = [row.get("event") for row in info["plant_trace"]]
    takeovers = events.count("return_takeover")
    rejected_departures = events.count("departure_rejected")
    extra = -1.8 * takeovers - 2.0 * rejected_departures if feedback else 0.0
    return extra, takeovers, rejected_departures


class ShieldFeedbackGym(FiniteHorizonGym):
    def __init__(self, *args, feedback: bool, **kwargs) -> None:
        if kwargs.get("shielded") is not True:
            raise ValueError("paired training requires the common shielded plant")
        super().__init__(*args, **kwargs)
        self.feedback = feedback

    def step(self, action: int):
        observation, reward, terminated, truncated, info = super().step(action)
        extra, takeovers, rejected = intervention_penalty(info, feedback=self.feedback)
        return observation, reward + extra, terminated, truncated, {
            **info,
            "safety_feedback_extra_reward": extra,
            "step_return_takeovers": takeovers,
            "step_rejected_departures": rejected,
        }
