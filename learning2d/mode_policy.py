"""PPO policy with physically meaningful actions in each operating mode.

The simulator's legacy 13-action interface aliases flight actions at the dock
and charge actions in flight.  This policy keeps the simulator untouched but
never assigns probability to those aliases.  It does not decide whether a
departure is energy safe; the policy must learn when to charge and return.
"""

from __future__ import annotations

import torch
from gymnasium.spaces import Discrete
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy

from dual_constraint_2d.action_adapter import ACTION_COUNT, FEATURE_COUNT


DOCK_FEATURE = 10
FLIGHT_FEATURE = 11
ENERGY_FEATURE = 8
CHARGE_FRACTIONS = (0.5, 0.75, 1.0)


def valid_action_mask(observation: torch.Tensor) -> torch.Tensor:
    """Return [batch, 13] mask for observations at policy decision points."""
    if observation.ndim != 2 or observation.shape[1] != FEATURE_COUNT:
        raise ValueError("expected a batch of 40-dimensional observations")
    dock = observation[:, DOCK_FEATURE] > 0.5
    flight = observation[:, FLIGHT_FEATURE] > 0.5
    if not torch.all(dock ^ flight):
        raise ValueError("each policy decision must be docked or in flight")
    mask = torch.zeros(
        (observation.shape[0], ACTION_COUNT), dtype=torch.bool,
        device=observation.device,
    )
    mask[flight, :10] = True
    mask[dock, 4] = True
    for action, fraction in enumerate(CHARGE_FRACTIONS, start=10):
        mask[:, action] = dock & (observation[:, ENERGY_FEATURE] + 1e-8 < fraction)
    return mask


class ModeMaskedPolicy(ActorCriticPolicy):
    """Actor-critic whose categorical support follows dock/flight semantics."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_space, Discrete) or self.action_space.n != ACTION_COUNT:
            raise ValueError("ModeMaskedPolicy requires the 13-action interface")
        if not isinstance(self.action_dist, CategoricalDistribution):
            raise ValueError("ModeMaskedPolicy requires a categorical distribution")

    def _mode_distribution(self, observation: torch.Tensor, latent_pi: torch.Tensor):
        logits = self.action_net(latent_pi)
        logits = logits.masked_fill(~valid_action_mask(observation), -1e9)
        return self.action_dist.proba_distribution(action_logits=logits)

    def _latents(self, observation: torch.Tensor):
        features = self.extract_features(observation)
        if self.share_features_extractor:
            return self.mlp_extractor(features)
        pi_features, vf_features = features
        return (
            self.mlp_extractor.forward_actor(pi_features),
            self.mlp_extractor.forward_critic(vf_features),
        )

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        latent_pi, latent_vf = self._latents(obs)
        values = self.value_net(latent_vf)
        distribution = self._mode_distribution(obs, latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape)), values, distribution.log_prob(actions)

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        latent_pi, latent_vf = self._latents(obs)
        distribution = self._mode_distribution(obs, latent_pi)
        return (
            self.value_net(latent_vf),
            distribution.log_prob(actions),
            distribution.entropy(),
        )

    def get_distribution(self, obs: torch.Tensor):
        features = self.extract_features(obs)
        if isinstance(features, tuple):
            features = features[0]
        latent_pi = self.mlp_extractor.forward_actor(features)
        return self._mode_distribution(obs, latent_pi)
