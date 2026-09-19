"""DAD location-finding model and sequential PCE, with explicit base randomness.

Model/architecture adapted from ae-foster/dad at 4b1008174e1531d1f14601d83cef481c0f586f36.
See THIRD_PARTY.md for attribution and the MIT license. No Pyro/MLflow dependency.
"""
import math
import torch
from torch import nn


def mean_signal(design, theta):
    """design (..., 2), theta (..., K=2, 2); observations are log intensity."""
    distance = (design.unsqueeze(-2) - theta).square().sum(-1)
    return (0.1 + (1e-4 + distance).reciprocal().sum(-1)).log()


class Policy(nn.Module):
    """Only public history + remaining budget; no theta/likelihood evaluator input.

    dad_native is the author's sum encoder + linear emitter, without budget.
    Other variants all concatenate remaining/8 at the same linear emitter.
    Attention uses encodings as keys AND values: no extra projection confound.
    Multiplying attention by history length makes uniform weights equal sum pool.
    """
    def __init__(self, kind, hidden=64, encoding=16, budget_scale=8):
        super().__init__()
        if kind not in {"pool", "pool_wide", "fixed_attention", "budget_attention", "dad_native"}:
            raise ValueError(kind)
        self.kind, self.encoding, self.budget_scale = kind, encoding, budget_scale
        if kind == "pool_wide":
            hidden += 2
        self.encoder = nn.Sequential(nn.Linear(3, hidden), nn.ReLU(), nn.Linear(hidden, encoding))
        self.emitter = nn.Linear(encoding + (kind != "dad_native"), 2)
        if "attention" in kind:
            self.query_base = nn.Parameter(torch.zeros(encoding))
        if kind == "budget_attention":
            self.query_slope = nn.Parameter(torch.zeros(encoding))

    def forward(self, history, remaining, query_override=None):
        # history: [batch, time, (design_x, design_y, observation)]
        batch, count, _ = history.shape
        n = history.new_full((batch, 1), float(remaining) / self.budget_scale)
        z = history.new_zeros(batch, self.encoding)
        if count:
            tokens = self.encoder(history)
            if "attention" in self.kind:
                query = self.query_base
                if self.kind == "budget_attention":
                    read_n = float(remaining if query_override is None else query_override) / self.budget_scale
                    query = query + read_n * self.query_slope
                weights = (tokens * query).sum(-1).div(math.sqrt(self.encoding)).softmax(-1)
                z = count * (tokens * weights.unsqueeze(-1)).sum(1)
            else:
                z = tokens.sum(1)
        return self.emitter(z if self.kind == "dad_native" else torch.cat((z, n), -1))


def coupled_policy(kind, seed, hidden=64, encoding=16):
    """Identical initial common weights for the three main ablations."""
    torch.manual_seed(seed)
    base = Policy("pool", hidden, encoding)
    torch.manual_seed(seed)
    policy = Policy(kind, hidden, encoding)
    target = policy.state_dict()
    for name, value in base.state_dict().items():
        if name in target and target[name].shape == value.shape:
            target[name] = value.clone()
    policy.load_state_dict(target)
    return policy


def random_inputs(seed, batch, horizon, contrasts, device="cpu"):
    # Independent per outer rollout contrastives, not shared inner particles.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    theta = torch.randn(batch, 2, 2, generator=gen)
    noise = torch.randn(batch, horizon, generator=gen)
    inner = torch.randn(contrasts, batch, 2, 2, generator=gen)
    return tuple(v.to(device) for v in (theta, noise, inner))


def rollout(policy, theta, noise, query_override=None):
    """Reparameterized environment; full gradients through released history."""
    horizon = noise.shape[1]
    history = theta.new_empty(theta.shape[0], 0, 3)
    for t in range(horizon):
        design = policy(history, horizon - t, query_override)
        observation = mean_signal(design, theta) + 0.5 * noise[:, t]
        history = torch.cat((history, torch.cat((design, observation[:, None]), -1)[:, None]), 1)
    return history


def log_likelihood(history, theta):
    """Fixed observed designs/history, including for contrastive latents.

    Contrastive particles MUST NOT get their own re-planned trajectories.
    theta: [L, batch, K, p] or [batch, K, p]. Return [L,batch] or [batch].
    """
    total = 0
    for t in range(history.shape[1]):
        mean = mean_signal(history[:, t, :2], theta)
        total = total + torch.distributions.Normal(mean, 0.5).log_prob(history[:, t, 2])
    return total


def information_samples(history, theta, inner, chunk=256):
    primary = log_likelihood(history, theta)
    denominator = None
    for particles in inner.split(chunk):
        part = log_likelihood(history, particles).logsumexp(0)
        denominator = part if denominator is None else torch.logaddexp(denominator, part)
    if denominator is None:
        raise ValueError("At least one independent contrastive sample required")
    lower = primary - torch.logaddexp(primary, denominator) + math.log(len(inner) + 1)
    upper = primary - denominator + math.log(len(inner))
    return lower, upper


def objective(policy, inputs, query_override=None):
    theta, noise, inner = inputs
    history = rollout(policy, theta, noise, query_override)
    return information_samples(history, theta, inner)[0].mean()
