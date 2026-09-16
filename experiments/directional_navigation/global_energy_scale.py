"""Single-factor predictive-scale repair; never modifies frozen head weights."""
from __future__ import annotations
import math
import torch
from torch import nn


def fit_global_sigma(log_residual: torch.Tensor, weights: torch.Tensor,
                     minimum: float = .05) -> float:
    """Weighted Gaussian residual NLL minimizer with fixed location.

    The input must contain OLD calibration safe states only. This function
    does not interpret the fitted scale as physical aleatoric uncertainty.
    """
    r, w = log_residual.double(), weights.double()
    if (r.ndim != 1 or r.shape != w.shape or not len(r) or minimum <= 0
            or not torch.isfinite(r).all() or not torch.isfinite(w).all()
            or torch.any(w < 0) or w.sum() <= 0):
        raise ValueError('finite aligned residuals and positive total weight required')
    return max(minimum, float(((w*r.square()).sum()/w.sum()).sqrt()))


class GlobalScaleEnergyHead(nn.Module):
    """Same f and mu exactly; only state-dependent sigma is removed.

    Forward returns [failure logit, log-energy location, sigma], unlike the
    legacy head's third raw-softplus output. Call probability/finite_mean.
    """
    def __init__(self, frozen_head: nn.Module, sigma: float, energy_scale: float):
        super().__init__()
        if not math.isfinite(sigma) or sigma <= 0 or not math.isfinite(energy_scale) or energy_scale <= 0:
            raise ValueError('positive finite scales required')
        self.head = frozen_head
        self.head.eval()
        for p in self.head.parameters(): p.requires_grad_(False)
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float64))
        self.register_buffer('energy_scale', torch.tensor(energy_scale, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.head.eval()
        p = self.head(x).double()
        return torch.stack((p[:,0],p[:,1],self.sigma.expand(len(p))),dim=1)

    def probability(self, x: torch.Tensor, budgets: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(budgets).all(): raise ValueError('finite budgets required')
        p = self(x)
        b = budgets.to(device=p.device,dtype=torch.float64)
        z = (torch.log(b.clamp_min(torch.finfo(torch.float64).tiny)/self.energy_scale)[None,:]-p[:,1,None])/self.sigma
        prob = (1-torch.sigmoid(p[:,0,None]))*.5*(1+torch.erf(z/math.sqrt(2)))
        return torch.where(b[None,:] > 0,prob,0.)

    def finite_mean(self, x: torch.Tensor) -> torch.Tensor:
        p = self(x)
        mean = self.energy_scale*torch.exp(p[:,1]+.5*self.sigma.square())
        if not torch.isfinite(mean).all():
            raise FloatingPointError('location extrapolation makes mean nonfinite; do not clip')
        return mean
