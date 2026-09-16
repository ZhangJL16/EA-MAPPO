from __future__ import annotations

import torch
from torch import nn


class ContractiveResidualMemory(nn.Module):
    """Small recurrent residual model with an explicit hidden-state contraction.

    The contraction applies only to hidden-state sensitivity for fixed inputs. It
    does not, by itself, certify physical state-estimation error.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        output_size: int = 3,
        *,
        recurrent_norm_bound: float = 0.8,
        leak: float = 0.5,
        output_bound: float = 10.0,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or output_size <= 0:
            raise ValueError("network dimensions must be positive")
        if not 0.0 <= recurrent_norm_bound < 1.0:
            raise ValueError("recurrent_norm_bound must lie in [0, 1)")
        if not 0.0 < leak <= 1.0:
            raise ValueError("leak must lie in (0, 1]")
        contraction_factor = (1.0 - float(leak)) + float(leak) * float(
            recurrent_norm_bound
        )
        if not contraction_factor <= 1.0 - 1e-6:
            raise ValueError(
                "the execution-dtype contraction factor must have at least a 1e-6 gap"
            )
        if output_bound <= 0.0:
            raise ValueError("output_bound must be positive")
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.recurrent = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.recurrent_norm_bound = float(recurrent_norm_bound)
        self.leak = float(leak)
        self.output_bound = float(output_bound)
        nn.init.orthogonal_(self.recurrent.weight)
        self.project_recurrent_weight()

    @property
    def contraction_factor(self) -> float:
        return (1.0 - self.leak) + self.leak * self.recurrent_norm_bound

    @torch.no_grad()
    def project_recurrent_weight(self) -> None:
        spectral_norm = torch.linalg.matrix_norm(self.recurrent.weight, ord=2)
        if spectral_norm > self.recurrent_norm_bound:
            self.recurrent.weight.mul_(self.recurrent_norm_bound / spectral_norm)

    def effective_recurrent_weight(self) -> torch.Tensor:
        self._require_supported_parameter_dtype()
        spectral_norm = torch.linalg.matrix_norm(self.recurrent.weight, ord=2)
        scale = torch.clamp(
            self.recurrent_norm_bound / torch.clamp(spectral_norm, min=1e-12),
            max=1.0,
        )
        return self.recurrent.weight * scale

    def _require_supported_parameter_dtype(self) -> None:
        if self.recurrent.weight.dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "the contraction implementation supports only float32 or float64"
            )

    def initial_hidden(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        parameter = next(self.parameters())
        return torch.zeros(
            batch_size,
            self.hidden_size,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[1] != self.input_layer.in_features:
            raise ValueError("features have the wrong shape")
        if hidden.shape != (features.shape[0], self.hidden_size):
            raise ValueError("hidden state has the wrong shape")
        self._require_supported_parameter_dtype()
        parameter_dtype = self.recurrent.weight.dtype
        if features.dtype != parameter_dtype or hidden.dtype != parameter_dtype:
            raise TypeError("features, hidden state, and parameters must share dtype")
        recurrent = torch.nn.functional.linear(
            hidden,
            self.effective_recurrent_weight(),
        )
        candidate = torch.tanh(self.input_layer(features) + recurrent)
        next_hidden = (1.0 - self.leak) * hidden + self.leak * candidate
        residual = self.output_bound * torch.tanh(self.output_layer(next_hidden))
        return residual, next_hidden
