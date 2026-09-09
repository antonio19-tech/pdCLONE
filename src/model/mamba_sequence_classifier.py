from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * scale * self.weight


class MambaMixer(nn.Module):
    """
    Lightweight pure-PyTorch Mamba-style mixer.

    This keeps the selective state-space recurrence and local depthwise
    convolution that make Mamba useful for sequence modelling, but avoids the
    external `mamba_ssm` dependency so the baseline runs in this repo as-is.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        d_conv: int,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self._d_state = d_state
        self._d_conv = d_conv
        self._inner_dim = expand * d_model
        self._dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self._inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            self._inner_dim,
            self._inner_dim,
            kernel_size=d_conv,
            groups=self._inner_dim,
            padding=0,
            bias=True,
        )
        self.x_proj = nn.Linear(
            self._inner_dim,
            self._dt_rank + (2 * d_state),
            bias=False,
        )
        self.dt_proj = nn.Linear(self._dt_rank, self._inner_dim, bias=True)
        self.A_log = nn.Parameter(
            torch.log(
                torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self._inner_dim, 1)
            )
        )
        self.D = nn.Parameter(torch.ones(self._inner_dim))
        self.out_proj = nn.Linear(self._inner_dim, d_model, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x, z = self.in_proj(hidden_states).chunk(2, dim=-1)

        x = x.transpose(1, 2)
        x = F.pad(x, (self._d_conv - 1, 0))
        x = self.conv1d(x).transpose(1, 2)
        x = F.silu(x)

        if mask is not None:
            mask_float = mask.unsqueeze(-1).to(dtype=x.dtype)
            x = x * mask_float
            z = z * mask_float

        y = self._selective_scan(x, mask=mask)
        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.x_proj(x)
        delta, b_term, c_term = torch.split(
            projected,
            [self._dt_rank, self._d_state, self._d_state],
            dim=-1,
        )
        delta = F.softplus(self.dt_proj(delta))

        if mask is not None:
            mask_float = mask.unsqueeze(-1).to(dtype=x.dtype)
            delta = delta * mask_float
            b_term = b_term * mask_float
            c_term = c_term * mask_float
            x = x * mask_float

        a = -torch.exp(self.A_log).to(dtype=x.dtype)
        state = x.new_zeros(x.size(0), self._inner_dim, self._d_state)
        outputs: list[torch.Tensor] = []

        for timestep in range(x.size(1)):
            dt = delta[:, timestep].unsqueeze(-1)
            transition = torch.exp(dt * a.unsqueeze(0))
            b_t = b_term[:, timestep].unsqueeze(1)
            c_t = c_term[:, timestep].unsqueeze(1)
            x_t = x[:, timestep].unsqueeze(-1)

            state = transition * state + (dt * b_t * x_t)
            y_t = (state * c_t).sum(dim=-1) + (self.D * x[:, timestep])

            if mask is not None:
                y_t = y_t * mask[:, timestep].unsqueeze(-1).to(dtype=y_t.dtype)

            outputs.append(y_t)

        return torch.stack(outputs, dim=1)


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int,
        expand: int,
        d_conv: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(
            d_model=d_model,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mixed = self.mixer(self.norm(hidden_states), mask=mask)
        return hidden_states + self.dropout(mixed)


class MambaSequenceClassifier(nn.Module):
    """
    Baseline classifier for raw PaHaW trajectories.

    Input shape: [batch, timesteps, features]
    Output shape: [batch, num_classes]
    """

    def __init__(
        self,
        num_classes: int,
        input_dim: int,
        d_model: int = 64,
        n_layers: int = 2,
        d_state: int = 16,
        expand: int = 2,
        d_conv: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [
                MambaBlock(
                    d_model=d_model,
                    d_state=d_state,
                    expand=expand,
                    d_conv=d_conv,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(
        self,
        sequences: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.input_proj(sequences)
        for block in self.blocks:
            hidden_states = block(hidden_states, mask=mask)

        hidden_states = self.norm(hidden_states)
        pooled = self._masked_mean_pool(hidden_states, mask=mask, lengths=lengths)
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

    @staticmethod
    def _masked_mean_pool(
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            return hidden_states.mean(dim=1)

        mask_float = mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        denominator = mask_float.sum(dim=1).clamp_min(1.0)
        if lengths is not None:
            denominator = lengths.unsqueeze(-1).to(dtype=hidden_states.dtype).clamp_min(1.0)

        return (hidden_states * mask_float).sum(dim=1) / denominator
