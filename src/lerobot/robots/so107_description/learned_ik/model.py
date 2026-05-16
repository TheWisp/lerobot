"""
Tiny MLP for learned IK on SO-107.

Architecture:
    input:  [current_joints (7), target_ee_delta (3)]      # 10 features
    output: target_joints (7)                              # 7 motor degrees at t+1

The "delta" framing (target_ee = current_ee + delta) is crucial:
- Network learns the LOCAL relationship between EE displacement and joint
  motion — i.e. exactly what we need for incremental teleop.
- Inference time mode: caller passes current motors + desired EE delta;
  network returns motor positions that would achieve it (in the data-implied
  human-natural way).

Normalization is baked into the model: stats computed at training time,
constants serialized with the weights. The wrapper does:
    x_norm = (x - x_mean) / x_std
    y_norm = MLP(x_norm)
    y = y_norm * y_std + y_mean
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class IKModelConfig:
    in_dim: int = 10  # 7 joints + 3 ee_delta
    out_dim: int = 7  # 7 joint outputs
    hidden_dims: tuple[int, ...] = (128, 128)
    activation: str = "silu"  # silu is smoother than relu, better for regression


class IKMLP(nn.Module):
    """Tiny MLP. Input: [normed joints, normed ee_delta]. Output: normed delta joints."""

    def __init__(self, config: IKModelConfig | None = None):
        super().__init__()
        self.config = config or IKModelConfig()
        act = {"relu": nn.ReLU, "silu": nn.SiLU, "gelu": nn.GELU}[self.config.activation]
        dims = (self.config.in_dim, *self.config.hidden_dims, self.config.out_dim)
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

        # Normalization stats (buffers — moved with .to(device), saved with state_dict).
        self.register_buffer("in_mean", torch.zeros(self.config.in_dim))
        self.register_buffer("in_std", torch.ones(self.config.in_dim))
        self.register_buffer("out_mean", torch.zeros(self.config.out_dim))
        self.register_buffer("out_std", torch.ones(self.config.out_dim))

    def set_normalization(
        self,
        in_mean: torch.Tensor,
        in_std: torch.Tensor,
        out_mean: torch.Tensor,
        out_std: torch.Tensor,
    ) -> None:
        self.in_mean.copy_(in_mean)
        self.in_std.copy_(in_std.clamp_min(1e-6))
        self.out_mean.copy_(out_mean)
        self.out_std.copy_(out_std.clamp_min(1e-6))

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """x_raw: (B, 10) — [joints_deg, ee_delta_m]. Returns delta_joints_deg (B, 7)."""
        x_norm = (x_raw - self.in_mean) / self.in_std
        y_norm = self.net(x_norm)
        y = y_norm * self.out_std + self.out_mean
        return y


def build_input(joints_deg: torch.Tensor, ee_delta_m: torch.Tensor) -> torch.Tensor:
    """Concat current joints and desired EE delta into the (B, 10) input tensor."""
    return torch.cat([joints_deg, ee_delta_m], dim=-1)
