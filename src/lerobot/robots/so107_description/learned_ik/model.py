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
    # in_dim default 22: current_joints(7) + current_ee_rot(9) + ee_delta(3) + target_ee_rot(3, axis-angle)
    # Older "position-only" models used in_dim=10; loaders dispatch on this field.
    in_dim: int = 22
    out_dim: int = 7  # joint deltas
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    activation: str = "silu"


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
    """Position-only input (legacy 10-D). Use build_input_full_pose for new training."""
    return torch.cat([joints_deg, ee_delta_m], dim=-1)


def rot_to_axis_angle(R: torch.Tensor) -> torch.Tensor:  # noqa: N803  (R is the rotation matrix)
    """Rotation matrices (B, 3, 3) -> axis-angle (B, 3). Continuous and small for small rotations."""
    # Use the standard log-of-rotation formula. Clamp to avoid asin domain issues.
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((trace - 1) / 2).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos)
    # Avoid division by zero near identity.
    safe_sin = torch.where(theta.abs() < 1e-6, torch.ones_like(theta), torch.sin(theta))
    axis = torch.stack(
        [
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1],
        ],
        dim=-1,
    ) / (2 * safe_sin.unsqueeze(-1))
    return axis * theta.unsqueeze(-1)


def build_input_full_pose(
    joints_deg: torch.Tensor,  # (B, 7)
    current_rot_9: torch.Tensor,  # (B, 9) flattened 3x3
    ee_delta_m: torch.Tensor,  # (B, 3)
    target_rot_9: torch.Tensor,  # (B, 9)
) -> torch.Tensor:
    """Build the (B, 22) input tensor for the full-pose model.

    Encoding choice:
        - current rot: keep as flattened 9-D (full geometric info, smooth)
        - target rot: pass DELTA in axis-angle (3-D) so absolute orientation
          frame is local to current pose, model doesn't need to learn the
          absolute frame from scratch.
    """
    B = joints_deg.shape[0]
    R_curr = current_rot_9.reshape(B, 3, 3)
    R_tgt = target_rot_9.reshape(B, 3, 3)
    R_delta = R_tgt @ R_curr.transpose(-1, -2)  # delta in world frame
    rot_delta_aa = rot_to_axis_angle(R_delta)  # (B, 3)
    return torch.cat([joints_deg, current_rot_9, ee_delta_m, rot_delta_aa], dim=-1)
