"""Lightweight actor-critic MLPs for RLT online RL.

Actor: Gaussian policy over action chunks, conditioned on RL token + state +
       reference action chunk from frozen S1.
Critic: TD3 ensemble of Q-functions over action chunks.

Architecture from Xu et al. 2026, Appendix B.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from torch import Tensor

from lerobot.policies.hvla.rlt.config import RLTConfig


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> nn.Sequential:
    """Build a ReLU MLP with the given number of hidden layers."""
    layers: list[nn.Module] = []
    in_d = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(in_d, hidden_dim))
        layers.append(nn.ReLU())
        in_d = hidden_dim
    layers.append(nn.Linear(in_d, output_dim))
    return nn.Sequential(*layers)


class RLTActor(nn.Module):
    """Gaussian actor over action chunks.

    pi_theta(a_{1:C} | x, a_ref) = N(mu(x, a_ref), sigma^2 I)

    Input: cat(z_rl, s_p, a_ref_flat) where a_ref may be zeroed (dropout).
    Output: action chunk mean [C * action_dim].
    """

    def __init__(self, config: RLTConfig, state_dim: int, action_dim: int):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.chunk_length = config.rl_chunk_length

        input_dim = config.rl_token_dim + state_dim + config.rl_chunk_length * action_dim
        output_dim = config.rl_chunk_length * action_dim

        self.mlp = _build_mlp(input_dim, config.actor_hidden_dim, output_dim, config.actor_num_layers)

        # Zero-init output layer so mu ≈ 0 initially.
        # Combined with BC regularizer (beta * ||a - ref||^2), the actor
        # starts outputting near-reference actions and deviates only as
        # the critic provides meaningful gradients.
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()

    def forward(
        self,
        z_rl: Tensor,
        state: Tensor,
        ref_chunk: Tensor,
        deterministic: bool = False,
    ) -> Tensor:
        """
        Args:
            z_rl: [B, D] RL token
            state: [B, state_dim] proprioceptive state
            ref_chunk: [B, C, action_dim] reference chunk from frozen S1
        Returns:
            actions: [B, C, action_dim] — direct output (paper Eq. 4)
        """
        B = z_rl.shape[0]
        ref_flat = ref_chunk.reshape(B, -1)                    # [B, C*A]
        x = torch.cat([z_rl, state, ref_flat], dim=-1)         # [B, D+S+C*A]
        mu = self.mlp(x)                                       # [B, C*A]
        mu = mu.reshape(B, self.chunk_length, self.action_dim) # [B, C, A]

        if deterministic:
            return mu

        noise = torch.randn_like(mu) * self.config.actor_sigma
        return mu + noise

    def mean(self, z_rl: Tensor, state: Tensor, ref_chunk: Tensor) -> Tensor:
        """Return deterministic mean (no exploration noise)."""
        return self.forward(z_rl, state, ref_chunk, deterministic=True)


class RLTCritic(nn.Module):
    """TD3 ensemble of Q-functions.

    Q_psi(x, a_{1:C}) -> R   for each Q in ensemble.
    """

    def __init__(self, config: RLTConfig, state_dim: int, action_dim: int):
        super().__init__()
        input_dim = config.rl_token_dim + state_dim + config.rl_chunk_length * action_dim

        self.q_nets = nn.ModuleList([
            _build_mlp(input_dim, config.critic_hidden_dim, 1, config.critic_num_layers)
            for _ in range(config.num_critics)
        ])

    def forward(self, z_rl: Tensor, state: Tensor, action_chunk: Tensor) -> list[Tensor]:
        """
        Args:
            z_rl: [B, D]
            state: [B, state_dim]
            action_chunk: [B, C, action_dim]
        Returns:
            list of [B, 1] Q-values, one per ensemble member
        """
        B = z_rl.shape[0]
        a_flat = action_chunk.reshape(B, -1)
        x = torch.cat([z_rl, state, a_flat], dim=-1)
        return [q(x) for q in self.q_nets]

    def min_q(self, z_rl: Tensor, state: Tensor, action_chunk: Tensor) -> Tensor:
        """Pessimistic Q estimate: min over ensemble."""
        qs = self.forward(z_rl, state, action_chunk)
        return torch.min(torch.cat(qs, dim=-1), dim=-1, keepdim=True).values


class TD3Agent:
    """Manages actor, critic, target networks, and update logic."""

    def __init__(self, config: RLTConfig, state_dim: int, action_dim: int, device: torch.device):
        self.config = config
        self.device = device

        self.actor = RLTActor(config, state_dim, action_dim).to(device)
        self.critic = RLTCritic(config, state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    @torch.no_grad()
    def soft_update_target(self):
        tau = self.config.tau
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.lerp_(p.data, tau)

    def update_critic(
        self,
        z_rl: Tensor,
        state: Tensor,
        action_chunk: Tensor,
        ref_chunk: Tensor,
        reward: Tensor,
        next_z_rl: Tensor,
        next_state: Tensor,
        next_ref_chunk: Tensor,
        done: Tensor,
    ) -> tuple[float, float]:
        """Single critic gradient step.

        Returns:
            (critic_loss, grad_norm) — grad_norm is the global L2 norm
            measured BEFORE clipping, so monitoring shows the true step
            magnitude. Values persistently near ``critic_grad_clip`` mean
            the clip is triggering often (defensive); values far below
            it mean the clip is a pure safety net and isn't constraining
            normal learning.
        """
        C = self.config.rl_chunk_length
        gamma = self.config.discount

        with torch.no_grad():
            # Target actions from actor (with clipped noise for TD3)
            next_action = self.actor(next_z_rl, next_state, next_ref_chunk, deterministic=False)
            target_q = self.critic_target.min_q(next_z_rl, next_state, next_action)
            # C-step return: reward is already the C-step cumulative
            q_target = reward + (gamma ** C) * (1 - done) * target_q

        qs = self.critic.forward(z_rl, state, action_chunk)
        critic_loss = sum(torch.nn.functional.mse_loss(q, q_target) for q in qs)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        # Clip gradient norm to bound update size per step — primary defense
        # against Q-bootstrap explosions (see Ville Kuosmanen's RLT reproduction
        # notes, and docs/rlt_design.md section on TD3 defenses). The returned
        # value is the PRE-clip norm, for monitoring.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.critic_grad_clip,
        ).item()
        self.critic_opt.step()
        self.soft_update_target()

        return critic_loss.item(), grad_norm

    def update_actor(
        self,
        z_rl: Tensor,
        state: Tensor,
        ref_chunk: Tensor,
    ) -> tuple[float, float, float]:
        """Single actor gradient step.

        Returns:
            (actor_loss, q_term, bc_term) where:
              q_term = -Q.mean() (negative = Q gradient pushing actor to high-Q actions)
              bc_term = β * ||a - ref||² (positive = BC penalty pulling toward ref)
              actor_loss = q_term + bc_term
            The ratio |q_term|/bc_term tells you which force dominates.
        """
        beta = self.config.beta

        # Reference action dropout: zero out ref_chunk with probability p
        if self.config.ref_action_dropout > 0:
            mask = (torch.rand(z_rl.shape[0], 1, 1, device=z_rl.device)
                    > self.config.ref_action_dropout).float()
            ref_input = ref_chunk * mask
        else:
            ref_input = ref_chunk

        action = self.actor.mean(z_rl, state, ref_input)
        q_val = self.critic.min_q(z_rl, state, action)
        # Paper Eq. 5: β||a - ã||²_2 — squared L2 norm (sum, not mean).
        # Sum over action dims (C×A), average over batch.
        bc_penalty = ((action - ref_chunk) ** 2).sum(dim=(-1, -2)).mean()
        q_term = -q_val.mean()
        bc_term = beta * bc_penalty
        actor_loss = q_term + bc_term

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        return actor_loss.item(), q_term.item(), bc_term.item()
