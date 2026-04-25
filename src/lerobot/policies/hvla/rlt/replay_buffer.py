"""Chunk-level replay buffer for RLT online RL.

Stores (z_rl, state, action_chunk, ref_chunk, reward, next_z_rl, next_state,
next_ref_chunk, done) tuples. z_rl is pre-computed and stored so replay
doesn't require re-running the frozen S1 encoder or RL token encoder.
"""

from __future__ import annotations

import threading

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer for chunk-level transitions.

    Thread-safe: all public methods are guarded by a lock.
    Inference thread writes during RL mode, main thread writes during
    intervention — concurrent access is possible.
    """

    def __init__(
        self,
        capacity: int,
        rl_token_dim: int,
        state_dim: int,
        action_dim: int,
        chunk_length: int,
        device: torch.device,
    ):
        self._lock = threading.Lock()
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

        A = chunk_length * action_dim
        # Store everything on CPU, move to device on sample
        self._z_rl = torch.zeros(capacity, rl_token_dim)
        self._state = torch.zeros(capacity, state_dim)
        self._action = torch.zeros(capacity, A)
        self._ref = torch.zeros(capacity, A)
        self._reward = torch.zeros(capacity, 1)
        self._next_z_rl = torch.zeros(capacity, rl_token_dim)
        self._next_state = torch.zeros(capacity, state_dim)
        self._next_ref = torch.zeros(capacity, A)
        self._done = torch.zeros(capacity, 1)

    def add(
        self,
        z_rl: Tensor,
        state: Tensor,
        action_chunk: Tensor,
        ref_chunk: Tensor,
        reward: float,
        next_z_rl: Tensor,
        next_state: Tensor,
        next_ref_chunk: Tensor,
        done: bool,
    ) -> None:
        """Add a single transition (all tensors should be 1D/2D, no batch dim)."""
        with self._lock:
            i = self.ptr
            self._z_rl[i] = z_rl.detach().cpu()
            self._state[i] = state.detach().cpu()
            self._action[i] = action_chunk.detach().cpu().reshape(-1)
            self._ref[i] = ref_chunk.detach().cpu().reshape(-1)
            self._reward[i, 0] = reward
            self._next_z_rl[i] = next_z_rl.detach().cpu()
            self._next_state[i] = next_state.detach().cpu()
            self._next_ref[i] = next_ref_chunk.detach().cpu().reshape(-1)
            self._done[i, 0] = float(done)

            self.ptr = (self.ptr + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, Tensor]:
        """Sample a random batch and move to device."""
        with self._lock:
            idx = torch.randint(0, self.size, (batch_size,))
            return {
                "z_rl": self._z_rl[idx].to(self.device),
                "state": self._state[idx].to(self.device),
                "action": self._action[idx].to(self.device),
                "ref": self._ref[idx].to(self.device),
                "reward": self._reward[idx].to(self.device),
                "next_z_rl": self._next_z_rl[idx].to(self.device),
                "next_state": self._next_state[idx].to(self.device),
                "next_ref": self._next_ref[idx].to(self.device),
                "done": self._done[idx].to(self.device),
            }

    def save(self, path: str) -> None:
        """Save buffer contents to disk. Atomic via tmp+rename so a
        crash mid-write can't leave a torn file at ``path``."""
        import os
        with self._lock:
            tmp = path + ".tmp"
            torch.save({
                "size": self.size,
                "ptr": self.ptr,
                "capacity": self.capacity,
                "z_rl": self._z_rl[:self.capacity],
                "state": self._state[:self.capacity],
                "action": self._action[:self.capacity],
                "ref": self._ref[:self.capacity],
                "reward": self._reward[:self.capacity],
                "next_z_rl": self._next_z_rl[:self.capacity],
                "next_state": self._next_state[:self.capacity],
                "next_ref": self._next_ref[:self.capacity],
                "done": self._done[:self.capacity],
            }, tmp)
            os.replace(tmp, path)

    def load(self, path: str) -> None:
        """Load buffer contents from disk. Supports same or larger capacity."""
        import logging
        with self._lock:
            data = torch.load(path, weights_only=True)
            saved_cap = data["capacity"]
            saved_size = data["size"]

            if saved_cap > self.capacity:
                raise ValueError(
                    f"Cannot load buffer: saved capacity {saved_cap} > current {self.capacity}. "
                    f"Increase replay_capacity to at least {saved_cap}."
                )

            n = saved_size
            self._z_rl[:n] = data["z_rl"][:n]
            self._state[:n] = data["state"][:n]
            self._action[:n] = data["action"][:n]
            self._ref[:n] = data["ref"][:n]
            self._reward[:n] = data["reward"][:n]
            self._next_z_rl[:n] = data["next_z_rl"][:n]
            self._next_state[:n] = data["next_state"][:n]
            self._next_ref[:n] = data["next_ref"][:n]
            self._done[:n] = data["done"][:n]
            self.size = n
            self.ptr = n % self.capacity

            if saved_cap < self.capacity:
                logging.getLogger(__name__).info(
                    "Replay buffer expanded: %d → %d capacity, %d transitions preserved",
                    saved_cap, self.capacity, n,
                )

    def __len__(self) -> int:
        with self._lock:
            return self.size
