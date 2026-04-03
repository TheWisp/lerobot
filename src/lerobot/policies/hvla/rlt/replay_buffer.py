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
        """Save buffer contents to disk."""
        with self._lock:
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
            }, path)

    def load(self, path: str) -> None:
        """Load buffer contents from disk. Capacity must match."""
        with self._lock:
            data = torch.load(path, weights_only=True)
            assert data["capacity"] == self.capacity, (
                f"Buffer capacity mismatch: saved={data['capacity']}, current={self.capacity}"
            )
            self.size = data["size"]
            self.ptr = data["ptr"]
            self._z_rl[:] = data["z_rl"]
            self._state[:] = data["state"]
            self._action[:] = data["action"]
            self._ref[:] = data["ref"]
            self._reward[:] = data["reward"]
            self._next_z_rl[:] = data["next_z_rl"]
            self._next_state[:] = data["next_state"]
            self._next_ref[:] = data["next_ref"]
            self._done[:] = data["done"]

    def __len__(self) -> int:
        with self._lock:
            return self.size
