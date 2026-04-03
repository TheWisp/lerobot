"""RLT training metrics collector.

Thread-safe metrics sink that _rlt_step writes to and the GUI reads via API.
Stores time-series for charting and current values for status display.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class RLTMetrics:
    """Thread-safe metrics store for RLT training visualization."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Time-series (append-only, bounded)
    episode_successes: list[bool] = field(default_factory=list)
    episode_timestamps: list[float] = field(default_factory=list)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    actor_deltas: list[float] = field(default_factory=list)
    q_values: list[float] = field(default_factory=list)
    step_timestamps: list[float] = field(default_factory=list)

    # Current values
    episode: int = 0
    step_count: int = 0
    buffer_size: int = 0
    total_updates: int = 0
    mode: str = "WARMUP"  # WARMUP or RL

    _MAX_SERIES_LEN: int = field(default=5000, repr=False)

    def record_step(
        self,
        step: int,
        delta: float,
        buffer_size: int,
        total_updates: int,
        mode: str,
        critic_loss: float | None = None,
        actor_loss: float | None = None,
        q_value: float | None = None,
    ) -> None:
        with self._lock:
            self.step_count = step
            self.buffer_size = buffer_size
            self.total_updates = total_updates
            self.mode = mode

            t = time.time()
            self.actor_deltas.append(delta)
            self.step_timestamps.append(t)

            if critic_loss is not None:
                self.critic_losses.append(critic_loss)
            if actor_loss is not None:
                self.actor_losses.append(actor_loss)
            if q_value is not None:
                self.q_values.append(q_value)

            # Bound series length
            for series in (
                self.actor_deltas, self.step_timestamps,
                self.critic_losses, self.actor_losses, self.q_values,
            ):
                if len(series) > self._MAX_SERIES_LEN:
                    del series[:len(series) - self._MAX_SERIES_LEN]

    def record_episode(self, episode: int, success: bool) -> None:
        with self._lock:
            self.episode = episode
            self.episode_successes.append(success)
            self.episode_timestamps.append(time.time())

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for the API."""
        with self._lock:
            successes = list(self.episode_successes)
            n = len(successes)
            recent_20 = successes[-20:] if successes else []
            success_rate = sum(recent_20) / len(recent_20) if recent_20 else 0.0

            return {
                "episode": self.episode,
                "step_count": self.step_count,
                "buffer_size": self.buffer_size,
                "total_updates": self.total_updates,
                "mode": self.mode,
                "success_rate": round(success_rate, 3),
                "total_successes": sum(successes),
                "total_episodes": n,
                # Time-series (last N points for charts)
                "series": {
                    "episode_successes": successes[-200:],
                    "critic_losses": self.critic_losses[-200:],
                    "actor_losses": self.actor_losses[-200:],
                    "actor_deltas": self.actor_deltas[-200:],
                    "q_values": self.q_values[-200:],
                },
            }


# Global singleton — written by s1_process (subprocess)
_global_metrics: RLTMetrics | None = None
# Path to metrics JSON — set by s1_process, read by GUI API
_metrics_path: str | None = None


def get_metrics() -> RLTMetrics:
    """Get or create the global metrics instance."""
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RLTMetrics()
    return _global_metrics


def set_metrics_path(path: str) -> None:
    """Set the file path for cross-process metrics sharing."""
    global _metrics_path
    _metrics_path = path


def save_metrics_to_file() -> None:
    """Write current metrics snapshot to JSON file (called by subprocess)."""
    import json
    if _global_metrics is None or _metrics_path is None:
        return
    try:
        snap = _global_metrics.snapshot()
        # Atomic write: write to tmp then rename
        tmp = _metrics_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        import os
        os.replace(tmp, _metrics_path)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Failed to save metrics: %s", e)


def load_metrics_from_file(path: str | None = None) -> dict:
    """Read metrics snapshot from JSON file (called by GUI API server)."""
    import json
    p = path or _metrics_path
    if not p:
        # Try default location
        import os
        for candidate in ["outputs/rlt_online/metrics.json"]:
            if os.path.exists(candidate):
                p = candidate
                break
    if not p:
        return {}
    try:
        with open(p) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def reset_metrics() -> None:
    """Reset metrics (new training session)."""
    global _global_metrics
    _global_metrics = RLTMetrics()
