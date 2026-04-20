"""RLT training metrics collector.

Thread-safe metrics sink written by inference thread and s1_process,
read by GUI API via file. Stores time-series for charting and current
values for status display.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RLTMetrics:
    """Thread-safe metrics store for RLT training visualization."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Per-episode records
    episode_successes: list[bool] = field(default_factory=list)
    episode_autonomous: list[bool] = field(default_factory=list)  # True = no intervention
    episode_timestamps: list[float] = field(default_factory=list)
    episode_lengths_s: list[float] = field(default_factory=list)  # seconds

    # Time-series (append-only, bounded)
    critic_losses: list[float] = field(default_factory=list)
    actor_losses: list[float] = field(default_factory=list)
    actor_deltas: list[float] = field(default_factory=list)
    q_values_mean: list[float] = field(default_factory=list)
    q_values_min: list[float] = field(default_factory=list)
    q_values_max: list[float] = field(default_factory=list)
    # Actor loss decomposition: actor_loss = q_term + bc_term.
    # q_term = -Q.mean() (sign depends on critic values).
    # bc_term = β · ||a - ref||² (always ≥ 0, pulls actor toward S1 reference).
    actor_q_terms: list[float] = field(default_factory=list)
    actor_bc_terms: list[float] = field(default_factory=list)
    step_timestamps: list[float] = field(default_factory=list)

    # Current values
    episode: int = 0
    step_count: int = 0
    buffer_size: int = 0
    total_updates: int = 0
    mode: str = "WARMUP"

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
        q_mean: float | None = None,
        q_min: float | None = None,
        q_max: float | None = None,
        actor_q_term: float | None = None,
        actor_bc_term: float | None = None,
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
            if q_mean is not None:
                self.q_values_mean.append(q_mean)
            if q_min is not None:
                self.q_values_min.append(q_min)
            if q_max is not None:
                self.q_values_max.append(q_max)
            if actor_q_term is not None:
                self.actor_q_terms.append(actor_q_term)
            if actor_bc_term is not None:
                self.actor_bc_terms.append(actor_bc_term)

            # Bound series length
            for series in (
                self.actor_deltas, self.step_timestamps,
                self.critic_losses, self.actor_losses,
                self.q_values_mean, self.q_values_min, self.q_values_max,
                self.actor_q_terms, self.actor_bc_terms,
            ):
                if len(series) > self._MAX_SERIES_LEN:
                    del series[:len(series) - self._MAX_SERIES_LEN]

    def record_episode(
        self,
        episode: int,
        success: bool,
        autonomous: bool,
        duration_s: float,
    ) -> None:
        with self._lock:
            self.episode = episode
            self.episode_successes.append(success)
            self.episode_autonomous.append(autonomous)
            self.episode_timestamps.append(time.time())
            self.episode_lengths_s.append(duration_s)

    @staticmethod
    def _smooth(data: list, window: int = 20) -> list:
        """Rolling mean for smoother charts."""
        if len(data) <= window:
            return data
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(sum(data[start:i + 1]) / (i - start + 1))
        return result

    def snapshot(self) -> dict:
        """Return a JSON-serializable snapshot for the API."""
        with self._lock:
            successes = list(self.episode_successes)
            autonomous = list(self.episode_autonomous)
            timestamps = list(self.episode_timestamps)
            lengths = list(self.episode_lengths_s)
            n = len(successes)

            # Total success rate (rolling 20)
            recent_s = successes[-20:]
            success_rate = sum(recent_s) / len(recent_s) if recent_s else 0.0

            # Autonomous success rate (rolling 20, out of ALL episodes)
            # Autonomous = succeeded WITHOUT intervention. Assisted = not autonomous.
            recent_auto = [(s, a) for s, a in zip(successes[-20:], autonomous[-20:])]
            auto_successes = sum(1 for s, a in recent_auto if s and a)
            autonomous_rate = auto_successes / len(recent_auto) if recent_auto else 0.0

            # Intervention rate (rolling 20)
            recent_a = autonomous[-20:]
            intervention_rate = 1.0 - (sum(recent_a) / len(recent_a)) if recent_a else 0.0

            # Throughput: autonomous successes per 10 min of active time
            # Only count time from episodes, not reset phases
            now = time.time()
            window_start = now - 600  # last 10 minutes
            throughput = 0
            for s, a, t, d in zip(successes, autonomous, timestamps, lengths):
                if t >= window_start and s and a:
                    throughput += 1

            # Pre-compute rolling autonomous rate for the chart
            # Same formula as autonomous_rate above, at every episode point
            auto_rate_rolling = []
            for i in range(n):
                win_s = successes[max(0, i - 19):i + 1]
                win_a = autonomous[max(0, i - 19):i + 1]
                auto_s = sum(1 for s, a in zip(win_s, win_a) if s and a)
                auto_rate_rolling.append(auto_s / len(win_s) if win_s else 0.0)

            return {
                "episode": self.episode,
                "step_count": self.step_count,
                "buffer_size": self.buffer_size,
                "total_updates": self.total_updates,
                "mode": self.mode,
                "success_rate": round(success_rate, 3),
                "autonomous_rate": round(autonomous_rate, 3),
                "intervention_rate": round(intervention_rate, 3),
                "throughput_10min": throughput,
                "total_successes": sum(successes),
                "total_autonomous": sum(1 for a in autonomous if a),
                "total_episodes": n,
                "series": {
                    # Per-episode records (needed for resume)
                    "episode_successes": successes[-200:],
                    "episode_autonomous": autonomous[-200:],
                    # Pre-computed rolling series (for chart, single source of truth)
                    "autonomous_rate_rolling": auto_rate_rolling[-200:],
                    # Training metrics (smoothed with rolling window for readability)
                    "critic_losses": self._smooth(self.critic_losses[-5000:], 20),
                    "actor_losses": self._smooth(self.actor_losses[-5000:], 20),
                    "actor_deltas": self._smooth(self.actor_deltas[-5000:], 20),
                    "q_values_mean": self.q_values_mean[-5000:],
                    "q_values_min": self.q_values_min[-5000:],
                    "q_values_max": self.q_values_max[-5000:],
                    "actor_q_terms": self._smooth(self.actor_q_terms[-5000:], 20),
                    "actor_bc_terms": self._smooth(self.actor_bc_terms[-5000:], 20),
                    # Wall-clock timestamps aligned with per-step series (same index).
                    "step_timestamps": self.step_timestamps[-5000:],
                },
            }


# Global singleton — written by s1_process (subprocess)
_global_metrics: RLTMetrics | None = None
_metrics_path: str | None = None


def get_metrics() -> RLTMetrics:
    global _global_metrics
    if _global_metrics is None:
        _global_metrics = RLTMetrics()
    return _global_metrics


def set_metrics_path(path: str) -> None:
    global _metrics_path
    _metrics_path = path


def save_metrics_to_file() -> None:
    import json
    if _global_metrics is None or _metrics_path is None:
        return
    try:
        snap = _global_metrics.snapshot()
        tmp = _metrics_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(snap, f)
        import os
        os.replace(tmp, _metrics_path)
    except Exception as e:
        logger.warning("Failed to save metrics: %s", e)


def load_metrics_from_file(path: str | None = None) -> dict:
    import json
    p = path or _metrics_path
    if not p:
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
    global _global_metrics
    _global_metrics = RLTMetrics()
