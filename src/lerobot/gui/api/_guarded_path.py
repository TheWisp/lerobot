"""Observed-arrival execution of a preplanned joint segment; no hardware access.

This checks timing, joint/workspace bounds and following error, not collisions.
The caller must plan/check the arm and payload against its current scene first.
Only the jog's hardware loop may tick this state machine, with a fresh checked read.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

LEASE_S = 1.0
ARRIVAL_DEG = 0.7
ARRIVAL_M = 0.002
ARRIVAL_ANGLE_DEG = 2.0
MAX_ERROR_DEG = 3.0
MAX_ERROR_M = 0.007
ARRIVAL_TIMEOUT_S = 2.0
JOINT_DEG_S = 5.0
GRIP_UNITS_S = 10.0
LINEAR_M_S = 0.020
ANGULAR_DEG_S = 20.0
MAX_SAMPLES = 6000
# The Feetech temperature register occasionally returns one implausibly high
# value in an otherwise valid packet. A real hot motor remains hot on the next
# reads; its firmware overheat fault is checked separately by the jog loop.
HOT_READS_TO_STOP = 3


def residual(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    cosine = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1.0) / 2.0
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])), float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


@dataclass
class GuardedPath:
    identity: str
    names: tuple[str, ...]
    q: np.ndarray
    poses: np.ndarray
    intervals: np.ndarray
    grip_index: int
    grip_contact: bool
    started: float
    heartbeat_at: float
    sent_at: float
    index: int = 0
    status: str = "running"
    reason: str = ""
    error_mm: float = 0.0
    error_deg: float = 0.0
    kin: Any = None  # private FK data, never shared with the jog/GUI's mutable Pinocchio state
    stream_samples: bool = False  # collision samples may be streamed; phase endpoints must still arrive
    high_temp_streak: dict[str, int] = field(default_factory=dict)

    @classmethod
    def prepare(
        cls,
        identity: str,
        names: tuple[str, ...],
        samples: Any,
        expected_poses: Any,
        kin: Any,
        ranges: tuple[np.ndarray, np.ndarray],
        workspace: tuple,
        grip_index: int,
        grip_contact: bool,
        now: float,
        *,
        stream_samples: bool = False,
    ) -> "GuardedPath":
        q = np.asarray(samples, dtype=float)
        expected = np.asarray(expected_poses, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(names) or not 2 <= len(q) <= MAX_SAMPLES:
            raise ValueError(f"path needs 2..{MAX_SAMPLES} complete joint samples")
        if not np.isfinite(q).all() or expected.shape != (len(q), 4, 4) or not np.isfinite(expected).all():
            raise ValueError("finite joints and one finite 4x4 tip pose per sample required")
        lo, hi = (np.array(v, dtype=float, copy=True) for v in ranges)
        lo[grip_index], hi[grip_index] = 0.0, 100.0
        if not np.isfinite([lo, hi]).all() or np.any(q < lo) or np.any(q > hi):
            raise ValueError("path exceeds calibrated servo ranges")
        arm = np.arange(len(names)) != grip_index
        delta = np.abs(np.diff(q, axis=0))
        if np.any(delta[:, arm] > 0.500001) or np.any(delta[:, grip_index] > 1.000001):
            raise ValueError("path samples must be at most 0.5 arm degrees / 1 gripper unit apart")
        moving_arm = bool(np.any(delta[:, arm] > 1e-8))
        moving_grip = bool(np.any(delta[:, grip_index] > 1e-8))
        if moving_arm and moving_grip:
            raise ValueError("arm transit and gripper changes must be separate segments")
        if grip_contact and (moving_arm or np.any(np.diff(q[:, grip_index]) < -1e-8)):
            raise ValueError("gripper contact is only allowed during a stationary closing segment")
        poses = np.stack([kin.forward_kinematics(row) for row in q])
        if not np.isfinite(poses).all() or not np.allclose(poses, expected, atol=1e-6, rtol=0):
            raise ValueError("planner tip poses do not match the connected arm's calibration")
        if np.any(poses[:, :3, 3] < workspace[0]) or np.any(poses[:, :3, 3] > workspace[1]):
            raise ValueError("path leaves the calibrated fingertip workspace")
        intervals = [0.0]
        for k, d in enumerate(delta, start=1):
            metres, degrees = residual(poses[k - 1], poses[k])
            if metres > 0.002001 or degrees > 2.000001:
                raise ValueError("path tip samples must be at most 2 mm / 2 degrees apart")
            intervals.append(
                max(
                    1.0 / 30.0,
                    float(d[arm].max()) / JOINT_DEG_S,
                    float(d[grip_index]) / GRIP_UNITS_S,
                    metres / LINEAR_M_S,
                    degrees / ANGULAR_DEG_S,
                )
            )
        if sum(intervals) > 180:
            raise ValueError("path segment exceeds three minutes; divide it into checked phases")
        path = cls(
            identity, names, q.copy(), poses, np.asarray(intervals), grip_index, grip_contact, now, now, now
        )
        path.kin = kin
        path.stream_samples = stream_samples
        return path

    def check_start(self, observed: dict[str, float], commanded: dict[str, float], kin: Any) -> None:
        q_now = np.array([observed[n] for n in self.names])
        if not np.isfinite(q_now).all() or not np.isfinite(commanded[self.names[self.grip_index]]):
            raise ValueError("invalid start telemetry")
        arm = np.arange(len(self.names)) != self.grip_index
        distance, _ = residual(kin.forward_kinematics(q_now), self.poses[0])
        if np.max(np.abs(q_now[arm] - self.q[0, arm])) > 0.25 or distance > 0.001:
            raise ValueError("arm moved since planning; first sample must match observed joints")
        if abs(commanded[self.names[self.grip_index]] - self.q[0, self.grip_index]) > 0.25:
            raise ValueError("first sample must keep the current gripper command")

    def fail(self, reason: str) -> None:
        self.status, self.reason = "failed", reason

    def tick(self, now: float, observed: dict, pose: np.ndarray, load: dict, temps: dict) -> dict | None:
        """Send every sample in order; always require observed arrival at the phase endpoint.

        Streaming is explicit: fine collision samples are not physical stops. A
        stiction-bound joint may need more than one sample of lead to move. Even
        then, the NEXT command must remain inside the following-error bounds.
        Gripper operations retain their per-sample contact/arrival behavior.
        """
        if self.status == "failed":
            return None
        if self.status == "running" and now - self.heartbeat_at > LEASE_S:
            self.fail("path client heartbeat expired")
            return None
        try:
            q_now = np.array([observed[n] for n in self.names], float)
            temperatures = np.array([temps[n] for n in self.names], float)
            loads = np.array([load[n] for n in self.names], float)
        except (KeyError, TypeError, ValueError):
            self.fail("incomplete path telemetry")
            return None
        if not np.isfinite([q_now, temperatures, loads]).all() or np.any(temperatures <= 0):
            self.fail("invalid path telemetry")
            return None
        for name, temperature in zip(self.names, temperatures, strict=True):
            self.high_temp_streak[name] = self.high_temp_streak.get(name, 0) + 1 if temperature > 50 else 0
        if any(count >= HOT_READS_TO_STOP for count in self.high_temp_streak.values()):
            self.fail("motor temperature above 50 C; stopped for recheck")
            return None
        arm = np.arange(len(self.names)) != self.grip_index
        if np.max(np.abs(loads[arm])) >= 700:
            self.fail("arm motor load reached 70 percent")
            return None
        error, angle = residual(pose, self.poses[self.index])
        lag = float(np.max(np.abs(q_now[arm] - self.q[self.index, arm])))
        self.error_mm, self.error_deg = error * 1000, lag
        if not np.isfinite([error, angle, lag]).all() or error > MAX_ERROR_M or lag > MAX_ERROR_DEG:
            self.fail(f"path following error: {error * 1000:.1f} mm, {lag:.1f} joint degrees")
            return None
        grip_moves = bool(np.any(np.diff(self.q[:, self.grip_index])))
        grip_arrived = (
            not grip_moves
            or self.grip_contact
            or abs(q_now[self.grip_index] - self.q[self.index, self.grip_index]) <= 1.0
        )
        arrived = lag <= ARRIVAL_DEG and error <= ARRIVAL_M and angle <= ARRIVAL_ANGLE_DEG and grip_arrived
        last = self.index == len(self.q) - 1
        advance = arrived
        if self.stream_samples and not grip_moves and not last:
            next_error, _ = residual(pose, self.poses[self.index + 1])
            next_lag = float(np.max(np.abs(q_now[arm] - self.q[self.index + 1, arm])))
            advance = next_error <= MAX_ERROR_M and next_lag <= MAX_ERROR_DEG
        if not arrived:
            self.status = "running"
        if not (arrived if last else advance) and now - self.sent_at > ARRIVAL_TIMEOUT_S:
            self.fail(
                f"path sample {self.index} did not arrive: {error * 1000:.1f} mm, {lag:.1f} joint degrees"
            )
            return None
        if arrived and last:
            self.status = "done"
        elif not last and advance and now - self.sent_at >= self.intervals[self.index + 1]:
            self.index += 1
            self.sent_at = now
        return dict(zip(self.names, map(float, self.q[self.index]), strict=True))

    def info(self) -> dict:
        return {
            "id": self.identity,
            "status": self.status,
            "reason": self.reason,
            "sample": self.index,
            "samples": len(self.q),
            "error_mm": self.error_mm,
            "error_deg": self.error_deg,
            "stream_samples": self.stream_samples,
        }
