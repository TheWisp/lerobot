#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Teleoperator implementation that replays a saved trajectory file.

The whole control loop runs unchanged — only the leader oracle is swapped
for a file reader. After the last frame, ``is_exhausted`` flips to True so
the calling loop can exit cleanly (one trajectory = one episode).
"""

import json
import logging
import time
from pathlib import Path

from lerobot.robots.safe_trajectory import validate_trajectory
from lerobot.types import RobotAction

from ..teleoperator import Teleoperator
from .configuration_trajectory_replay import TrajectoryReplayTeleopConfig

logger = logging.getLogger(__name__)


class TrajectoryReplayTeleop(Teleoperator):
    """Stand-in "leader" that emits frames from a recorded trajectory.

    Frame selection is by **elapsed wall time since first** ``get_action()``,
    not iteration count: if the calling loop overruns we skip frames, if
    the loop underruns we repeat a frame. Either way the trajectory always
    finishes in the same wall time as it was recorded.

    Exhaustion (``is_exhausted == True``) is reached when elapsed time
    exceeds the last recorded timestamp. The calling loop is expected to
    check ``is_exhausted`` and exit.
    """

    config_class = TrajectoryReplayTeleopConfig
    name = "trajectory_replay"

    def __init__(self, config: TrajectoryReplayTeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._trajectory: dict | None = None
        self._timestamps: list[float] = []
        self._positions: list[list[float]] = []
        self._joints: list[str] = []
        self._start_t: float | None = None
        self._exhausted: bool = False

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(self._joints, float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._trajectory is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_exhausted(self) -> bool:
        """True once the recorded duration has elapsed since first
        ``get_action()``. The loop driver should break when this flips."""
        return self._exhausted

    @property
    def duration_s(self) -> float:
        return self._timestamps[-1] if self._timestamps else 0.0

    @property
    def frame_count(self) -> int:
        return len(self._timestamps)

    @property
    def start_pose(self) -> dict[str, float]:
        """First-frame joint positions, suitable for ``move_to_rest_position``.

        Recording's invariant is that frame 0 IS the rest_position of the
        robot — recording always starts after a move-to-rest. Callers can
        use this for an automated reset between episodes (e.g. multi-
        episode dataset recording) without needing the profile's
        rest_position dict separately.

        Raises ``RuntimeError`` if called before ``connect()``.
        """
        if self._trajectory is None:
            raise RuntimeError("start_pose accessed before connect()")
        return {j: float(p) for j, p in zip(self._joints, self._positions[0], strict=True)}

    def connect(self, calibrate: bool = True) -> None:
        if not self.config.trajectory_path:
            raise ValueError("trajectory_path is empty; pass --teleop.trajectory_path=...")
        path = Path(self.config.trajectory_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        traj = json.loads(path.read_text())
        validate_trajectory(traj)
        if not traj["timestamps"]:
            raise ValueError(f"Trajectory at {path} is empty")
        self._trajectory = traj
        self._timestamps = list(traj["timestamps"])
        self._positions = list(traj["positions"])
        self._joints = list(traj["joints"])
        self._start_t = None
        self._exhausted = False
        logger.info(
            "TrajectoryReplayTeleop loaded %s: %d frames over %.1fs (recorded at %d fps)",
            path,
            len(self._timestamps),
            self.duration_s,
            traj.get("fps", 0),
        )

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if self._trajectory is None:
            raise RuntimeError("TrajectoryReplayTeleop.get_action() called before connect()")
        now = time.perf_counter()
        if self._start_t is None:
            self._start_t = now
        elapsed = now - self._start_t

        # Exhaustion is gated on wall time (elapsed), not the lookahead-
        # shifted query time — the loop exits when the trajectory's wall
        # duration is up, regardless of how far ahead we were peeking.
        # When exhausted, clamp to the last recorded frame (which is the
        # rest pose by recording invariant) rather than extrapolating off
        # the end of the trajectory.
        if elapsed >= self._timestamps[-1]:
            self._exhausted = True
            frame = self._positions[-1]
        else:
            frame = self._action_at(elapsed)
        return {j: float(p) for j, p in zip(self._joints, frame, strict=True)}

    def _action_at(self, elapsed: float) -> list[float]:
        """Resolve the action vector for a given elapsed wall-time.

        Handles three cases:

        1. Full visibility (``simulate_chunk_size is None``): linear
           interpolation between bracketing trajectory frames, with the
           query shifted by ``lookahead_s``.
        2. Chunk simulation, query inside the active chunk: same linear
           interpolation but only over frames the current chunk exposes.
        3. Chunk simulation, query past chunk end: linear extrapolation
           from the chunk's own last-two-frame forward difference —
           modelling what a chunked policy must do when its lookahead
           outruns the chunk it currently holds.
        """
        query_t = elapsed + self.config.lookahead_s
        n = len(self._timestamps)

        # The chunk window is determined by *elapsed* (current wall time),
        # not by query_t — chunk arrival is a real-time event, the
        # lookahead just shifts where we look INSIDE / PAST the chunk.
        if self.config.simulate_chunk_size and self.config.simulate_chunk_size > 0:
            size = self.config.simulate_chunk_size
            current_idx = self._locate_frame(min(elapsed, self._timestamps[-1]))
            chunk_start = (current_idx // size) * size
            chunk_end = min(chunk_start + size - 1, n - 1)
        else:
            chunk_start = 0
            chunk_end = n - 1

        chunk_end_t = self._timestamps[chunk_end]

        # Clamp left edge (before trajectory starts).
        if query_t <= self._timestamps[chunk_start]:
            return list(self._positions[chunk_start])

        # Interpolation branch: query lands inside the chunk's visible window.
        if query_t <= chunk_end_t:
            idx = self._locate_frame_in_window(query_t, chunk_start, chunk_end)
            if idx + 1 <= chunk_end:
                t0 = self._timestamps[idx]
                t1 = self._timestamps[idx + 1]
                dt = t1 - t0
                alpha = (query_t - t0) / dt if dt > 1e-9 else 0.0
                p0 = self._positions[idx]
                p1 = self._positions[idx + 1]
                return [a + (b - a) * alpha for a, b in zip(p0, p1, strict=True)]
            return list(self._positions[idx])

        # Extrapolation branch: query is past the visible chunk. Use the
        # forward difference of the last two chunk frames as a velocity
        # estimate — same information a real chunked policy would have.
        if chunk_end == chunk_start:
            # Pathological 1-frame chunk: no velocity available, hold.
            return list(self._positions[chunk_end])
        t_prev = self._timestamps[chunk_end - 1]
        t_last = self._timestamps[chunk_end]
        dt = t_last - t_prev
        p_prev = self._positions[chunk_end - 1]
        p_last = self._positions[chunk_end]
        if dt < 1e-9:
            return list(p_last)
        ahead = query_t - t_last
        return [last + (last - prev) * (ahead / dt) for prev, last in zip(p_prev, p_last, strict=True)]

    def _locate_frame(self, elapsed: float) -> int:
        """Index of the latest frame whose timestamp <= elapsed."""
        # Maintain a cursor (`_cursor`) so the common forward-walking
        # case is O(1) amortized rather than O(N) per call.
        cursor = getattr(self, "_cursor", 0)
        n = len(self._timestamps)
        # Reset cursor if it's stale (e.g., elapsed went backwards in tests).
        if cursor >= n or self._timestamps[cursor] > elapsed:
            cursor = 0
        while cursor + 1 < n and self._timestamps[cursor + 1] <= elapsed:
            cursor += 1
        self._cursor = cursor
        return cursor

    def _locate_frame_in_window(self, t: float, lo: int, hi: int) -> int:
        """Latest index in ``[lo, hi]`` whose timestamp <= t.

        Linear scan from ``lo`` — chunks are small (typically <20 frames),
        so a binary search isn't worth the complexity.
        """
        idx = lo
        while idx + 1 <= hi and self._timestamps[idx + 1] <= t:
            idx += 1
        return idx

    def send_feedback(self, feedback: dict) -> None:
        # No physical leader — feedback is a no-op.
        pass

    def disconnect(self) -> None:
        self._trajectory = None
        self._timestamps = []
        self._positions = []
        self._joints = []
        self._start_t = None
        self._exhausted = False
        if hasattr(self, "_cursor"):
            delattr(self, "_cursor")
