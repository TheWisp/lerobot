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

        if elapsed >= self._timestamps[-1]:
            self._exhausted = True
            frame = self._positions[-1]
        else:
            # Binary search would be more efficient for very long traj,
            # but for minutes-of-30fps (~thousands of frames) the linear
            # scan with a cached cursor is fine and stays simple.
            idx = self._locate_frame(elapsed)
            frame = self._positions[idx]
        return {j: float(p) for j, p in zip(self._joints, frame, strict=True)}

    def _locate_frame(self, elapsed: float) -> int:
        """Index of the latest frame whose timestamp <= elapsed."""
        # Maintain a cursor (`_cursor`) so the common forward-walking
        # case is O(1) amortized rather than O(N) per call.
        cursor = getattr(self, "_cursor", 0)
        n = len(self._timestamps)
        while cursor + 1 < n and self._timestamps[cursor + 1] <= elapsed:
            cursor += 1
        self._cursor = cursor
        return cursor

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
