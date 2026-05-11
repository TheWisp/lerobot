"""Tests for the trajectory_replay teleoperator."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from lerobot.robots.safe_trajectory import SCHEMA_VERSION
from lerobot.teleoperators.trajectory_replay import (
    TrajectoryReplayTeleop,
    TrajectoryReplayTeleopConfig,
)


def _write_trajectory(tmp_path: Path, frames: list[tuple[float, list[float]]], joints: list[str]) -> Path:
    """Write a minimal valid trajectory JSON and return its path."""
    timestamps = [t for t, _ in frames]
    positions = [p for _, p in frames]
    traj = {
        "schema_version": SCHEMA_VERSION,
        "robot_type": "test",
        "fps": 30,
        "joints": joints,
        "timestamps": timestamps,
        "positions": positions,
    }
    path = tmp_path / "test.trajectory.json"
    path.write_text(json.dumps(traj))
    return path


def _make_teleop(
    tmp_path: Path,
    frames=None,
    joints=None,
    *,
    lookahead_s: float = 0.0,
    simulate_chunk_size: int | None = None,
) -> TrajectoryReplayTeleop:
    if frames is None:
        frames = [(0.0, [1.0, 2.0]), (0.1, [1.1, 2.1]), (0.2, [1.2, 2.2])]
    if joints is None:
        joints = ["a.pos", "b.pos"]
    path = _write_trajectory(tmp_path, frames, joints)
    cfg = TrajectoryReplayTeleopConfig(
        id="test",
        trajectory_path=str(path),
        lookahead_s=lookahead_s,
        simulate_chunk_size=simulate_chunk_size,
    )
    return TrajectoryReplayTeleop(cfg)


class TestConnect:
    def test_loads_valid_trajectory(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        assert t.is_connected
        assert t.frame_count == 3
        assert t.duration_s == pytest.approx(0.2)
        assert t.action_features == {"a.pos": float, "b.pos": float}

    def test_missing_path_rejected(self, tmp_path):
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path="")
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(ValueError, match="trajectory_path"):
            t.connect()

    def test_missing_file_rejected(self, tmp_path):
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path=str(tmp_path / "nope.json"))
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(FileNotFoundError):
            t.connect()

    def test_empty_trajectory_rejected(self, tmp_path):
        t = _make_teleop(tmp_path, frames=[])
        with pytest.raises(ValueError, match="empty"):
            t.connect()

    def test_malformed_trajectory_rejected(self, tmp_path):
        path = tmp_path / "bad.trajectory.json"
        path.write_text(
            json.dumps({"schema_version": 999, "fps": 30, "joints": [], "timestamps": [], "positions": []})
        )
        cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path=str(path))
        t = TrajectoryReplayTeleop(cfg)
        with pytest.raises(ValueError, match="schema_version"):
            t.connect()


class TestGetAction:
    def test_first_frame_at_t0(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter",
            return_value=100.0,
        ):
            a = t.get_action()
        assert a == {"a.pos": 1.0, "b.pos": 2.0}
        assert not t.is_exhausted

    def test_frame_interpolated_between_recorded_samples(self, tmp_path):
        """Action is linearly interpolated between bracketing trajectory frames.

        At a sample-time boundary the interpolation result equals the
        recorded frame exactly; between samples it returns the lerp,
        producing a smooth ramp instead of a stairstep when the loop
        runs faster than the trajectory's native fps.
        """
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()  # primes _start_t = 100.0
            # Elapsed 0.05s — alpha=0.5 between frame 0 [1.0, 2.0] and frame 1 [1.1, 2.1]
            mock_t.return_value = 100.05
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.05)
            assert a["b.pos"] == pytest.approx(2.05)
            # Elapsed 0.11s — alpha=0.1 between frame 1 and frame 2 [1.2, 2.2]
            mock_t.return_value = 100.11
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.11)
            assert a["b.pos"] == pytest.approx(2.11)
            # Elapsed 0.15s — alpha=0.5 between frame 1 and frame 2
            mock_t.return_value = 100.15
            a = t.get_action()
            assert a["a.pos"] == pytest.approx(1.15)
            assert a["b.pos"] == pytest.approx(2.15)

    def test_exhausted_after_last_frame(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.25  # past final timestamp 0.2
            a = t.get_action()
        assert t.is_exhausted
        # When exhausted, returns the last recorded frame (clamps).
        assert a == {"a.pos": 1.2, "b.pos": 2.2}

    def test_call_before_connect_raises(self, tmp_path):
        t = _make_teleop(tmp_path)
        with pytest.raises(RuntimeError, match="connect"):
            t.get_action()


class TestLookahead:
    def test_zero_lookahead_matches_baseline(self, tmp_path):
        """``lookahead_s = 0`` reproduces the no-lookahead interpolation."""
        t = _make_teleop(tmp_path, lookahead_s=0.0)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.05  # alpha=0.5 between frame 0 and 1
            a = t.get_action()
        assert a["a.pos"] == pytest.approx(1.05)

    def test_lookahead_shifts_query_forward(self, tmp_path):
        """With 50 ms lookahead, action at t=0 should equal trajectory at 0.05."""
        t = _make_teleop(tmp_path, lookahead_s=0.05)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            a = t.get_action()
        # frame 0 = [1.0, 2.0], frame 1 = [1.1, 2.1], elapsed+lookahead=0.05 → alpha=0.5
        assert a["a.pos"] == pytest.approx(1.05)
        assert a["b.pos"] == pytest.approx(2.05)


class TestChunkSimulation:
    def test_no_chunk_full_trajectory_visible(self, tmp_path):
        """``simulate_chunk_size = None`` exposes the whole trajectory."""
        # 5 frames at 0.1s spacing: positions 1.0..1.4
        frames = [(i * 0.1, [1.0 + i * 0.1]) for i in range(5)]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"])
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()  # primes start_t
            mock_t.return_value = 100.05  # interp between frame 0 and 1
            a = t.get_action()
        assert a["a.pos"] == pytest.approx(1.05)

    def test_chunk_interp_inside_window(self, tmp_path):
        """When the query lands inside the active chunk, interpolate normally."""
        frames = [(i * 0.1, [1.0 + i * 0.1]) for i in range(6)]
        # chunk_size=3 → chunk 0 = frames 0,1,2 (t in [0, 0.2])
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], simulate_chunk_size=3)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.15  # inside chunk 0 (frames 1 and 2)
            a = t.get_action()
        # alpha = 0.5 between frame 1 (1.1) and frame 2 (1.2)
        assert a["a.pos"] == pytest.approx(1.15)

    def test_chunk_extrapolation_past_window(self, tmp_path):
        """Query past chunk end extrapolates from chunk-end forward diff.

        Chunk 0 = frames [0, 1, 2] with positions [1.0, 1.1, 1.2] at times
        [0, 0.1, 0.2]. Velocity at chunk end (forward diff of last two):
        (1.2 - 1.1) / 0.1 = 1.0 unit/sec.

        Lookahead 0.05s past chunk_end (t=0.25) should give
        1.2 + 1.0 * 0.05 = 1.25 — and crucially NOT 1.25 from interpolating
        into chunk 1 (which it can't see).
        """
        # Chunk 1 (frames 3,4,5) has a *kink*: positions jump to 2.0+, so
        # honest extrapolation from chunk 0's velocity diverges from
        # ground truth — proving the source isn't cheating.
        frames = [
            (0.0, [1.0]),
            (0.1, [1.1]),
            (0.2, [1.2]),
            (0.3, [2.0]),  # discontinuity at chunk boundary
            (0.4, [2.5]),
            (0.5, [3.0]),
        ]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], simulate_chunk_size=3)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.25  # 0.05s past chunk 0's end
            a = t.get_action()
        # Extrapolating chunk 0: 1.2 + 1.0 * 0.05 = 1.25 (NOT peeking at 2.0).
        assert a["a.pos"] == pytest.approx(1.25)

    def test_chunk_advances_with_wall_time(self, tmp_path):
        """Once elapsed crosses a chunk boundary, the active chunk shifts."""
        frames = [
            (0.0, [1.0]),
            (0.1, [1.1]),
            (0.2, [1.2]),
            (0.3, [2.0]),  # chunk 1 begins
            (0.4, [2.5]),
            (0.5, [3.0]),
        ]
        t = _make_teleop(tmp_path, frames=frames, joints=["a.pos"], simulate_chunk_size=3)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            # Elapsed 0.35 is inside chunk 1 (frames 3,4,5), interpolate
            # between frame 3 (2.0) and 4 (2.5): alpha=0.5 → 2.25
            mock_t.return_value = 100.35
            a = t.get_action()
        assert a["a.pos"] == pytest.approx(2.25)

    def test_lookahead_plus_chunk_combine(self, tmp_path):
        """Lookahead and chunk simulation compose: lookahead shifts the
        query, chunk window is still driven by current wall time."""
        frames = [(i * 0.1, [1.0 + i * 0.1]) for i in range(6)]
        # chunk_size=3 + 50 ms lookahead. At elapsed=0.18 we're still in
        # chunk 0 (frame index 1 → chunk 0 covers frames 0..2). Query time
        # = 0.18 + 0.05 = 0.23 → past chunk 0 end (t=0.2). Extrapolate.
        t = _make_teleop(
            tmp_path,
            frames=frames,
            joints=["a.pos"],
            simulate_chunk_size=3,
            lookahead_s=0.05,
        )
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.18
            a = t.get_action()
        # chunk 0 velocity = (1.2 - 1.1) / 0.1 = 1.0; ahead = 0.23 - 0.2 = 0.03
        # → 1.2 + 1.0 * 0.03 = 1.23
        assert a["a.pos"] == pytest.approx(1.23)


class TestStartPose:
    def test_returns_first_frame_as_dict(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        assert t.start_pose == {"a.pos": 1.0, "b.pos": 2.0}

    def test_raises_if_not_connected(self, tmp_path):
        t = _make_teleop(tmp_path)
        with pytest.raises(RuntimeError, match="connect"):
            _ = t.start_pose


class TestDisconnect:
    def test_clears_state(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        t.get_action()
        t.disconnect()
        assert not t.is_connected
        assert not t.is_exhausted
        # Reconnecting works (re-loads the file, re-arms).
        t.connect()
        assert t.is_connected

    def test_reconnect_resets_exhausted(self, tmp_path):
        t = _make_teleop(tmp_path)
        t.connect()
        with patch(
            "lerobot.teleoperators.trajectory_replay.teleop_trajectory_replay.time.perf_counter"
        ) as mock_t:
            mock_t.return_value = 100.0
            t.get_action()
            mock_t.return_value = 100.5
            t.get_action()
            assert t.is_exhausted
        t.disconnect()
        t.connect()
        assert not t.is_exhausted
