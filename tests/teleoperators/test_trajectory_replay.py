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


def _make_teleop(tmp_path: Path, frames=None, joints=None) -> TrajectoryReplayTeleop:
    if frames is None:
        frames = [(0.0, [1.0, 2.0]), (0.1, [1.1, 2.1]), (0.2, [1.2, 2.2])]
    if joints is None:
        joints = ["a.pos", "b.pos"]
    path = _write_trajectory(tmp_path, frames, joints)
    cfg = TrajectoryReplayTeleopConfig(id="test", trajectory_path=str(path))
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
