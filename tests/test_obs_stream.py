"""Tests for the robot observation stream (shared memory)."""

import os
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.robots.config import RobotConfig
from lerobot.robots.obs_stream import (
    ENV_VAR,
    ObservationStream,
    ObservationStreamReader,
    _active_robot_id,
    _active_stream,
    _connect_depth,
)
from lerobot.robots.robot import Robot


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_features():
    obs = {"j1.pos": float, "j2.pos": float, "cam": (240, 320, 3)}
    act = {"j1.pos": float, "j2.pos": float}
    return obs, act


@pytest.fixture
def stream(simple_features):
    obs_ft, act_ft = simple_features
    s = ObservationStream(obs_ft, act_ft)
    yield s
    s.cleanup()


@pytest.fixture
def obs_dict():
    return {
        "j1.pos": 10.5,
        "j2.pos": -3.2,
        "cam": np.random.randint(0, 255, (240, 320, 3), dtype=np.uint8),
    }


@pytest.fixture
def action_dict():
    return {"j1.pos": 11.0, "j2.pos": -2.5}


# ============================================================================
# Writer / Reader round-trip
# ============================================================================


class TestRoundTrip:
    def test_scalar_obs(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        result = reader.read_obs()
        assert result is not None
        obs, ts = result
        assert abs(obs["j1.pos"] - 10.5) < 1e-4
        assert abs(obs["j2.pos"] - (-3.2)) < 1e-4
        assert ts > 0
        reader.close()

    def test_action(self, stream, obs_dict, action_dict):
        stream.write_obs(obs_dict)  # meta block needs data
        stream.write_action(action_dict)
        reader = ObservationStreamReader()
        result = reader.read_action()
        assert result is not None
        act, ts = result
        assert abs(act["j1.pos"] - 11.0) < 1e-4
        assert abs(act["j2.pos"] - (-2.5)) < 1e-4
        reader.close()

    def test_image(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        result = reader.read_image("cam")
        assert result is not None
        img, ts = result
        assert img.shape == (240, 320, 3)
        assert img.dtype == np.uint8
        assert np.array_equal(img, obs_dict["cam"])
        reader.close()

    def test_missing_camera_returns_none(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        assert reader.read_image("nonexistent") is None
        reader.close()

    def test_no_data_returns_none(self, stream):
        """Reader attached but no write_obs called yet."""
        reader = ObservationStreamReader()
        assert reader.read_obs() is None
        assert reader.read_action() is None
        assert reader.read_image("cam") is None
        reader.close()


# ============================================================================
# Sequence counter
# ============================================================================


class TestSequenceCounter:
    def test_seq_increments(self, stream, obs_dict):
        reader = ObservationStreamReader()
        assert reader.image_seq("cam") == 0

        stream.write_obs(obs_dict)
        assert reader.image_seq("cam") == 1

        stream.write_obs(obs_dict)
        assert reader.image_seq("cam") == 2
        reader.close()

    def test_seq_unchanged_without_write(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        seq1 = reader.image_seq("cam")
        seq2 = reader.image_seq("cam")
        assert seq1 == seq2
        reader.close()


# ============================================================================
# Layout metadata
# ============================================================================


class TestMetadata:
    def test_reader_sees_layout(self, stream, obs_dict):
        stream.write_obs(obs_dict)
        reader = ObservationStreamReader()
        assert sorted(reader.obs_scalar_keys) == ["j1.pos", "j2.pos"]
        assert sorted(reader.action_keys) == ["j1.pos", "j2.pos"]
        assert "cam" in reader.image_keys
        assert reader.image_keys["cam"] == [240, 320, 3]
        reader.close()


# ============================================================================
# __init_subclass__ wrapping with mock robots
# ============================================================================


@dataclass
class _MockConfig(RobotConfig):
    pass


class _SimpleRobot(Robot):
    config_class = _MockConfig
    name = "test_simple"

    def __init__(self):
        self._connected = False

    @property
    def observation_features(self):
        return {"j.pos": float, "cam": (120, 160, 3)}

    @property
    def action_features(self):
        return {"j.pos": float}

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        return {"j.pos": 1.0, "cam": np.zeros((120, 160, 3), dtype=np.uint8)}

    def send_action(self, action):
        return action


class _SubRobot(Robot):
    """Sub-robot for composite testing."""

    config_class = _MockConfig
    name = "test_sub"

    def __init__(self, prefix):
        self._connected = False
        self._prefix = prefix

    @property
    def observation_features(self):
        return {f"{self._prefix}_j.pos": float}

    @property
    def action_features(self):
        return self.observation_features

    @property
    def is_connected(self):
        return self._connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self._connected = True

    def disconnect(self):
        self._connected = False

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        return {f"{self._prefix}_j.pos": 5.0}

    def send_action(self, action):
        return action


class _CompositeRobot(Robot):
    """Composite robot wrapping two sub-robots."""

    config_class = _MockConfig
    name = "test_composite"

    def __init__(self):
        self.left = _SubRobot("left")
        self.right = _SubRobot("right")

    @property
    def observation_features(self):
        return {**self.left.observation_features, **self.right.observation_features}

    @property
    def action_features(self):
        return {**self.left.action_features, **self.right.action_features}

    @property
    def is_connected(self):
        return self.left.is_connected and self.right.is_connected

    @property
    def is_calibrated(self):
        return True

    def connect(self, calibrate=True):
        self.left.connect()
        self.right.connect()

    def disconnect(self):
        self.left.disconnect()
        self.right.disconnect()

    def calibrate(self):
        pass

    def configure(self):
        pass

    def get_observation(self):
        obs = {}
        obs.update(self.left.get_observation())
        obs.update(self.right.get_observation())
        return obs

    def send_action(self, action):
        self.left.send_action({k: v for k, v in action.items() if "left" in k})
        self.right.send_action({k: v for k, v in action.items() if "right" in k})
        return action


class TestInitSubclassWrapping:
    """Test that Robot.__init_subclass__ wrapping works correctly."""

    def test_methods_are_wrapped(self):
        """connect/get_observation/send_action/disconnect should be closures."""
        for method_name in ("connect", "disconnect", "get_observation", "send_action"):
            fn = _SimpleRobot.__dict__[method_name]
            assert fn.__closure__ is not None, f"{method_name} should be wrapped"

    def test_no_stream_without_env_var(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV_VAR, None)
            robot = _SimpleRobot()
            robot.connect()
            assert mod._active_stream is None
            robot.disconnect()

    def test_stream_created_with_env_var(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {ENV_VAR: "1"}):
            robot = _SimpleRobot()
            robot.connect()
            assert mod._active_stream is not None
            assert mod._active_robot_id == id(robot)

            # get_observation should publish
            obs = robot.get_observation()
            reader = ObservationStreamReader()
            result = reader.read_obs()
            assert result is not None
            assert abs(result[0]["j.pos"] - 1.0) < 1e-4
            reader.close()

            robot.disconnect()
            assert mod._active_stream is None

    def test_composite_only_outermost_publishes(self):
        import lerobot.robots.obs_stream as mod

        with patch.dict(os.environ, {ENV_VAR: "1"}):
            robot = _CompositeRobot()
            robot.connect()

            # Only the composite should own the stream
            assert mod._active_robot_id == id(robot)
            assert mod._active_robot_id != id(robot.left)
            assert mod._active_robot_id != id(robot.right)

            # Composite's get_observation publishes both joints
            robot.get_observation()
            reader = ObservationStreamReader()
            result = reader.read_obs()
            assert result is not None
            assert "left_j.pos" in result[0]
            assert "right_j.pos" in result[0]
            reader.close()

            robot.disconnect()
            assert mod._active_stream is None


# ============================================================================
# Multiple cameras / various feature shapes
# ============================================================================


class TestMultipleCameras:
    def test_four_cameras(self):
        obs_ft = {f"j{i}.pos": float for i in range(14)}
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            obs_ft[cam] = (720, 1280, 3)
        act_ft = {f"j{i}.pos": float for i in range(14)}

        stream = ObservationStream(obs_ft, act_ft)
        obs = {f"j{i}.pos": float(i) for i in range(14)}
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            obs[cam] = np.zeros((720, 1280, 3), dtype=np.uint8)
        stream.write_obs(obs)

        reader = ObservationStreamReader()
        assert len(reader.image_keys) == 4
        for cam in ["front", "top", "left_wrist", "right_wrist"]:
            result = reader.read_image(cam)
            assert result is not None
            assert result[0].shape == (720, 1280, 3)
        reader.close()
        stream.cleanup()

    def test_no_cameras(self):
        """Robots with only scalar features (no cameras)."""
        stream = ObservationStream({"j.pos": float}, {"j.pos": float})
        stream.write_obs({"j.pos": 42.0})

        reader = ObservationStreamReader()
        assert len(reader.image_keys) == 0
        result = reader.read_obs()
        assert result is not None
        assert abs(result[0]["j.pos"] - 42.0) < 1e-4
        reader.close()
        stream.cleanup()
