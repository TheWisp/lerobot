"""Regression tests for camera preview FD lifecycle.

The GUI was leaking `/dev/video*` file descriptors when camera initialisation
failed partway through `_detect_and_open_cameras()`. The per-camera body now
uses an ownership-transfer try/finally so any error between `connect()` and
successful registration triggers `disconnect()`.
"""

from unittest.mock import MagicMock, patch

import pytest

from lerobot.gui.api import robot as robot_module


@pytest.fixture(autouse=True)
def _reset_preview_state():
    """Ensure module-level preview lists are clean around each test."""
    robot_module._preview_cameras.clear()
    robot_module._preview_camera_info.clear()
    yield
    robot_module._preview_cameras.clear()
    robot_module._preview_camera_info.clear()


def _make_mock_camera_class(connect_side_effects):
    """Create a mock camera class whose instances follow the given connect()
    side effects in order. Each instance's `disconnect()` is a MagicMock.

    Returns (MockClass, list_of_instances) so callers can assert on per-camera
    behaviour after the call.
    """
    instances: list[MagicMock] = []
    side_effect_iter = iter(connect_side_effects)

    def factory(_config):
        inst = MagicMock()
        try:
            effect = next(side_effect_iter)
        except StopIteration:
            effect = None
        if effect is None:
            inst.connect = MagicMock(return_value=None)
        else:
            inst.connect = MagicMock(side_effect=effect)
        instances.append(inst)
        return inst

    cls = MagicMock(side_effect=factory)
    return cls, instances


def test_opencv_connect_failure_invokes_disconnect():
    """If OpenCV camera connect() raises, the partial handle is disconnected.

    Mirrors the originally-reported FD leak: connect() opens the V4L2 node,
    then warmup raises, and the old code abandoned the partially-opened
    handle. With the fix, the finally clause must call disconnect().
    """
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([RuntimeError("warmup failed")])
    # Use integer ID to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(return_value=[{"id": 0, "name": "test-cam"}])

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 1
    opencv_instances[0].disconnect.assert_called_once()
    # The leaked camera must NOT remain in the preview list.
    assert robot_module._preview_cameras == []
    assert robot_module._preview_camera_info == []


def test_realsense_connect_failure_invokes_disconnect():
    """Same invariant for the RealSense branch."""
    mock_opencv_stub = MagicMock()
    mock_opencv_stub.find_cameras = MagicMock(return_value=[])

    mock_realsense_cls, rs_instances = _make_mock_camera_class([RuntimeError("librealsense init failed")])
    mock_realsense_cls.find_cameras = MagicMock(return_value=[{"id": "123456789", "name": "D435"}])

    with (
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_cls,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_stub,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(rs_instances) == 1
    rs_instances[0].disconnect.assert_called_once()
    assert robot_module._preview_cameras == []


def test_successful_connect_does_not_disconnect():
    """The happy path must NOT spuriously disconnect (ownership transferred).

    Without the `camera = None` reset after successful registration, a buggy
    fix could call disconnect() on every camera.
    """
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([None, None])
    # Use integer IDs to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(
        return_value=[
            {"id": 0, "name": "cam-a"},
            {"id": 1, "name": "cam-b"},
        ]
    )

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 2
    for inst in opencv_instances:
        inst.disconnect.assert_not_called()
    assert len(robot_module._preview_cameras) == 2


def test_partial_failure_does_not_leak_other_cameras():
    """One camera failing mid-iteration must not affect the others."""
    mock_opencv_cls, opencv_instances = _make_mock_camera_class([None, RuntimeError("cam 2 failed"), None])
    # Use integer IDs to bypass the `/dev/video*` sysfs RealSense check.
    mock_opencv_cls.find_cameras = MagicMock(
        return_value=[
            {"id": 0, "name": "ok-1"},
            {"id": 1, "name": "bad"},
            {"id": 2, "name": "ok-2"},
        ]
    )

    mock_realsense_stub = MagicMock()
    mock_realsense_stub.find_cameras = MagicMock(return_value=[])

    with (
        patch(
            "lerobot.cameras.opencv.camera_opencv.OpenCVCamera",
            mock_opencv_cls,
        ),
        patch(
            "lerobot.cameras.opencv.configuration_opencv.OpenCVCameraConfig",
            MagicMock(),
        ),
        patch(
            "lerobot.cameras.realsense.camera_realsense.RealSenseCamera",
            mock_realsense_stub,
        ),
        patch(
            "lerobot.cameras.realsense.configuration_realsense.RealSenseCameraConfig",
            MagicMock(),
        ),
    ):
        robot_module._detect_and_open_cameras()

    assert len(opencv_instances) == 3
    # The two successful cameras are registered and NOT disconnected.
    opencv_instances[0].disconnect.assert_not_called()
    opencv_instances[2].disconnect.assert_not_called()
    # The failing camera IS disconnected.
    opencv_instances[1].disconnect.assert_called_once()
    assert len(robot_module._preview_cameras) == 2
