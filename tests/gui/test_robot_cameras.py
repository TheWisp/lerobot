"""Tests for the camera preview endpoints' interaction with active run subprocesses."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException


class TestDetectCamerasGuard:
    """POST /api/robot/detect-cameras must refuse while a run subprocess owns the cameras."""

    def test_refuses_when_run_active(self):
        from lerobot.gui.api.robot import detect_cameras

        proc = AsyncMock()
        proc.returncode = None  # still running
        with patch("lerobot.gui.api.run._active_process", proc):
            with pytest.raises(HTTPException) as exc_info:
                asyncio.run(detect_cameras())
        assert exc_info.value.status_code == 409
        assert "cameras" in exc_info.value.detail

    def test_opens_previews_when_no_run(self):
        from lerobot.gui.api import robot

        fake_cams = [{"id": "/dev/video0", "type": "opencv"}]
        with (
            patch("lerobot.gui.api.run._active_process", None),
            patch.object(robot, "_detect_and_open_cameras", return_value=fake_cams),
        ):
            result = asyncio.run(robot.detect_cameras())
        assert result == fake_cams


class TestPreviewReleaseOnLaunch:
    """Launching a run must hand the cameras over, not compete for them.

    The Robot tab holds a V4L2 / librealsense handle per previewed camera for as
    long as the GUI runs. Only the server shutdown hook released them, so a run
    launched from the Run tab after visiting the Robot tab fought the GUI for its
    own cameras. Reproduced on real hardware: a teleop-style open of /dev/video0
    at 1280x720 MJPG succeeds with nothing held and raises ConnectionError with a
    GUI-style preview held on the same device.

    The complementary guard — refusing to open a preview *while* a run is active
    — is tested in TestDetectCamerasGuard above. This is the other direction.
    """

    def test_launch_releases_held_previews(self):
        import lerobot.gui.api.robot as robot_mod
        from lerobot.gui.api.run import _launch_subprocess

        class FakePreviewCamera:
            def __init__(self):
                self.disconnected = False

            def disconnect(self):
                self.disconnected = True

        held = [FakePreviewCamera(), FakePreviewCamera()]
        robot_mod._preview_cameras.extend(held)
        robot_mod._preview_camera_info.extend([{"id": "a"}, {"id": "b"}])

        proc = AsyncMock()
        proc.returncode = None
        proc.pid = 999
        proc.stdout = None
        proc.stderr = None

        try:
            with (
                patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
                patch("lerobot.gui.api.run._read_stream", AsyncMock()),
            ):
                asyncio.run(_launch_subprocess(["true"], "test", {}))
        finally:
            robot_mod._preview_cameras.clear()
            robot_mod._preview_camera_info.clear()

        assert all(c.disconnected for c in held), (
            "previews were still held when the subprocess launched — it will compete "
            "with the GUI for the same /dev/video* devices"
        )

    def test_release_preview_cameras_reports_and_clears(self):
        import lerobot.gui.api.robot as robot_mod
        from lerobot.gui.api.robot import release_preview_cameras

        class FakePreviewCamera:
            def disconnect(self):
                pass

        robot_mod._preview_cameras.extend([FakePreviewCamera(), FakePreviewCamera()])
        robot_mod._preview_camera_info.extend([{"id": "a"}, {"id": "b"}])

        assert release_preview_cameras() == 2
        assert robot_mod._preview_cameras == []
        assert robot_mod._preview_camera_info == []
        # Idempotent: a second call on an empty set is a no-op, not an error.
        assert release_preview_cameras() == 0
