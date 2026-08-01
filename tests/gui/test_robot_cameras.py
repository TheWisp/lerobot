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


class TestProbeMotorSpecCache:
    """Wiggle must not rebuild a whole robot before every motor twitch.

    _probe_motor_spec needs only the bus class and the first motor, but derived
    them by constructing the entire robot via make_robot_from_config — measured
    at ~1.1s for a bimanual SO-107, paid before the motor moves on every Wiggle
    click. Not a regression (identical on main), just slow on an interactive
    path, and the answer is a pure function of the profile.
    """

    def _profile(self):
        return {
            "type": "bi_so107_follower",
            "fields": {"left_arm_port": "/dev/ttyACM0", "right_arm_port": "/dev/ttyACM1"},
        }

    def test_repeat_probe_is_served_from_cache(self):
        import lerobot.gui.api.robot as robot_mod
        from lerobot.gui.api.robot import _probe_motor_spec

        robot_mod._MOTOR_SPEC_CACHE.clear()
        profile = self._profile()
        try:
            first = _probe_motor_spec(profile)
        except Exception as e:  # robot extra not installed in this env
            pytest.skip(f"bi_so107_follower unavailable here: {type(e).__name__}")

        assert len(robot_mod._MOTOR_SPEC_CACHE) == 1, "first probe did not populate the cache"

        with patch("lerobot.robots.utils.make_robot_from_config") as never_called:
            second = _probe_motor_spec(profile)
            assert not never_called.called, "second probe rebuilt the robot instead of using the cache"

        assert second[0] is first[0], "bus class changed between probes"
        assert second[1] == first[1], "motor name changed between probes"

    def test_edited_profile_is_not_served_a_stale_spec(self):
        """Keyed on content, so changing a field must miss the cache."""
        import lerobot.gui.api.robot as robot_mod
        from lerobot.gui.api.robot import _probe_motor_spec

        robot_mod._MOTOR_SPEC_CACHE.clear()
        profile = self._profile()
        try:
            _probe_motor_spec(profile)
        except Exception as e:
            pytest.skip(f"bi_so107_follower unavailable here: {type(e).__name__}")

        edited = {**profile, "fields": {**profile["fields"], "left_arm_port": "/dev/ttyACM9"}}
        _probe_motor_spec(edited)
        assert len(robot_mod._MOTOR_SPEC_CACHE) == 2, (
            "an edited profile reused the previous entry — a stale motor spec would be probed"
        )

    def test_cached_motor_is_a_copy(self):
        """The caller hands the motor to a bus constructor; sharing would leak mutation."""
        import lerobot.gui.api.robot as robot_mod
        from lerobot.gui.api.robot import _probe_motor_spec

        robot_mod._MOTOR_SPEC_CACHE.clear()
        profile = self._profile()
        try:
            first = _probe_motor_spec(profile)
        except Exception as e:
            pytest.skip(f"bi_so107_follower unavailable here: {type(e).__name__}")
        second = _probe_motor_spec(profile)
        assert first[2] is not second[2], "probes share one Motor instance"
