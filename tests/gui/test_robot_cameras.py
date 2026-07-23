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
