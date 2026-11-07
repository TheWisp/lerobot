"""Camera detection with a ZED on the host (node numbers chosen not to exist on a developer machine, whose real sysfs the scan reads): the unit is offered once, as a ZED,
and the V4L2 node the SDK holds is not listed again as a broken OpenCV camera."""

from __future__ import annotations

from unittest.mock import patch

from lerobot.cameras.zed.camera_zed import ZEDCamera
from lerobot.gui.api import robot

ZED_INFO = {
    "name": "Stereolabs ZED-M",
    "type": "ZED",
    "id": "11975257",
    "state": "AVAILABLE",
    "path": "/dev/video94",
}


def _fake_connect(self, warmup=True):
    self._zed = object()
    self.width, self.height, self.fps = 1280, 720, 30


def _run_detection(zed_found, opencv_found, candidates):
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
    from lerobot.cameras.realsense.camera_realsense import RealSenseCamera

    with (
        patch.object(ZEDCamera, "find_cameras", staticmethod(lambda: list(zed_found))),
        patch.object(ZEDCamera, "connect", _fake_connect),
        patch.object(ZEDCamera, "disconnect", lambda self: None),
        patch.object(RealSenseCamera, "find_cameras", staticmethod(lambda: [])),
        patch.object(OpenCVCamera, "find_cameras", staticmethod(lambda: list(opencv_found))),
        patch.object(OpenCVCamera, "connect", lambda self, warmup=True: None),
        patch.object(OpenCVCamera, "disconnect", lambda self: None),
        patch.object(robot, "_linux_video_capture_candidates", lambda: list(candidates)),
    ):
        try:
            return robot._detect_and_open_cameras()
        finally:
            robot._close_preview_cameras()


def test_a_zed_is_offered_once_and_its_node_is_not_a_broken_opencv_camera():
    candidates = [
        {"name": "ZED-M: ZED-M @ /dev/video94", "type": "OpenCV", "id": "/dev/video94"},
        {"name": "Arducam @ /dev/video90", "type": "OpenCV", "id": "/dev/video90"},
    ]
    opencv_found = [{"name": "Arducam @ /dev/video90", "type": "OpenCV", "id": "/dev/video90"}]
    found = _run_detection([ZED_INFO], opencv_found, candidates)
    by_id = {(c["type"], c["id"]): c for c in found}
    assert ("ZED", "11975257") in by_id and "error" not in by_id[("ZED", "11975257")]
    assert by_id[("ZED", "11975257")]["default_stream_profile"] == {"width": 1280, "height": 720, "fps": 30}
    assert ("OpenCV", "/dev/video90") in by_id
    assert ("OpenCV", "/dev/video94") not in by_id


def test_without_a_zed_an_unopenable_node_is_still_reported():
    candidates = [{"name": "ZED-M: ZED-M @ /dev/video94", "type": "OpenCV", "id": "/dev/video94"}]
    found = _run_detection([], [], candidates)
    assert [(c["type"], c["id"]) for c in found] == [("OpenCV", "/dev/video94")]
    assert "error" in found[0]
