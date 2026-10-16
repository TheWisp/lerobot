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
"""The ZED driver against a stand-in SDK that behaves like the real one did on
the rig (ZED-M, SDK 5.2.3, HD720): BGRA eyes, uint16 millimetre depth with 0
for no measurement, rectified intrinsics on the camera information."""

import sys
import time
import types
from enum import Enum

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.cameras.zed import ZEDCamera, ZEDCameraConfig
from lerobot.utils.errors import DeviceNotConnectedError

W, H = 1280, 720
FX, FY, CX, CY = 732.717, 732.717, 626.361, 355.161
SERIAL = 11975257


def _enum(name, members):
    return Enum(name, {m: i for i, m in enumerate(members)})


class _Mat:
    def __init__(self):
        self._data = None

    def get_data(self):
        return self._data


class _CamParams:
    fx, fy, cx, cy = FX, FY, CX, CY
    disto = np.zeros(12)


class _Size:
    width, height = W, H


class _Calib:
    left_cam = _CamParams()
    right_cam = _CamParams()

    @staticmethod
    def get_camera_baseline():
        return 62.941


class _Config:
    resolution = _Size()
    fps = 30.0
    calibration_parameters = _Calib()


class _Info:
    serial_number = SERIAL
    camera_model = "ZED-M"
    camera_configuration = _Config()


class _Device:
    serial_number = SERIAL
    camera_model = "ZED-M"
    camera_state = "AVAILABLE"
    path = "/dev/video4"


def make_fake_sdk(*, open_ok=True, grab_ok=True, grab_stall_s=0.0, stall_at_grab=None):
    """A ``pyzed.sl`` stand-in. Each grab produces a new frame: the left eye's
    first pixel counts grabs, the right eye differs from the left, and the
    depth map is a ramp with a zero (unmeasured) band."""
    sl = types.ModuleType("pyzed.sl")
    sl.RESOLUTION = _enum("RESOLUTION", ["HD2K", "HD1080", "HD1200", "HD720", "SVGA", "VGA", "AUTO"])
    sl.DEPTH_MODE = _enum(
        "DEPTH_MODE", ["NONE", "PERFORMANCE", "QUALITY", "ULTRA", "NEURAL_LIGHT", "NEURAL", "NEURAL_PLUS"]
    )
    sl.VIEW = _enum("VIEW", ["LEFT", "RIGHT"])
    sl.MEASURE = _enum("MEASURE", ["DEPTH", "DEPTH_U16_MM"])
    sl.UNIT = _enum("UNIT", ["MILLIMETER", "METER"])
    sl.MEM = _enum("MEM", ["CPU", "GPU"])
    sl.FLIP_MODE = _enum("FLIP_MODE", ["OFF", "ON", "AUTO"])
    sl.ERROR_CODE = _enum("ERROR_CODE", ["SUCCESS", "CAMERA_NOT_DETECTED", "FAILURE"])
    sl.Mat = _Mat

    class InitParameters:
        def __init__(self):
            self.serial = None

        def set_from_serial_number(self, serial):
            self.serial = serial

    class RuntimeParameters:
        enable_depth = True
        enable_fill_mode = False

    class Camera:
        instances = []

        def __init__(self):
            self.opened = False
            self.init = None
            self.grabs = 0
            Camera.instances.append(self)

        def open(self, init):
            self.init = init
            if not open_ok:
                return sl.ERROR_CODE.CAMERA_NOT_DETECTED
            self.opened = True
            return sl.ERROR_CODE.SUCCESS

        def is_opened(self):
            return self.opened

        def close(self):
            self.opened = False

        def get_camera_information(self):
            return _Info()

        @staticmethod
        def get_device_list():
            return [_Device()]

        def grab(self, runtime):
            self.grabs += 1
            if not grab_ok:
                return sl.ERROR_CODE.FAILURE
            # One grab blocks for grab_stall_s, the way the real SDK sometimes does.
            if stall_at_grab is not None and self.grabs == stall_at_grab:
                time.sleep(grab_stall_s)
            time.sleep(0.033)  # the real camera's cadence at 30 fps
            return sl.ERROR_CODE.SUCCESS

        def retrieve_image(self, mat, view, mem):
            bgra = np.zeros((H, W, 4), np.uint8)
            bgra[..., 3] = 255
            bgra[..., 2] = 200 if view == sl.VIEW.LEFT else 50  # red channel in BGRA
            bgra[0, 0, 0] = self.grabs % 256  # grab counter in blue
            mat._data = bgra

        def retrieve_measure(self, mat, measure, mem):
            assert measure == sl.MEASURE.DEPTH_U16_MM
            depth = np.tile(np.linspace(300, 1500, W, dtype=np.float32), (H, 1)).astype(np.uint16)
            depth[:, :100] = 0  # no measurement
            mat._data = depth

    sl.InitParameters = InitParameters
    sl.RuntimeParameters = RuntimeParameters
    sl.Camera = Camera
    return sl


@pytest.fixture
def fake_sdk(monkeypatch):
    sl = make_fake_sdk()
    pkg = types.ModuleType("pyzed")
    pkg.sl = sl
    monkeypatch.setitem(sys.modules, "pyzed", pkg)
    monkeypatch.setitem(sys.modules, "pyzed.sl", sl)
    return sl


def _config(**kw):
    base = {"fps": 30, "width": W, "height": H, "serial_number": SERIAL, "use_depth": True, "warmup_s": 1}
    base.update(kw)
    return ZEDCameraConfig(**base)


def test_config_validates_resolution_and_depth_mode():
    with pytest.raises(ValueError, match="eye resolution"):
        ZEDCameraConfig(fps=30, width=1234, height=720)
    # Sizes name the frames as published: a quarter turn is configured
    # transposed, and the sensor's frame behind it is what must be an SDK size.
    ZEDCameraConfig(fps=30, width=720, height=1280, rotation=Cv2Rotation.ROTATE_90)
    with pytest.raises(ValueError, match="sensor's frame behind a published 1280x720"):
        ZEDCameraConfig(fps=30, width=1280, height=720, rotation=Cv2Rotation.ROTATE_90)
    with pytest.raises(ValueError, match="depth_mode"):
        ZEDCameraConfig(depth_mode="turbo")
    with pytest.raises(ValueError, match="use_depth"):
        ZEDCameraConfig(use_depth=True, depth_mode="none")


def test_without_sdk_the_module_loads_and_connect_says_how_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "pyzed", None)
    monkeypatch.setitem(sys.modules, "pyzed.sl", None)
    cam = ZEDCamera(_config())
    with pytest.raises(ImportError, match="ZED SDK"):
        cam.connect()


def test_find_cameras_lists_the_sdk_devices(fake_sdk):
    found = ZEDCamera.find_cameras()
    assert [(c["id"], c["type"]) for c in found] == [(str(SERIAL), "ZED")]


def test_connect_reports_rectified_intrinsics_and_opens_the_named_unit(fake_sdk):
    cam = ZEDCamera(_config())
    cam.connect()
    try:
        assert fake_sdk.Camera.instances[-1].init.serial == SERIAL
        assert cam.intrinsics is not None and cam.intrinsics_right is not None
        assert (cam.intrinsics.fx, cam.intrinsics.cy) == (FX, CY)
        assert cam.intrinsics.model == "rectified_pinhole" and cam.intrinsics.coeffs == ()
        assert cam.baseline_m == pytest.approx(0.062941)
        assert cam.serial_number == str(SERIAL) and cam.device_name == "Stereolabs ZED-M"
    finally:
        cam.disconnect()


def test_frames_are_rgb_eyes_and_uint16_depth_from_one_grab(fake_sdk):
    cam = ZEDCamera(_config())
    cam.connect()
    try:
        left, right, depth = cam.read_latest_stereo_and_depth()
        assert left.shape == (H, W, 3) and left.dtype == np.uint8
        assert left[1, 1].tolist() == [200, 0, 0]  # BGRA red -> RGB first channel
        assert right[1, 1].tolist() == [50, 0, 0]  # the other eye
        assert depth.shape == (H, W, 1) and depth.dtype == np.uint16
        assert int(depth[:, :100].max()) == 0 and int(depth[0, -1, 0]) == 1500
        assert left[0, 0, 2] == right[0, 0, 2]  # the grab counter: both eyes from the same grab
        assert cam.read_latest().shape == (H, W, 3)
        assert cam.read_latest_right()[1, 1, 0] == 50
        assert cam.read_latest_depth().shape == (H, W, 1)
        colour, d2 = cam.read_latest_color_and_depth()
        assert colour.shape == (H, W, 3) and d2.shape == (H, W, 1)
    finally:
        cam.disconnect()


def test_rotation_turns_frames_depth_and_intrinsics_together(fake_sdk):
    """A portrait mount is configured as published (H wide, W high); the SDK
    is opened at the sensor's landscape size and everything handed out is
    turned together."""
    cam = ZEDCamera(_config(width=H, height=W, rotation=Cv2Rotation.ROTATE_90))
    cam.connect()
    try:
        assert fake_sdk.Camera.instances[-1].init.camera_resolution == fake_sdk.RESOLUTION.HD720
        left, _, depth = cam.read_latest_stereo_and_depth()
        assert left.shape == (W, H, 3) and depth.shape == (W, H, 1)
        assert (cam.width, cam.height) == (H, W)
        assert (cam.intrinsics.width, cam.intrinsics.height) == (H, W)
    finally:
        cam.disconnect()


def test_depth_off_means_no_depth_reads_and_no_runtime_depth(fake_sdk):
    """Colour only: the SDK is opened with no depth mode at all, whatever the
    profile's depth_mode says, so no model is loaded and the GPU stays free."""
    cam = ZEDCamera(_config(use_depth=False))
    cam.connect()
    try:
        assert fake_sdk.Camera.instances[-1].init.depth_mode == fake_sdk.DEPTH_MODE.NONE
        assert cam._runtime.enable_depth is False
        with pytest.raises(RuntimeError, match="use_depth"):
            cam.read_latest_depth()
        assert cam.read_latest_stereo()[0].shape == (H, W, 3)
    finally:
        cam.disconnect()


def test_open_failure_is_a_connection_error_and_leaves_nothing_open(fake_sdk, monkeypatch):
    failing = make_fake_sdk(open_ok=False)
    monkeypatch.setitem(sys.modules, "pyzed.sl", failing)
    sys.modules["pyzed"].sl = failing
    cam = ZEDCamera(_config())
    with pytest.raises(ConnectionError, match="Failed to open"):
        cam.connect()
    assert not cam.is_connected
    with pytest.raises(DeviceNotConnectedError):
        cam.disconnect()


def test_attach_writes_both_eyes_and_the_depth_aligned_to_the_left(fake_sdk, tmp_path):
    from lerobot.datasets.camera_calibration import attach_camera_intrinsics

    cam = ZEDCamera(_config())
    cam.connect()
    try:
        features = {
            "observation.images.top_l": {"shape": [H, W, 3], "info": {}},
            "observation.images.top_r": {"shape": [H, W, 3], "info": {}},
            "observation.images.top_depth": {"shape": [H, W, 1], "info": {"is_depth_map": True}},
        }
        info = types.SimpleNamespace(to_dict=lambda: {"features": features})
        meta = types.SimpleNamespace(features=features, info=info, root=tmp_path)
        written = attach_camera_intrinsics(meta, {"top": cam})
    finally:
        cam.disconnect()
    assert written == [
        "observation.images.top_l",
        "observation.images.top_r",
        "observation.images.top_depth",
    ]
    depth_block = features["observation.images.top_depth"]["info"]["camera"]
    assert depth_block["aligned_to"] == "observation.images.top_l"
    assert depth_block["intrinsics"]["fx"] == FX and depth_block["serial"] == str(SERIAL)
    assert "aligned_to" not in features["observation.images.top_r"]["info"]["camera"]


def _install(monkeypatch, sl):
    pkg = types.ModuleType("pyzed")
    pkg.sl = sl
    monkeypatch.setitem(sys.modules, "pyzed", pkg)
    monkeypatch.setitem(sys.modules, "pyzed.sl", sl)


def test_a_stalled_grab_hands_out_the_last_frame_without_waiting(monkeypatch, caplog):
    """The SDK paused on one grab: a peek past its limit gets the frame in
    hand at once, with one warning, rather than blocking the control loop
    or ending the episode."""
    _install(monkeypatch, make_fake_sdk(grab_stall_s=1.0, stall_at_grab=60))
    cam = ZEDCamera(_config(warmup_s=1))
    cam.connect()
    try:
        sdk_cam = sys.modules["pyzed.sl"].Camera.instances[-1]
        while sdk_cam.grabs < 60:  # the stalled grab is in flight once the counter reads 60
            time.sleep(0.005)
        time.sleep(0.3)  # the cached frame is now ~300 ms old: past a 100 ms limit, inside 4x
        with caplog.at_level("WARNING"):
            t = time.perf_counter()
            left, right, depth = cam.read_latest_stereo_and_depth(max_age_ms=100)
            took_s = time.perf_counter() - t
            cam.read_latest_stereo_and_depth(max_age_ms=100)
        assert took_s < 0.05  # handed out, not waited for
        assert left.shape == (H, W, 3) and depth.shape == (H, W, 1)
        stale = [r for r in caplog.records if "handing out the last one" in r.message]
        assert len(stale) == 1  # one warning a second, not one per frame
    finally:
        cam.disconnect()


def test_a_grab_that_stays_away_is_a_timeout(monkeypatch):
    _install(monkeypatch, make_fake_sdk(grab_stall_s=5.0, stall_at_grab=60))
    cam = ZEDCamera(_config(warmup_s=1))
    cam.connect()
    try:
        sdk_cam = sys.modules["pyzed.sl"].Camera.instances[-1]
        while sdk_cam.grabs < 60:  # past connect's one-second warm-up
            time.sleep(0.005)
        time.sleep(0.5)  # ~500 ms old: past 4x a 100 ms limit
        t = time.perf_counter()
        with pytest.raises(TimeoutError, match="gave up at 4x"):
            cam.read_latest_stereo_and_depth(max_age_ms=100)
        assert time.perf_counter() - t < 0.05  # raised at once, not after the 5 s stall
    finally:
        cam.disconnect()


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")  # the loop gives up
def test_a_failed_warmup_leaves_nothing_open(monkeypatch):
    """Grabs that never succeed fail the connect as a ConnectionError, and
    the unit, thread and runtime are released so a retry opens it afresh."""
    _install(monkeypatch, make_fake_sdk(grab_ok=False))
    cam = ZEDCamera(_config(warmup_s=1))
    with pytest.raises(ConnectionError, match="warmup"):
        cam.connect()
    assert not cam.is_connected
    assert cam.thread is None and cam._zed is None and cam._runtime is None
    assert sys.modules["pyzed.sl"].Camera.instances[-1].opened is False
    with pytest.raises(ConnectionError, match="warmup"):  # a retry, not "already connected"
        cam.connect()
