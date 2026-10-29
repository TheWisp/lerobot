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
"""A Stereolabs ZED camera through the ZED SDK.

The SDK owns the stereo geometry: it fetches the unit's factory calibration,
refines it when the camera opens, rectifies both eyes and computes depth on the
GPU in the left eye's frame. This driver only moves frames: one grab thread
pulls the left eye, optionally the right eye and the depth map, and the control
loop peeks the latest set under one lock, the way the RealSense driver does.

Depth is taken from the SDK as ``DEPTH_U16_MM``: uint16 millimetres with 0
where there is no measurement, which is the dataset's depth contract as is.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..intrinsics import CameraIntrinsics
from ..utils import get_cv2_rotation
from .configuration_zed import RESOLUTIONS, ZEDCameraConfig

logger = logging.getLogger(__name__)

#: A stale frame is handed out with a warning until it is this many times
#: older than the peek's limit; past that the camera is treated as gone.
STALE_GIVE_UP = 4

INSTALL_HINT = (
    "pyzed, the ZED SDK's Python API, is not importable. It is not on PyPI: install the "
    "ZED SDK on this host (https://www.stereolabs.com/developers/release), then its wheel "
    "into this environment with `/usr/local/zed/get_python_api.py` or `uv pip install <wheel>`."
)


def _sl() -> Any:
    """The SDK module, imported on use so a host without it can still load this
    module, list camera types and read datasets recorded elsewhere."""
    try:
        import pyzed.sl as sl
    except ImportError as e:
        raise ImportError(INSTALL_HINT) from e
    return sl


def _intrinsics_of(cam_params: Any, width: int, height: int) -> CameraIntrinsics:
    """The SDK's rectified pinhole for one eye. Rectified images carry no
    distortion, which is what the zeros in ``disto`` say; they are kept so a
    reader sees the model named rather than assumed."""
    coeffs = tuple(float(c) for c in np.asarray(cam_params.disto).ravel())
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=float(cam_params.fx),
        fy=float(cam_params.fy),
        cx=float(cam_params.cx),
        cy=float(cam_params.cy),
        model="rectified_pinhole",
        coeffs=coeffs if any(coeffs) else (),
    )


class ZEDCamera(Camera):
    """Stereolabs ZED through the SDK, with depth in the left eye's frame.

    Reads mirror the other drivers: ``read_latest`` is the left eye,
    ``read_latest_right`` and ``read_latest_stereo`` the right eye and the pair
    (when ``stereo_split``), ``read_latest_depth`` the depth map, and
    ``read_latest_stereo_and_depth`` all three from one grab.
    """

    def __init__(self, config: ZEDCameraConfig):
        super().__init__(config)
        self.config = config
        self.serial_number: str | None = str(config.serial_number) if config.serial_number else None
        self.device_name: str | None = None
        self.color_mode = config.color_mode
        self.use_rgb = True
        self.use_depth = config.use_depth
        self.stereo_split = config.stereo_split
        self.warmup_s = config.warmup_s
        self.rotation: int | None = get_cv2_rotation(config.rotation)
        # The config names the frames as published; the sensor's frame, which
        # is what the SDK is asked for, is the transpose under a quarter turn.
        quarter = self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.capture_width: int | None = self.height if quarter else self.width
        self.capture_height: int | None = self.width if quarter else self.height
        self._last_stale_warning = 0.0
        #: The right eye's intrinsics, the left's being ``intrinsics``.
        self.intrinsics_right: CameraIntrinsics | None = None
        self.baseline_m: float | None = None

        self._zed: Any | None = None
        self._runtime: Any | None = None
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_right_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[np.uint16] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()
        #: How long the SDK's last grab blocked, and the whole last grab-thread
        #: cycle (grab, retrieves, conversion), for the stall diagnostics.
        self.last_grab_ms: float = 0.0
        self.last_cycle_ms: float = 0.0

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number or 'first'})"

    @property
    def is_connected(self) -> bool:
        return self._zed is not None and bool(self._zed.is_opened())

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """Every ZED the SDK sees, with its serial, model and state."""
        sl = _sl()
        found = []
        for dev in sl.Camera.get_device_list():
            found.append(
                {
                    "name": f"Stereolabs {dev.camera_model}",
                    "type": "ZED",
                    "id": str(dev.serial_number),
                    "state": str(dev.camera_state),
                    "path": str(getattr(dev, "path", "")),
                }
            )
        return found

    def _init_parameters(self, sl: Any) -> Any:
        init = sl.InitParameters()
        if self.capture_width is not None and self.capture_height is not None:
            init.camera_resolution = getattr(
                sl.RESOLUTION, RESOLUTIONS[(self.capture_width, self.capture_height)]
            )
        if self.fps is not None:
            init.camera_fps = int(self.fps)
        # No depth asked for, no depth computed: the neural modes load a model
        # and take the GPU, which a preview or a colour-only profile never needs.
        init.depth_mode = (
            getattr(sl.DEPTH_MODE, self.config.depth_mode.upper()) if self.use_depth else sl.DEPTH_MODE.NONE
        )
        init.coordinate_units = sl.UNIT.MILLIMETER
        init.depth_minimum_distance = float(self.config.depth_min_m * 1000.0)
        init.depth_maximum_distance = float(self.config.depth_max_m * 1000.0)
        # Orientation is this driver's ``rotation`` setting; the SDK's IMU flip
        # would silently rotate the frames under a fixed calibration.
        init.camera_image_flip = sl.FLIP_MODE.OFF
        init.sdk_verbose = 0
        if self.config.serial_number:
            init.set_from_serial_number(int(self.config.serial_number))
        return init

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        sl = _sl()
        zed = sl.Camera()
        err = zed.open(self._init_parameters(sl))
        if err != sl.ERROR_CODE.SUCCESS:
            raise ConnectionError(
                f"Failed to open {self}: {err}. Run `lerobot-find-cameras zed` to list the units the SDK sees."
            )
        self._zed = zed

        info = zed.get_camera_information()
        cfg = info.camera_configuration
        width, height = int(cfg.resolution.width), int(cfg.resolution.height)  # the sensor's frame
        if self.capture_width is not None and (self.capture_width, self.capture_height) != (width, height):
            zed.close()
            self._zed = None
            raise RuntimeError(
                f"{self} opened at {width}x{height}, not the {self.capture_width}x{self.capture_height} "
                "the config asks for."
            )
        self.capture_width, self.capture_height = width, height
        quarter = self.rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE)
        self.width, self.height = (height, width) if quarter else (width, height)  # as published
        if self.fps is None:
            self.fps = int(cfg.fps)
        self.serial_number = str(info.serial_number)
        self.device_name = f"Stereolabs {info.camera_model}"
        calib = cfg.calibration_parameters
        self.intrinsics = _intrinsics_of(calib.left_cam, width, height).rotated(self.rotation)
        self.intrinsics_right = _intrinsics_of(calib.right_cam, width, height).rotated(self.rotation)
        self.baseline_m = float(calib.get_camera_baseline()) / 1000.0

        runtime = sl.RuntimeParameters()
        runtime.enable_depth = bool(self.use_depth)
        # Holes stay holes: a filled pixel is a guess the dataset could not
        # tell from a measurement, and 0 is the only invalid marker the format
        # keeps.
        runtime.enable_fill_mode = bool(self.config.fill_holes)
        self._runtime = runtime

        self._start_read_thread()
        try:
            if warmup:
                deadline = time.time() + max(self.warmup_s, 1)
                while time.time() < deadline:
                    self.async_read(timeout_ms=max(self.warmup_s, 1) * 1000)
                    time.sleep(0.1)
                with self.frame_lock:
                    if self.latest_color_frame is None or (
                        self.use_depth and self.latest_depth_frame is None
                    ):
                        raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except Exception as e:
            # Nothing stays open behind a failed connect: a retry must be able
            # to open the unit rather than be told it is connected already.
            self._stop_read_thread()
            zed.close()
            self._zed = None
            self._runtime = None
            if isinstance(e, ConnectionError):
                raise
            raise ConnectionError(f"{self} failed to capture frames during warmup: {e}") from e
        logger.info(
            f"{self} connected: {self.device_name} {self.width}x{self.height}@{self.fps}, "
            f"depth {self.config.depth_mode if self.use_depth else 'off'}."
        )

    def _postprocess_image(self, bgra: NDArray[Any]) -> NDArray[Any]:
        """The SDK hands out BGRA; publish RGB (or BGR), rotated like the other drivers."""
        code = cv2.COLOR_BGRA2BGR if self.color_mode == ColorMode.BGR else cv2.COLOR_BGRA2RGB
        image = cv2.cvtColor(bgra, code)
        if self.rotation is not None:
            image = cv2.rotate(image, self.rotation)
        return image

    def _postprocess_depth(self, depth_mm: NDArray[np.uint16]) -> NDArray[np.uint16]:
        depth = np.ascontiguousarray(depth_mm, dtype=np.uint16).copy()
        if self.rotation is not None:
            depth = cv2.rotate(depth, self.rotation)
        return depth[..., np.newaxis]

    @property
    def depth_processing(self) -> dict[str, Any]:
        """What the SDK does to the depth map before this driver publishes it:
        the mode that computes it and whether unmeasured pixels are filled. A
        recording carries this next to the depth's intrinsics."""
        return {"depth_mode": self.config.depth_mode, "fill_holes": bool(self.config.fill_holes)}

    def _read_loop(self) -> None:
        """Grab on the SDK's own cadence and cache the latest set under one lock.

        Post: ``latest_color_frame`` (and the right eye and depth when enabled)
        are from the same grab as ``latest_timestamp``; a reader that takes the
        lock once sees one grab, never a mix of two.
        """
        sl = _sl()
        stop_event = self.stop_event
        zed, runtime = self._zed, self._runtime
        if stop_event is None or zed is None or runtime is None:
            raise RuntimeError(f"{self}: read loop started before connect.")
        left_mat, right_mat, depth_mat = sl.Mat(), sl.Mat(), sl.Mat()
        failures = 0
        while not stop_event.is_set():
            try:
                t_grab = time.perf_counter()
                err = zed.grab(runtime)
                self.last_grab_ms = (time.perf_counter() - t_grab) * 1e3
                if err == sl.ERROR_CODE.CAMERA_NOT_DETECTED:
                    raise DeviceNotConnectedError(f"{self}: camera not detected.")
                if err != sl.ERROR_CODE.SUCCESS:
                    raise RuntimeError(f"{self}: grab failed: {err}")
                zed.retrieve_image(left_mat, sl.VIEW.LEFT, sl.MEM.CPU)
                left = self._postprocess_image(left_mat.get_data())
                right = None
                if self.stereo_split:
                    zed.retrieve_image(right_mat, sl.VIEW.RIGHT, sl.MEM.CPU)
                    right = self._postprocess_image(right_mat.get_data())
                depth = None
                if self.use_depth:
                    zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH_U16_MM, sl.MEM.CPU)
                    depth = self._postprocess_depth(depth_mat.get_data())
                capture_time = time.perf_counter()
                self.last_cycle_ms = (capture_time - t_grab) * 1e3  # grab + retrieves + conversion
                with self.frame_lock:
                    self.latest_color_frame = left
                    self.latest_right_frame = right
                    self.latest_depth_frame = depth
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failures = 0
            except DeviceNotConnectedError:
                break
            except Exception as e:
                failures += 1
                if failures <= 10:
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.stop_event = None

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 200) -> NDArray[Any]:
        """The next left-eye frame. The SDK is grabbed from one thread only, so
        this waits on the grab thread rather than grabbing itself."""
        if color_mode is not None and color_mode != self.color_mode:
            raise ValueError(f"{self} publishes {self.color_mode}; per-read colour modes are not supported.")
        return self.async_read(timeout_ms=timeout_ms)

    def _async_read(self, timeout_ms: float, which: str) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")
        with self.frame_lock:
            frame = {
                "left": self.latest_color_frame,
                "right": self.latest_right_frame,
                "depth": self.latest_depth_frame,
            }[which]
            self.new_frame_event.clear()
        if frame is None:
            raise RuntimeError(f"Internal error: event set but no {which} frame available for {self}.")
        return frame

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        return self._async_read(timeout_ms, "left")

    @check_if_not_connected
    def async_read_depth(self, timeout_ms: float = 200) -> NDArray[np.uint16]:
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")
        return self._async_read(timeout_ms, "depth")

    def _latest_frames(
        self, max_age_ms: int
    ) -> tuple[NDArray[Any], NDArray[Any] | None, NDArray[Any] | None]:
        """Peek (left, right, depth) of the latest grab under one lock.

        Raises:
            RuntimeError: If the read thread is not running or nothing has been captured yet.
            TimeoutError: If the latest grab is older than ``max_age_ms``.
        """
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self.frame_lock:
            left, right, depth = self.latest_color_frame, self.latest_right_frame, self.latest_depth_frame
            timestamp = self.latest_timestamp
        if left is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms <= max_age_ms:
            return left, right, depth
        # The SDK's grab occasionally pauses for about half a second (seen on
        # a ZED-M at HD720 with neural depth, three of five recording starts,
        # no error returned). The control loop must neither wait on it nor
        # lose the episode over it: hand out the frame in hand, say so, and
        # give up only once the pause is well past what a hiccup takes.
        if age_ms > STALE_GIVE_UP * max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms, gave up at "
                f"{STALE_GIVE_UP}x); last grab {self.last_grab_ms:.0f} ms, last cycle {self.last_cycle_ms:.0f} ms."
            )
        now = time.perf_counter()
        if now - self._last_stale_warning > 1.0:
            self._last_stale_warning = now
            logger.warning(
                f"{self}: frame is {age_ms:.0f} ms old (limit {max_age_ms} ms); handing out the last one while "
                f"the grab catches up (last grab {self.last_grab_ms:.0f} ms, last cycle {self.last_cycle_ms:.0f} ms)."
            )
        return left, right, depth

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """The latest left-eye frame, without waiting."""
        return self._latest_frames(max_age_ms)[0]

    @check_if_not_connected
    def read_latest_right(self, max_age_ms: int = 500) -> NDArray[Any]:
        if not self.stereo_split:
            raise RuntimeError(f"{self} was not configured with stereo_split; there is no right eye.")
        right = self._latest_frames(max_age_ms)[1]
        if right is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        return right

    @check_if_not_connected
    def read_latest_stereo(self, max_age_ms: int = 500) -> tuple[NDArray[Any], NDArray[Any]]:
        """Both eyes of one grab, (left, right)."""
        if not self.stereo_split:
            raise RuntimeError(f"{self} was not configured with stereo_split; there is no right eye.")
        left, right, _ = self._latest_frames(max_age_ms)
        if right is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        return left, right

    @check_if_not_connected
    def read_latest_depth(self, max_age_ms: int = 500) -> NDArray[np.uint16]:
        """The latest depth map, (H, W, 1) uint16 millimetres, 0 where unmeasured."""
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")
        depth = self._latest_frames(max_age_ms)[2]
        if depth is None:
            raise RuntimeError(f"{self} has not captured any depth yet.")
        return depth

    @check_if_not_connected
    def read_latest_color_and_depth(self, max_age_ms: int = 500) -> tuple[NDArray[Any], NDArray[np.uint16]]:
        """The left eye and its depth from one grab."""
        if not self.use_depth:
            raise RuntimeError(f"{self}: cannot read depth — camera was configured with use_depth=False.")
        left, _, depth = self._latest_frames(max_age_ms)
        if depth is None:
            raise RuntimeError(f"{self} has not captured any depth yet.")
        return left, depth

    @check_if_not_connected
    def read_latest_stereo_and_depth(
        self, max_age_ms: int = 500
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[np.uint16]]:
        """Both eyes and the depth map of one grab, (left, right, depth)."""
        if not self.stereo_split or not self.use_depth:
            raise RuntimeError(f"{self}: needs stereo_split and use_depth.")
        left, right, depth = self._latest_frames(max_age_ms)
        if right is None or depth is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        return left, right, depth

    def disconnect(self) -> None:
        if self._zed is None and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )
        self._stop_read_thread()
        if self._zed is not None:
            self._zed.close()
            self._zed = None
        self._runtime = None
        logger.info(f"{self} disconnected.")
