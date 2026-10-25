"""Perception-only subprocess for the FLASH live surface debugger.

Camera capture has one latest-frame slot; inference never queues old images.
Click selection freezes a specific frame before accepting pixel coordinates.
No robot or motor API is imported. State files are atomic, bounded snapshots.
"""

from __future__ import annotations

import argparse
import base64
import json
import signal
import sys
import time
import traceback
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np

from .appearance import Sam3MaskSource
from .perception import LatestRGBDFrameStore, RGBDFrame
from .surface_correspondence import SurfaceMotionConfig
from .surface_grounding import ShapeConfig
from .surface_observation import SurfaceObservationConfig
from .surface_tracking import SurfaceTrackingSession, load_ufm


def write_snapshot(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


def image_snapshot(frame: RGBDFrame) -> dict:
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode the camera image")
    return {
        "frame_id": frame.frame_id,
        "captured_s": frame.source_time_s,
        "width": frame.rgb.shape[1],
        "height": frame.rgb.shape[0],
        "intrinsics": frame.intrinsics.tolist(),
        "depth_m_per_unit": frame.depth_m_per_unit,
        "image": "data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
    }


def seed_frame(command: dict, frozen: RGBDFrame | None, latest: RGBDFrame | None) -> RGBDFrame:
    """Never reinterpret a click on a later camera image."""
    if command.get("point") is not None:
        if frozen is None or command.get("frame_id") != frozen.frame_id:
            raise ValueError("Freeze the camera image again before selecting a point")
        x, y = command["point"]
        if not (0 <= x < frozen.rgb.shape[1] and 0 <= y < frozen.rgb.shape[0]):
            raise ValueError("The selected point is outside the camera image")
        return frozen
    frame = frozen if frozen is not None else latest
    if frame is None:
        raise ValueError("Wait for the camera image")
    return frame


def new_selection(adapter, label: str, point, sequence: int):
    """Reuse weights, explicitly forget the old instance, and advance click IDs.

    SAM's video reset preserves clicked masks for scrub/recovery; selecting a
    different object must clear those masks through its public control API.
    """
    sequence += 1
    adapter.set_control({"click_seq": sequence, "clicks": {"debug": []}})
    adapter.reset()
    sequence += 1
    source = Sam3MaskSource(
        {"object": label},
        camera_id="debug",
        seed_points_by_role={"object": tuple(point)} if point else None,
        click_sequence_start=sequence,
        adapter_factory=lambda **_: adapter,
    )
    return source, sequence


def run(directory: Path, config: dict) -> None:
    import pyrealsense2 as rs

    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

    stop = Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    cv2.setNumThreads(1)
    latest = LatestRGBDFrameStore()
    commands = Queue(maxsize=8)
    capture_error = []
    camera = None
    capture = None
    state_path = directory / "state.json"
    frozen = None
    source = session = adapter = matcher = None
    last_frame_id = -1
    command_id = None
    click_sequence = 0
    object_name = None
    tracking = False

    def state(status, **kwargs):
        write_snapshot(
            state_path,
            {
                "status": status,
                "command_id": command_id,
                "updated_s": time.perf_counter(),
                "object_name": object_name,
                **kwargs,
            },
        )

    def read_commands():
        for line in sys.stdin:
            commands.put(json.loads(line))
        stop.set()

    try:
        state("opening_camera")
        serial = config["serial"]
        if not serial:
            cameras = RealSenseCamera.find_cameras()
            if len(cameras) != 1:
                raise ValueError(f"Select a RealSense serial; found {[c['id'] for c in cameras]}")
            serial = cameras[0]["id"]
        camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=config["width"],
                height=config["height"],
                fps=config["fps"],
                use_depth=True,
                enable_decimation=False,
                enable_spatial=False,
                enable_temporal=False,
                enable_hole_filling=False,
            )
        )
        camera.connect()
        values = camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        k = np.array([[values.fx, 0, values.ppx], [0, values.fy, values.ppy], [0, 0, 1]], float)
        depth_scale = camera.rs_profile.get_device().first_depth_sensor().get_depth_scale()

        def capture_frames():
            frame_id = 0
            last_capture_s = None
            last_preview_s = 0
            try:
                while not stop.is_set():
                    rgb, depth, captured_s = camera.read_color_and_aligned_depth_with_timestamp(
                        timeout_ms=1000
                    )
                    if captured_s == last_capture_s:
                        stop.wait(1 / config["fps"])
                        continue
                    previous_capture = last_capture_s
                    last_capture_s = captured_s
                    frame = RGBDFrame(
                        frame_id,
                        rgb,
                        depth,
                        k,
                        depth_scale,
                        np.eye(4),
                        captured_s,
                        time.perf_counter(),
                        "camera_native",
                    )
                    latest.publish(frame)
                    frame_id += 1
                    if captured_s - last_preview_s >= 1 / config["preview_fps"]:
                        write_snapshot(
                            directory / "preview.json",
                            {
                                **image_snapshot(frame),
                                "camera_serial": serial,
                                "capture_fps": 1 / (captured_s - previous_capture)
                                if previous_capture
                                else None,
                            },
                        )
                        last_preview_s = captured_s
            except Exception as exc:
                capture_error.append(str(exc))
                stop.set()

        capture = Thread(target=capture_frames, daemon=True, name="surface-debug-capture")
        capture.start()
        Thread(target=read_commands, daemon=True, name="surface-debug-controls").start()
        state("preview", config=config, camera_serial=serial)
        while not stop.is_set():
            try:
                command = commands.get_nowait()
            except Empty:
                command = None
            selected = None
            if command:
                command_id = command.get("id")
                try:
                    if command["action"] == "freeze":
                        frozen = seed_frame({}, None, latest.snapshot())
                        tracking = False
                        state("frozen", frame=image_snapshot(frozen))
                    elif command["action"] == "clear":
                        tracking = False
                        frozen = source = session = None
                        state("preview")
                    elif command["action"] == "select":
                        selected = seed_frame(command, frozen, latest.snapshot())
                        tracking = False
                        state("loading_models", frame=image_snapshot(selected))
                        if matcher is None:
                            matcher = load_ufm()
                        if adapter is None:
                            adapter = Sam3TrackByDetectionAdapter(
                                device="cuda", resolution=config["sam_resolution"]
                            )
                        label = command.get("prompt", "").strip() or "selected object"
                        point = command.get("point")
                        source, click_sequence = new_selection(adapter, label, point, click_sequence)
                        object_name = label
                        session = SurfaceTrackingSession(
                            matcher,
                            observation_config=SurfaceObservationConfig(**config["observation"]),
                            shape_config=ShapeConfig(**config["shape"]),
                            motion_config=SurfaceMotionConfig(**config["motion"]),
                        )
                        frozen = None
                        last_frame_id = -1
                        tracking = True
                except Exception as exc:
                    state("error", message=str(exc))
                    traceback.print_exc()
            if not tracking:
                stop.wait(0.02)
                continue
            frame = selected if selected is not None else latest.snapshot()
            if frame is None or frame.frame_id == last_frame_id:
                stop.wait(0.002)
                continue
            begin = time.perf_counter()
            skipped = max(0, frame.frame_id - last_frame_id - 1) if last_frame_id >= 0 else 0
            last_frame_id = frame.frame_id
            try:
                mask = source.masks(frame).get("object", np.zeros(frame.depth.shape, bool))
                mask_ms = (time.perf_counter() - begin) * 1000
                result = session.update(frame.rgb, frame.metric_depth(), mask, k, frame.frame_id)
                contours, _ = cv2.findContours(
                    mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                elapsed = time.perf_counter() - begin
                state(
                    result.pop("status"),
                    **result,
                    frame=image_snapshot(frame),
                    contours=[c[:, 0].tolist() for c in contours],
                    timings={
                        "mask_ms": mask_ms,
                        "total_ms": elapsed * 1000,
                        "processed_fps": 1 / elapsed,
                        "skipped_camera_frames": skipped,
                        "age_at_publication_ms": (time.perf_counter() - frame.source_time_s) * 1000,
                    },
                )
            except Exception as exc:
                tracking = False
                state("error", message=str(exc), frame=image_snapshot(frame))
                traceback.print_exc()
        if capture_error:
            raise RuntimeError(capture_error[0])
    except Exception as exc:
        state("error", message=str(exc))
        raise
    finally:
        stop.set()
        if capture:
            capture.join(timeout=2)
        if camera and camera.is_connected:
            camera.disconnect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.directory, json.loads((args.directory / "config.json").read_text()))


if __name__ == "__main__":
    main()
