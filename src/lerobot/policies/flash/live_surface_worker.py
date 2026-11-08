"""Camera-only host of the experiment frame session. Never opens a robot.

One latest RGB-D slot bounds acquisition backlog. Display uses the exact processed
RGB-D frame, never a newer preview image. Name/click selection starts a fresh session.
"""

from __future__ import annotations

import contextlib
import gc
import json
import signal
import sys
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread

import cv2
import numpy as np

from .depth_enhancement import make_depth_source, needs_stereo_pair, tracking_frame_with_depth_source
from .live_surface_io import (
    image_snapshot,
    new_selection,
    preview_depth_packet,
    seed_frame,
    tracker_depth_layout,
    write_snapshot,
)
from .live_surface_session import LIVE_METHOD, LiveSurfaceSession
from .perception import LatestRGBDFrameStore, RGBDFrame
from .surface_acceleration import SearchPool, compiled_kernels
from .surface_tracking import load_ufm
from .tracking_history import RollingHistory
from .vision_pipeline import VisionPipelineConfig


def _freeze_startup_heap():
    """Keep process-lifetime imports out of live GC scans.

    This worker serves one camera session and exits on Stop. Collect startup
    cycles first; later SAM sessions and per-frame objects still use normal GC.
    """
    gc.collect()
    gc.freeze()


def _prepare_measurements(frame, live, depth_source, stereo_geometry, mask_executor):
    """Join the same frame's SAM mask and optional learned depth before fitting."""
    if depth_source is None:
        return frame, None, 0.0
    depth_started = time.perf_counter()
    mask_future = mask_executor.submit(live.measure_masks, frame)
    try:
        tracking_frame = tracking_frame_with_depth_source(frame, depth_source, stereo_geometry)
        depth_ms = (time.perf_counter() - depth_started) * 1000
    finally:
        # Finish SAM before this session can be cleared or replaced.
        measured_masks = mask_future.result()
    return tracking_frame, measured_masks, depth_ms


def run(directory: Path, config: dict):
    import pyrealsense2 as rs

    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    from lerobot.gui.flash_group_overlay import SceneGroupOverlay
    from lerobot.gui.flash_live_scene import live_scene_packet
    from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

    stop, latest, commands = Event(), LatestRGBDFrameStore(), Queue(maxsize=8)
    camera = capture = live = adapter = pool = history = group_overlay = mask_executor = None
    model = None
    model_loaded = False
    frozen = None
    displayed_frame = None
    preview_frames = deque(maxlen=8)
    preview_lock = Lock()
    command_id = None
    last_id = -1
    sequence = 0
    paused = True
    errors = []
    camera_streams = None
    stereo_geometry = None
    depth_source_name = config.get("depth_source", "sensor")
    depth_source = None
    settings = config["pipeline_settings"]
    vision = VisionPipelineConfig.from_dict(settings["config"])
    cv2.setNumThreads(1)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    def state(status, **data):
        write_snapshot(
            directory / "state.json",
            {
                "status": status,
                "command_id": command_id,
                "engine": "experiment",
                "updated_s": time.perf_counter(),
                "camera_streams": camera_streams,
                **data,
            },
        )

    def controls():
        for line in sys.stdin:
            with contextlib.suppress(ValueError, Full):
                commands.put_nowait(json.loads(line))
        stop.set()

    try:
        if config.get("history_directory"):
            history = RollingHistory(
                config["history_directory"],
                seconds=config.get("history_seconds", 60),
            )
        state("opening_camera")
        serial = config["serial"]
        if not serial:
            cameras = RealSenseCamera.find_cameras()
            if len(cameras) != 1:
                raise ValueError("Select a camera serial")
            serial = cameras[0]["id"]
        camera = RealSenseCamera(
            RealSenseCameraConfig(
                serial_number_or_name=serial,
                width=config["width"],
                height=config["height"],
                fps=config["fps"],
                use_depth=True,
                # Decimation changes the aligned pixel grid; FLASH requires a
                # one-to-one depth/color correspondence for mask grounding.
                enable_decimation=False,
                enable_spatial=config.get("depth_processing", {}).get("spatial", False),
                enable_temporal=config.get("depth_processing", {}).get("temporal", False),
                enable_hole_filling=config.get("depth_processing", {}).get("hole_filling", False),
                enable_infrared=needs_stereo_pair(depth_source_name),
            )
        )
        if history:
            # The camera calls this for every aligned hardware grab. Tracking
            # may skip frames in its latest-only slot, but recording does not.
            recording_geometry = []

            def record_capture(rgb, depth, before_filters, stamp, sequence, left=None, right=None):
                if not recording_geometry:
                    profile = camera.rs_profile
                    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
                    intrinsics = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
                    unit = profile.get_device().first_depth_sensor().get_depth_scale()
                    recording_geometry.append((intrinsics, unit))
                intrinsics, unit = recording_geometry[0]
                depth_processing, revision = camera.last_applied_depth_processing
                history.submit_frame(
                    RGBDFrame(
                        sequence,
                        rgb,
                        tracker_depth_layout(rgb, depth),
                        intrinsics,
                        unit,
                        np.eye(4),
                        stamp,
                        time.perf_counter(),
                        "camera_native",
                        depth_processing,
                        revision,
                        tracker_depth_layout(rgb, before_filters) if any(depth_processing.values()) else None,
                        left,
                        right,
                    )
                )

            if needs_stereo_pair(depth_source_name):
                camera.aligned_frame_with_stereo_listener = (
                    lambda rgb, depth, before, left, right, stamp, sequence: record_capture(
                        rgb, depth, before, stamp, sequence, left, right
                    )
                )
            else:
                camera.aligned_frame_with_raw_listener = record_capture
        if depth_source_name != "sensor":
            # ONNX/PyTorch initialization can hold the GIL for seconds. Load
            # before the camera read thread starts, or acquisition can time out.
            state("loading_depth_model", depth_source=depth_source_name)
            depth_source = make_depth_source(depth_source_name)
            # Only one in-flight mask for the camera frame being processed.
            mask_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-live-mask")
        _freeze_startup_heap()
        camera.connect()
        intr = camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        scale = camera.rs_profile.get_device().first_depth_sensor().get_depth_scale()
        streams = {}
        for name, stream in (("color", rs.stream.color), ("depth", rs.stream.depth)):
            profile = camera.rs_profile.get_stream(stream).as_video_stream_profile()
            streams[name] = {
                "width": profile.width(),
                "height": profile.height(),
                "fps": profile.fps(),
                "format": str(profile.format()),
            }
        streams["depth_m_per_unit"] = scale
        if needs_stereo_pair(depth_source_name):
            left_profile = camera.rs_profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
            right_profile = camera.rs_profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
            left_intr = left_profile.get_intrinsics()
            left_k = np.array(
                [[left_intr.fx, 0, left_intr.ppx], [0, left_intr.fy, left_intr.ppy], [0, 0, 1]], float
            )
            to_color = left_profile.get_extrinsics_to(camera.rs_profile.get_stream(rs.stream.color))
            color_from_left = np.eye(4)
            color_from_left[:3, :3] = np.asarray(to_color.rotation).reshape(3, 3, order="F")
            color_from_left[:3, 3] = to_color.translation
            baseline = abs(left_profile.get_extrinsics_to(right_profile).translation[0])
            stereo_geometry = (left_k, color_from_left, baseline)
            streams["infrared"] = {
                "width": left_profile.width(),
                "height": left_profile.height(),
                "fps": left_profile.fps(),
                "left_intrinsics": left_k.tolist(),
                "color_from_left": color_from_left.tolist(),
                "baseline_m": baseline,
            }
        camera_streams = streams
        if history:
            write_snapshot(history.directory / "camera_streams.json", streams)

        def acquire():
            last_stamp, published = None, 0
            try:
                while not stop.is_set():
                    grabbed = camera.read_color_and_aligned_depth_with_timestamp(
                        timeout_ms=1000,
                        include_sequence=True,
                        include_processing=True,
                        include_before_filters=True,
                        include_infrared=stereo_geometry is not None,
                    )
                    rgb, depth, stamp, sequence, processing, revision, before_filters = grabbed[:7]
                    left, right = grabbed[7:] if stereo_geometry is not None else (None, None)
                    depth = tracker_depth_layout(rgb, depth)
                    if stamp == last_stamp:
                        stop.wait(0.005)
                        continue
                    last_stamp = stamp
                    frame = RGBDFrame(
                        sequence,
                        rgb,
                        depth,
                        k,
                        scale,
                        np.eye(4),
                        stamp,
                        time.perf_counter(),
                        "camera_native",
                        processing,
                        revision,
                        tracker_depth_layout(rgb, before_filters)
                        if before_filters is not None and processing and any(processing.values())
                        else None,
                        left,
                        right,
                    )
                    latest.publish(frame)
                    if time.perf_counter() - published >= 1 / config["preview_fps"]:
                        with preview_lock:
                            preview_frames.append(frame)
                        preview_depth = directory / "preview_depth.tmp"
                        preview_depth.write_bytes(preview_depth_packet(frame))
                        preview_depth.replace(directory / "preview_depth.bin")
                        write_snapshot(directory / "preview.json", image_snapshot(frame))
                        published = time.perf_counter()
            except Exception as exc:
                errors.append(str(exc))
                stop.set()

        capture = Thread(target=acquire, daemon=True, name="flash-live-capture")
        capture.start()
        Thread(target=controls, daemon=True, name="flash-live-controls").start()
        kernels = compiled_kernels() if vision.compiled_surface_fit else None
        pool = SearchPool(vision.shape.batch_query_workers) if vision.reuse_surface_search else None
        state("preview", depth_processing=camera.depth_processing, camera_streams=streams)
        while not stop.is_set():
            selected = None
            try:
                command = commands.get_nowait()
            except Empty:
                command = None
            if command:
                command_id = command["id"]
                try:
                    if command["action"] == "freeze":
                        frozen = seed_frame({}, None, latest.snapshot())
                        paused = True
                        state("frozen", frame=image_snapshot(frozen))
                    elif command["action"] == "clear":
                        if live:
                            sequence = max(sequence, live.mask_source.next_click_sequence)
                            live.close()
                        live, frozen, paused, group_overlay = None, None, True, None
                        state("preview")
                    elif command["action"] == "clear_history":
                        if history:
                            history.clear()
                    elif command["action"] == "depth_filters":
                        applied = camera.set_depth_processing(**command["depth_processing"])
                        state(
                            "filters_updated",
                            depth_processing=applied,
                            depth_filter_revision=camera.depth_filter_revision,
                        )
                    elif command["action"] == "select":
                        displayed = frozen
                        if command.get("point") is not None:
                            frame_id = command.get("frame_id")
                            if displayed is None or displayed.frame_id != frame_id:
                                displayed = (
                                    displayed_frame
                                    if displayed_frame and displayed_frame.frame_id == frame_id
                                    else None
                                )
                            if displayed is None:
                                with preview_lock:
                                    displayed = next(
                                        (f for f in reversed(preview_frames) if f.frame_id == frame_id), None
                                    )
                        selected = seed_frame(
                            command,
                            displayed,
                            latest.snapshot(),
                            history.directory if history else None,
                        )
                        paused = True
                        state("loading_models", frame=image_snapshot(selected))
                        if not model_loaded:
                            shared = settings["shared_matching"]
                            model = load_ufm() if shared is None or shared["backend"] == "ufm" else None
                            model_loaded = True
                        if adapter is None:
                            adapter = Sam3TrackByDetectionAdapter(
                                device="cuda", resolution=vision.sam_resolution
                            )
                        if live:
                            sequence = max(sequence, live.mask_source.next_click_sequence)
                            live.close()
                        source, sequence = new_selection(
                            adapter,
                            command.get("prompt", "").strip() or "selected object",
                            command.get("point"),
                            sequence,
                        )
                        source.detection_correction = vision.detection_correction.control(offline=False)
                        live = LiveSurfaceSession(settings, source, model, kernels, pool)
                        group_overlay = SceneGroupOverlay()
                        frozen, last_id, paused = None, -1, False
                    else:
                        raise ValueError("Unsupported control for this inspector")
                except Exception as exc:
                    paused = True
                    state("error", message=str(exc), depth_processing=camera.depth_processing)
                    traceback.print_exc()
            if paused:
                stop.wait(0.02)
                continue
            frame = selected if selected is not None else latest.snapshot()
            if frame is None or frame.frame_id <= last_id:
                stop.wait(0.005)
                continue
            try:
                started = time.perf_counter()
                tracking_frame, measured_masks, depth_ms = _prepare_measurements(
                    frame, live, depth_source, stereo_geometry, mask_executor
                )
                row, matches, masks = live.step(tracking_frame, measured_masks=measured_masks)
                render_started = time.perf_counter()
                packet = live_scene_packet(
                    tracking_frame,
                    live,
                    row,
                    masks,
                    matches,
                    group_overlay=group_overlay,
                    depth_processing=frame.depth_processing,
                    depth_source=depth_source_name,
                    rgb_format="jpeg",
                )
                temporary = directory / "scene.tmp"
                temporary.write_bytes(packet)
                temporary.replace(directory / "scene.bin")
                if history:
                    history.attach_result(packet)
                elapsed = time.perf_counter() - started
                state(
                    "tracking",
                    frame_id=frame.frame_id,
                    depth_processing=frame.depth_processing,
                    depth_source=depth_source_name,
                    depth_filter_revision=frame.depth_filter_revision,
                    method=LIVE_METHOD,
                    history={"dropped_frames": history.dropped, "error": history.error} if history else None,
                    timings={
                        "processed_fps": 1 / elapsed,
                        "processing_ms": row["processing_ms"],
                        "mask_ms": row["mask_ms"],
                        "depth_ms": depth_ms,
                        "image_matching_ms": row.get("scene_point_tracking_ms", 0.0),
                        "display_ms": (time.perf_counter() - render_started) * 1000,
                        "skipped_camera_frames": max(0, frame.frame_id - last_id - 1) if last_id >= 0 else 0,
                        "age_ms": (time.perf_counter() - frame.source_time_s) * 1000,
                    },
                )
                last_id = frame.frame_id
                displayed_frame = frame
            except Exception as exc:
                paused = True
                state("error", message=str(exc))
                traceback.print_exc()
        if errors:
            raise RuntimeError(errors[0])
    except Exception as exc:
        state("error", message=str(exc))
        raise
    finally:
        stop.set()
        if mask_executor:
            mask_executor.shutdown(wait=True)
        if live:
            live.close()
        if pool:
            pool.close()
        if capture:
            capture.join(timeout=2)
        if camera and camera.is_connected:
            camera.disconnect()
        if history:
            history.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.directory, json.loads((args.directory / "config.json").read_text()))


if __name__ == "__main__":
    main()
