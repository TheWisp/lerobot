"""Camera-only host of the experiment frame session. Never opens a robot.

One latest RGB-D slot bounds acquisition backlog. Display uses the exact processed
RGB-D frame, never a newer preview image. Name/click selection starts a fresh session.
"""

from __future__ import annotations

import contextlib
import json
import signal
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2
import numpy as np

from .live_surface_io import image_snapshot, new_selection, seed_frame, write_snapshot
from .live_surface_session import LiveSurfaceSession
from .perception import LatestRGBDFrameStore, RGBDFrame
from .surface_acceleration import SearchPool, compiled_kernels
from .surface_tracking import load_ufm
from .vision_pipeline import VisionPipelineConfig


def run(directory: Path, config: dict):
    import pyrealsense2 as rs

    from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    from lerobot.gui.flash_group_overlay import GroupOverlay
    from lerobot.gui.flash_live_scene import live_scene_packet
    from lerobot.overlays.adapters import Sam3TrackByDetectionAdapter

    stop, latest, commands = Event(), LatestRGBDFrameStore(), Queue(maxsize=8)
    camera = capture = live = adapter = pool = None
    model = None
    model_loaded = False
    frozen = None
    command_id = None
    last_id = -1
    sequence = 0
    paused = True
    overlays = defaultdict(GroupOverlay)
    errors = []
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
                **data,
            },
        )

    def controls():
        for line in sys.stdin:
            with contextlib.suppress(ValueError, Full):
                commands.put_nowait(json.loads(line))
        stop.set()

    try:
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
                enable_decimation=False,
                enable_spatial=False,
                enable_temporal=False,
                enable_hole_filling=False,
            )
        )
        camera.connect()
        intr = camera.rs_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        k = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
        scale = camera.rs_profile.get_device().first_depth_sensor().get_depth_scale()

        def acquire():
            index, last_stamp, published = 0, None, 0
            try:
                while not stop.is_set():
                    rgb, depth, stamp = camera.read_color_and_aligned_depth_with_timestamp(timeout_ms=1000)
                    if stamp == last_stamp:
                        stop.wait(0.005)
                        continue
                    last_stamp = stamp
                    frame = RGBDFrame(
                        index, rgb, depth, k, scale, np.eye(4), stamp, time.perf_counter(), "camera_native"
                    )
                    latest.publish(frame)
                    index += 1
                    if time.perf_counter() - published >= 1 / config["preview_fps"]:
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
        state("preview")
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
                        live, frozen, paused = None, None, True
                        state("preview")
                    elif command["action"] == "select":
                        selected = seed_frame(command, frozen, latest.snapshot())
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
                        live = LiveSurfaceSession(settings, config["method"], source, model, kernels, pool)
                        overlays.clear()
                        frozen, last_id, paused = None, -1, False
                    else:
                        raise ValueError("Unsupported control for this inspector")
                except Exception as exc:
                    paused = True
                    state("error", message=str(exc))
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
                row, _, masks = live.step(frame)
                packet = live_scene_packet(frame, live, row, masks, overlays)
                temporary = directory / "scene.tmp"
                temporary.write_bytes(packet)
                temporary.replace(directory / "scene.bin")
                elapsed = time.perf_counter() - started
                state(
                    "tracking",
                    frame_id=frame.frame_id,
                    method=config["method"],
                    timings={
                        "processed_fps": 1 / elapsed,
                        "processing_ms": row["processing_ms"],
                        "mask_ms": row["mask_ms"],
                        "image_matching_ms": row.get("scene_point_tracking_ms", 0.0),
                        "display_ms": elapsed * 1000 - row["processing_ms"],
                        "skipped_camera_frames": max(0, frame.frame_id - last_id - 1) if last_id >= 0 else 0,
                        "age_ms": (time.perf_counter() - frame.source_time_s) * 1000,
                    },
                )
                last_id = frame.frame_id
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
        if live:
            live.close()
        if pool:
            pool.close()
        if capture:
            capture.join(timeout=2)
        if camera and camera.is_connected:
            camera.disconnect()


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    run(args.directory, json.loads((args.directory / "config.json").read_text()))


if __name__ == "__main__":
    main()
