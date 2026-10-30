"""Perception-only subprocess for the FLASH live surface debugger.

Camera capture has one latest-frame slot; perception prepares at most one frame ahead.
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
from threading import Event, Lock, Thread

import cv2
import numpy as np

from .appearance import Sam3MaskSource
from .causal_service import CausalTrackingConfig, build_causal_pose_service
from .perception import LatestPoseStore, LatestRGBDFrameStore, ObjectReference, ReferenceSnippet, RGBDFrame
from .perception_step import PerceptionPipeline, PerceptionResult, PerceptionRunner
from .surface_tracking import load_ufm
from .vision_pipeline import VisionPipelineConfig, build_vision_backend

CAMERA_READ_TIMEOUT_MS = 1000


class RGBDTrace:
    """Bounded-duration evidence for replaying one live tracking failure offline."""

    def __init__(self, duration_s: float, config: dict):
        root = Path.home() / ".cache" / "huggingface" / "lerobot" / "flash_debug_traces"
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / f"trace-{time.strftime('%Y%m%d-%H%M%S')}-{time.time_ns() % 1_000_000_000:09d}"
        self.path.mkdir()
        for name in ("rgb", "depth", "masks"):
            (self.path / name).mkdir()
        (self.path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        self.end_s = time.monotonic() + duration_s
        self.frames = 0
        self.complete = False

    def save(self, frame: RGBDFrame, masks: dict[str, np.ndarray], details: dict[str, dict]) -> float:
        start = time.perf_counter()
        stem = f"{frame.frame_id:06d}"
        compression = [cv2.IMWRITE_PNG_COMPRESSION, 1]
        if not cv2.imwrite(
            str(self.path / "rgb" / f"{stem}.png"), cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR), compression
        ) or not cv2.imwrite(str(self.path / "depth" / f"{stem}.png"), frame.depth, compression):
            raise RuntimeError("could not save aligned RGB-D diagnostic frame")
        for role, mask in masks.items():
            role_dir = self.path / "masks" / role
            role_dir.mkdir(exist_ok=True)
            if not cv2.imwrite(str(role_dir / f"{stem}.png"), mask.astype(np.uint8) * 255, compression):
                raise RuntimeError(f"could not save {role} diagnostic mask")
        row = {
            "frame_id": frame.frame_id,
            "source_time_s": frame.source_time_s,
            "intrinsics": frame.intrinsics.tolist(),
            "depth_m_per_unit": frame.depth_m_per_unit,
            "calibration_version": frame.calibration_version,
            "objects": {
                role: {
                    key: value
                    for key, value in detail.items()
                    if key
                    in (
                        "status",
                        "accepted",
                        "camera_from_reference",
                        "diagnostics",
                        "group_evidence",
                        "group_estimator",
                        "group_frame_id",
                        "group_pose",
                    )
                }
                for role, detail in details.items()
            },
        }
        with (self.path / "frames.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.frames += 1
        return (time.perf_counter() - start) * 1000

    def finish(self) -> None:
        if not self.complete:
            (self.path / "complete.json").write_text(json.dumps({"frames": self.frames}) + "\n")
            self.complete = True

    def status(self) -> dict:
        return {"path": str(self.path), "frames": self.frames, "complete": self.complete}


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


def debug_reference(role: str, frame: RGBDFrame) -> ObjectReference:
    return ObjectReference(
        role,
        f"preview:{frame.frame_id}",
        (
            ReferenceSnippet(
                "live-preview",
                0,
                frame.frame_id,
                role,
                np.eye(4),
                "unrecorded",
                frame.calibration_version,
            ),
        ),
    )


def make_debug_pipeline(config: VisionPipelineConfig, source, matcher, frame: RGBDFrame):
    backend = build_vision_backend(config, source, matcher=matcher, local_reference=True)
    return PerceptionPipeline(backend, (debug_reference("object", frame),), require_masks=True)


def replace_debug_pipeline(previous, config, source, matcher, frame):
    """Both the first selection and a replacement get a fresh reference owner."""
    if previous is not None:
        previous.close()
    pipeline = make_debug_pipeline(config, source, matcher, frame)
    return pipeline, PerceptionRunner(pipeline)


class DebugGroupFallback:
    """The live scene estimator, fed every camera frame and delayed vision anchors."""

    def __init__(self, config: CausalTrackingConfig):
        self.config = config
        self.store = LatestPoseStore()
        self.service = None
        self._lock = Lock()
        self._last_frame_id = -1
        self._published: dict[str, tuple] = {}

    def _remember(self, service, estimate) -> None:
        frame_id = estimate.source_frame_id
        if frame_id is None:
            return
        frame = next((item for item in reversed(service.history) if item.frame_id == frame_id), None)
        if frame is None:
            return
        with self._lock:
            previous = self._published.get(estimate.role)
            if previous is None or frame_id >= previous[0].source_frame_id:
                self._published[estimate.role] = estimate, frame

    def submit_frame(self, frame: RGBDFrame) -> None:
        with self._lock:
            if self.service is not None and frame.frame_id > self._last_frame_id:
                self.service.submit_frame(frame)
                self._last_frame_id = frame.frame_id

    def submit_result(self, frame: RGBDFrame, result: PerceptionResult) -> None:
        if self.service is None:
            # The click/name reference is initialized by this very result.
            # Seed in source order before the capture thread starts forwarding.
            service = build_causal_pose_service(
                self.store,
                self.config,
                on_layer=lambda estimate: self._remember(service, estimate),
            )
            service.process_offline(frame, result.estimates, result.masks, proposals=result.proposals)
            service.start()
            with self._lock:
                self._last_frame_id = frame.frame_id
                self.service = service
        else:
            self.service.submit_anchor(
                frame.frame_id,
                result.estimates,
                result.masks,
                source_time_s=result.source_time_s,
                calibration_version=result.calibration_version,
                proposals=result.proposals,
            )
        if self.service.error is not None:
            raise RuntimeError("scene/group tracking failed") from self.service.error

    def estimates(self) -> dict:
        return self.store.snapshot()[1]

    def latest_frame(self) -> tuple[RGBDFrame | None, dict]:
        """One image and only estimates that actually came from that image."""
        with self._lock:
            if not self._published:
                return None, {}
            frame_id = max(item.source_frame_id for item, _ in self._published.values())
            selected = {
                role: item for role, item in self._published.items() if item[0].source_frame_id == frame_id
            }
            frame = next(iter(selected.values()))[1]
            return frame, {role: item for role, (item, _) in selected.items()}

    def close(self) -> None:
        with self._lock:
            service, self.service = self.service, None
        if service is not None:
            service.close()
        self.store = LatestPoseStore()
        self._last_frame_id = -1
        with self._lock:
            self._published.clear()


def group_display(detail: dict, estimate, frame: RGBDFrame, reference_detail: dict | None = None) -> dict:
    """Expose scene choice and project it only on its own source RGB-D frame."""
    if estimate is None:
        return {"group_evidence": "unavailable"}
    display = {
        "group_evidence": estimate.evidence.value,
        "group_estimator": estimate.estimator,
        "group_frame_id": estimate.source_frame_id,
        "group_pose": estimate.pose.tolist() if estimate.pose is not None else None,
        "group_current_frame": estimate.source_frame_id == frame.frame_id,
    }
    if detail.get("reference_version") != estimate.reference_version:
        return display
    if not display["group_current_frame"] or estimate.pose is None:
        return display
    reference_detail = (
        detail if detail.get("accepted") is True and "camera_from_reference" in detail else reference_detail
    )
    if reference_detail is None or "camera_from_object" not in reference_detail:
        return display
    direct_ref = np.asarray(reference_detail["camera_from_reference"], dtype=float)
    direct_object = np.asarray(reference_detail["camera_from_object"], dtype=float)
    reference_from_object = np.linalg.inv(direct_ref) @ direct_object
    camera_from_reference = (
        np.linalg.inv(frame.base_from_camera) @ estimate.pose @ np.linalg.inv(reference_from_object)
    )
    delta = camera_from_reference @ np.linalg.inv(direct_ref)
    display["group_camera_from_reference"] = camera_from_reference.tolist()
    for key in ("reference_m", "anchors_m", "axes_m"):
        if key not in reference_detail:
            continue
        points = np.asarray(reference_detail[key], dtype=float)
        if points.size == 0:
            continue
        moved = points @ delta[:3, :3].T + delta[:3, 3]
        if not np.isfinite(moved).all() or np.any(moved[:, 2] <= 0):
            continue
        projected = moved @ frame.intrinsics.T
        projected = projected[:, :2] / projected[:, 2:3]
        display[f"group_{key}"] = moved.tolist()
        display[f"group_{key.removesuffix('_m')}_px"] = projected.tolist()
    return display


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
    runner = None
    last_acquired_frame_id = -1
    last_publication_s = None
    last_frame_id = -1
    command_id = None
    click_sequence = 0
    object_name = None
    tracking = False
    tracked_names: dict[str, str] = {}
    last_objects: dict[str, dict] = {}
    reference_details: dict[str, dict] = {}
    prepared_mask_ms: float | None = None
    trace: RGBDTrace | None = None
    group = (
        DebugGroupFallback(CausalTrackingConfig.from_config(config["causal_tracking"]))
        if config.get("group_fallback")
        else None
    )

    def state(status, **kwargs):
        write_snapshot(
            state_path,
            {
                "status": status,
                "command_id": command_id,
                "updated_s": time.perf_counter(),
                "object_name": object_name,
                "tracks": [
                    {
                        "role": role,
                        "name": name,
                        "status": last_objects.get(role, {}).get("status"),
                        "accepted": last_objects.get(role, {}).get("accepted"),
                        "group_evidence": last_objects.get(role, {}).get("group_evidence"),
                        "group_estimator": last_objects.get(role, {}).get("group_estimator"),
                    }
                    for role, name in tracked_names.items()
                ],
                "trace": trace.status() if trace is not None else None,
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
                        timeout_ms=CAMERA_READ_TIMEOUT_MS
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
                    if group is not None:
                        group.submit_frame(frame)
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
                if runner is not None:
                    drained = runner.drain()
                    if drained is not None:
                        last_frame_id = drained[0].frame_id
                command_id = command.get("id")
                try:
                    if command["action"] in {"freeze", "clear", "select"} and trace is not None:
                        trace.finish()
                    if command["action"] == "freeze":
                        if group is not None:
                            group.close()
                        frozen = seed_frame({}, None, latest.snapshot())
                        tracking = False
                        last_publication_s = None
                        # A newly selected role must join on a frame later than the
                        # last update of every existing rigid reference.
                        previous = max(last_frame_id, source.last_frame_id or -1) if source else last_frame_id
                        deadline = time.monotonic() + 2 * CAMERA_READ_TIMEOUT_MS / 1000
                        while tracked_names and frozen.frame_id <= previous and not stop.is_set():
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Camera produced no newer frame for object selection")
                            stop.wait(0.01)
                            frozen = seed_frame({}, None, latest.snapshot())
                        state("frozen", frame=image_snapshot(frozen))
                    elif command["action"] == "clear":
                        if group is not None:
                            group.close()
                        tracking = False
                        if source is not None:
                            click_sequence = max(click_sequence, source.next_click_sequence)
                        if runner is not None:
                            runner.close()
                        runner = None
                        frozen = source = session = None
                        tracked_names.clear()
                        last_objects.clear()
                        reference_details.clear()
                        state("preview")
                    elif command["action"] == "record":
                        if not tracking:
                            raise ValueError("Select an object before recording a diagnostic trace")
                        if trace is not None and not trace.complete:
                            raise ValueError("A diagnostic trace is already recording")
                        trace = RGBDTrace(command["duration_s"], config)
                        state("tracking", message=f"Recording aligned RGB-D and masks to {trace.path}")
                    elif command["action"] == "select":
                        selected = seed_frame(command, frozen, latest.snapshot())
                        tracking = False
                        state("loading_models", frame=image_snapshot(selected))
                        if matcher is None:
                            matcher = load_ufm()
                        if adapter is None:
                            adapter = Sam3TrackByDetectionAdapter(
                                device="cuda", resolution=config["vision_pipeline"]["sam_resolution"]
                            )
                            adapter.set_control(
                                {
                                    "detection_correction": VisionPipelineConfig.from_dict(
                                        config["vision_pipeline"]
                                    ).detection_correction.control(offline=False)
                                }
                            )
                        label = command.get("prompt", "").strip() or "selected object"
                        point = command.get("point")
                        prepared_mask_ms = None
                        if point is not None and source is not None and tracked_names:
                            role = f"object_{len(tracked_names) + 1}"
                            used_labels = set(tracked_names.values())
                            if label in used_labels:
                                base = label
                                suffix = 2
                                while label in used_labels:
                                    label = f"{base} ({suffix})"
                                    suffix += 1
                            if len(tracked_names) >= adapter.MAX_OBJECTS:
                                raise ValueError(f"At most {adapter.MAX_OBJECTS} objects can be tracked")
                            added, prepared_mask_ms = source.add_clicked_role(role, label, point, selected)
                            if not added:
                                raise ValueError(
                                    "Click did not create a new mask; choose an untracked object"
                                )
                            session.add_reference(debug_reference(role, selected))
                            tracked_names[role] = label
                        else:
                            if group is not None:
                                group.close()
                            if source is not None:
                                click_sequence = max(click_sequence, source.next_click_sequence)
                            source, click_sequence = new_selection(adapter, label, point, click_sequence)
                            object_name = label
                            vision_config = VisionPipelineConfig.from_dict(config["vision_pipeline"])
                            session, runner = replace_debug_pipeline(
                                runner, vision_config, source, matcher, selected
                            )
                            last_acquired_frame_id = -1
                            last_publication_s = None
                            tracked_names.clear()
                            tracked_names["object"] = label
                            last_objects.clear()
                            reference_details.clear()
                            last_frame_id = -1
                        frozen = None
                        tracking = True
                except Exception as exc:
                    state("error", message=str(exc))
                    traceback.print_exc()
            if not tracking:
                stop.wait(0.02)
                continue
            selected_frames = [selected] if selected is not None else []

            def take_frame(selected_frames=selected_frames):
                nonlocal last_acquired_frame_id
                if stop.is_set() or (not selected_frames and not commands.empty()):
                    return None
                frame = selected_frames.pop() if selected_frames else latest.snapshot()
                if frame is None or frame.frame_id <= last_acquired_frame_id:
                    return None
                last_acquired_frame_id = frame.frame_id
                return frame

            begin = time.perf_counter()
            frame = None
            try:
                item = runner.process_next(take_frame)
                if item is None:
                    stop.wait(0.002)
                    continue
                frame, output = item
                if group is not None:
                    group.submit_result(frame, output)
                group_estimates = group.estimates() if group is not None else {}
                skipped = max(0, frame.frame_id - last_frame_id - 1) if last_frame_id >= 0 else 0
                last_frame_id = frame.frame_id
                mask_ms = prepared_mask_ms if prepared_mask_ms is not None else session.backend.last_mask_ms
                prepared_mask_ms = None
                objects = {}
                for role, name in tracked_names.items():
                    mask = output.masks.get(role, np.zeros(frame.depth.shape, bool))
                    result = dict(output.details[role])
                    if result.get("accepted") is True and "camera_from_reference" in result:
                        reference_details[role] = result
                    contours, _ = cv2.findContours(
                        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    objects[role] = {
                        **result,
                        **(
                            group_display(
                                result, group_estimates.get(role), frame, reference_details.get(role)
                            )
                            if group
                            else {}
                        ),
                        "name": name,
                        "contours": [c[:, 0].tolist() for c in contours],
                    }
                last_objects = objects
                group_frame, group_at_frame = group.latest_frame() if group is not None else (None, {})
                group_objects = (
                    {
                        role: {
                            "name": tracked_names[role],
                            **group_display(
                                output.details[role], estimate, group_frame, reference_details.get(role)
                            ),
                        }
                        for role, estimate in group_at_frame.items()
                        if role in tracked_names
                    }
                    if group_frame is not None
                    else {}
                )
                trace_ms = None
                if trace is not None and not trace.complete:
                    if time.monotonic() < trace.end_s:
                        trace_ms = trace.save(frame, output.masks, objects)
                    else:
                        trace.finish()
                publication_s = time.perf_counter()
                elapsed = publication_s - begin
                interval = elapsed if last_publication_s is None else publication_s - last_publication_s
                last_publication_s = publication_s
                state(
                    "tracking",
                    objects=objects,
                    frame=image_snapshot(frame),
                    group_frame=image_snapshot(group_frame) if group_frame is not None else None,
                    group_objects=group_objects,
                    timings={
                        "mask_ms": mask_ms,
                        "total_ms": elapsed * 1000,
                        "processed_fps": 1 / interval,
                        "publication_interval_ms": interval * 1000,
                        "skipped_camera_frames": skipped,
                        "age_at_publication_ms": (time.perf_counter() - frame.source_time_s) * 1000,
                        "trace_write_ms": trace_ms,
                    },
                )
            except Exception as exc:
                tracking = False
                if group is not None:
                    group.close()
                runner.close(commit_pending=False)
                runner = source = session = None
                tracked_names.clear()
                state("error", message=str(exc), **({"frame": image_snapshot(frame)} if frame else {}))
                traceback.print_exc()
        if capture_error:
            raise RuntimeError(capture_error[0])
    except Exception as exc:
        state("error", message=str(exc))
        raise
    finally:
        stop.set()
        if trace is not None:
            trace.finish()
        if runner is not None:
            runner.close(commit_pending=False)
        if group is not None:
            group.close()
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
