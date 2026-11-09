"""Rolling camera RGB-D with optional tracking results keyed by capture frame.

Capture never waits for the tracker. The writer compresses measured frames on
its own thread; a later result adds metadata without replacing those pixels.
"""

# hardcode-lint: ignore-file - FLASH history exports one camera under its fixed episode feature schema.

from __future__ import annotations

import fcntl
import json
import os
import shutil
import struct
import time
import uuid
import zlib
from contextlib import contextmanager, suppress
from dataclasses import replace
from pathlib import Path
from queue import Full, Queue
from threading import Event, Thread

import numpy as np
import zstandard as zstd

from .live_surface_io import write_snapshot


def history_root():
    return Path(
        os.environ.get("LEROBOT_FLASH_HISTORY_DIR", "~/.cache/flash-tracker-restart/live-history")
    ).expanduser()


@contextmanager
def locked(directory, *, exclusive=False):
    with (directory / "lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def read_index(directory):
    path = Path(directory) / "index.json"
    return json.loads(path.read_text()) if path.exists() else {"frames": [], "dropped_frames": 0}


def read_packet(directory, frame):
    directory = Path(directory)
    with locked(directory):
        if not any(row["frame"] == frame for row in read_index(directory)["frames"]):
            raise FileNotFoundError("Frame is no longer retained")
        base = directory / f"{frame:010d}"
        modern = base.with_suffix(".bin.zst")
        packet = (
            zstd.ZstdDecompressor().decompress(modern.read_bytes())
            if modern.exists()
            else zlib.decompress(base.with_suffix(".bin.z").read_bytes())
        )
        processed = base.with_suffix(".processed.json")
        overlay = base.with_suffix(".overlay.json")
        if processed.exists() or overlay.exists():
            n = struct.unpack("<I", packet[:4])[0]
            header = json.loads(packet[4 : 4 + n])
            if processed.exists():
                update = json.loads(processed.read_text())
                header.update(update)
            if overlay.exists():
                update = json.loads(overlay.read_text())
                if update.get("depth_source"):
                    header["tracking_depth_source"] = update["depth_source"]
                header.update(
                    {
                        key: value
                        for key, value in update.items()
                        if key
                        not in (
                            "camera",
                            "depth_bytes",
                            "depth_before_filter_bytes",
                            "infrared_shape",
                            "infrared_bytes",
                            "source_frame_id",
                            "base_from_camera",
                            "available_time_s",
                            "calibration_version",
                            "rgb_format",
                            "depth_format",
                            "depth_m_per_unit",
                            "depth_before_filter_m_per_unit",
                            "depth_processing",
                            "depth_filter_revision",
                            "depth_source",
                        )
                    }
                )
            model_file = base.with_suffix(".model_depth.zst")
            model_depth = (
                zstd.ZstdDecompressor().decompress(model_file.read_bytes()) if model_file.is_file() else b""
            )
            header["model_depth_bytes"] = len(model_depth)
            old_payload = packet[4 + n :]
            insert = (
                header["depth_bytes"]
                + header.get("depth_before_filter_bytes", 0)
                + 2 * header.get("infrared_bytes", 0)
            )
            metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
            packet = (
                struct.pack("<I", len(metadata))
                + metadata
                + old_payload[:insert]
                + model_depth
                + old_payload[insert:]
            )
        return packet


def read_processed_frame(directory, frame):
    """Exact processed evidence for episode playback, without duplicating RGB video."""
    directory = Path(directory)
    with locked(directory):
        row = next((r for r in read_index(directory)["frames"] if r["frame"] == frame), None)
        if row is None or "processed_ready_time_s" not in row:
            raise FileNotFoundError("Processed frame is unavailable")
        base = directory / f"{frame:010d}"
        metadata = json.loads(base.with_suffix(".processed.json").read_text())
        model_file = base.with_suffix(".model_depth.zst")
        if metadata["processed_depth_source"] != "sensor" and not model_file.is_file():
            raise FileNotFoundError("Processed model depth is unavailable")
        depth = zstd.ZstdDecompressor().decompress(model_file.read_bytes()) if model_file.is_file() else b""
        metadata["source_frame_id"] = frame
        metadata["model_depth_bytes"] = len(depth)
        header = json.dumps(metadata, allow_nan=False, separators=(",", ":")).encode()
        return struct.pack("<I", len(header)) + header + depth


def capture_packet(frame):
    """Lossless camera input in the same scene envelope as processed packets."""
    depth = np.ascontiguousarray(frame.depth)
    before = (
        np.ascontiguousarray(frame.depth_before_filters) if frame.depth_before_filters is not None else None
    )
    left = np.ascontiguousarray(frame.infrared_left) if frame.infrared_left is not None else None
    right = np.ascontiguousarray(frame.infrared_right) if frame.infrared_right is not None else None
    rgb = np.ascontiguousarray(frame.rgb)
    header = {
        "frame": frame.frame_id,
        "time_s": frame.source_time_s,
        "source_frame_id": frame.frame_id,
        "available_time_s": frame.available_time_s,
        "base_from_camera": frame.base_from_camera.tolist(),
        "calibration_version": frame.calibration_version,
        "camera": {"width": rgb.shape[1], "height": rgb.shape[0], "intrinsics": frame.intrinsics.tolist()},
        "depth_bytes": depth.nbytes,
        "depth_before_filter_bytes": before.nbytes if before is not None else 0,
        "infrared_shape": list(left.shape) if left is not None else None,
        "infrared_bytes": left.nbytes if left is not None else 0,
        "depth_format": "sensor_u16" if depth.dtype == np.uint16 else "metric_f32",
        "depth_m_per_unit": frame.depth_m_per_unit,
        "depth_processing": frame.depth_processing,
        "depth_source": "sensor",
        "depth_filter_revision": frame.depth_filter_revision,
        "rgb_format": "raw_rgb8",
        "objects": {},
        "scene_group_tracks": [],
        "masks_rle": {},
        "result": None,
    }
    metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return (
        struct.pack("<I", len(metadata))
        + metadata
        + depth.tobytes()
        + (before.tobytes() if before is not None else b"")
        + (left.tobytes() + right.tobytes() if left is not None else b"")
        + rgb.tobytes()
    )


def decode_record(packet):
    """Restore measured inputs and the unchanged pipeline result for inspection."""
    import cv2
    import numpy as np

    from .perception import RGBDFrame

    n = struct.unpack("<I", packet[:4])[0]
    header = json.loads(packet[4 : 4 + n])
    camera = header["camera"]
    start = 4 + n
    depth_dtype = "<u2" if header.get("depth_format") == "sensor_u16" else "<f4"
    depth = (
        np.frombuffer(packet[start : start + header["depth_bytes"]], depth_dtype)
        .reshape(camera["height"], camera["width"])
        .copy()
    )
    before_start = start + header["depth_bytes"]
    before_bytes = header.get("depth_before_filter_bytes", 0)
    before = (
        np.frombuffer(packet[before_start : before_start + before_bytes], "<u2")
        .reshape(camera["height"], camera["width"])
        .copy()
        if before_bytes
        else None
    )
    ir_start = before_start + before_bytes
    ir_bytes = header.get("infrared_bytes", 0)
    infrared_shape = header.get("infrared_shape")
    left = (
        (np.frombuffer(packet[ir_start : ir_start + ir_bytes], np.uint8).reshape(infrared_shape).copy())
        if ir_bytes
        else None
    )
    right = (
        (
            np.frombuffer(packet[ir_start + ir_bytes : ir_start + 2 * ir_bytes], np.uint8)
            .reshape(infrared_shape)
            .copy()
        )
        if ir_bytes
        else None
    )
    rgb_bytes = packet[ir_start + 2 * ir_bytes + header.get("model_depth_bytes", 0) :]
    if header.get("rgb_format") == "raw_rgb8":
        rgb = np.frombuffer(rgb_bytes, np.uint8).reshape(camera["height"], camera["width"], 3).copy()
    else:
        rgb = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(rgb_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
    frame = RGBDFrame(
        header["source_frame_id"],
        rgb,
        depth,
        np.asarray(camera["intrinsics"]),
        header.get("depth_m_per_unit", 1.0),
        np.asarray(header["base_from_camera"]),
        header["time_s"],
        header["available_time_s"],
        header["calibration_version"],
        header.get("depth_processing"),
        header.get("depth_filter_revision", 0),
        before,
        left,
        right,
    )
    masks = {}
    for role, runs in header["masks_rle"].items():
        mask = np.zeros(depth.size, bool)
        for first, length in runs:
            mask[first : first + length] = True
        masks[role] = mask.reshape(depth.shape)
    return frame, masks, header.get("result"), header.get("image_matches")


def save_window(directory, destination_root):
    """Pin exactly the advertised frames. The writer cannot prune during linking."""
    directory, root = Path(directory), Path(destination_root)
    identity = "saved-" + uuid.uuid4().hex
    temporary = root / ("." + identity)
    final = root / identity
    temporary.mkdir(parents=True)
    try:
        with locked(directory):
            index = read_index(directory)
            if not index["frames"]:
                raise ValueError("No recorded camera frames yet")
            for row in index["frames"]:
                base = f"{row['frame']:010d}"
                for suffix in (".bin.zst", ".bin.z", ".processed.json", ".overlay.json", ".model_depth.zst"):
                    name = base + suffix
                    if (directory / name).exists():
                        os.link(directory / name, temporary / name)
            shutil.copy2(directory / "config.json", temporary / "config.json")
            if (directory / "camera_streams.json").is_file():
                shutil.copy2(directory / "camera_streams.json", temporary / "camera_streams.json")
            write_snapshot(temporary / "index.json", {**index, "saved_at": time.time()})
        temporary.rename(final)
    except Exception:
        # Only this incomplete export; existing recordings are never touched.
        # safe-destruct: remove only the temporary directory created for this failed save.
        shutil.rmtree(temporary)
        raise
    return identity


def export_episode(directory):
    """Make a saved RGB-D window an ordinary one-episode LeRobot dataset.

    The temporary rolling packet files remain as exact reprocessing evidence;
    the episode supplies video playback and lossless sensor-depth images.
    """
    from lerobot.configs.video import RGBEncoderConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    directory = Path(directory)
    final = directory / "episode"
    with (directory / "episode-export.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if final.is_dir():
            return final
        rows = read_index(directory)["frames"]
        if not rows:
            raise ValueError("No saved camera frames to export")
        first, _, _, _ = decode_record(read_packet(directory, rows[0]["frame"]))
        h, w = first.depth.shape
        times = np.array([row["time_s"] for row in rows], np.float64)
        elapsed = np.diff(times)
        positive = elapsed[elapsed > 0]
        fps = max(1, min(120, round(1 / float(np.median(positive))))) if positive.size else 30
        features = {
            "observation.images.camera": {
                "dtype": "video",
                "shape": (h, w, 3),
                "names": ["height", "width", "channels"],
            },
            "observation.images.camera_depth": {
                "dtype": "image",
                "shape": (h, w, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            },
            "observation.camera.source_frame_id": {"dtype": "int64", "shape": (1,), "names": None},
            "observation.camera.source_time_s": {"dtype": "float64", "shape": (1,), "names": None},
            "observation.camera.available_time_s": {"dtype": "float64", "shape": (1,), "names": None},
            "observation.camera.depth_m_per_unit": {"dtype": "float64", "shape": (1,), "names": None},
            "observation.camera.intrinsics": {"dtype": "float64", "shape": (9,), "names": None},
            "observation.camera.base_from_camera": {"dtype": "float64", "shape": (16,), "names": None},
        }
        has_before = any(row.get("has_before_filters", False) for row in rows)
        has_infrared = any(row.get("has_infrared", False) for row in rows)
        has_model_depth = any(
            row.get("processed_depth_source", row.get("tracking_depth_source", "sensor")) != "sensor"
            for row in rows
        )
        if has_before:
            features["observation.images.camera_depth_before_filters"] = {
                "dtype": "image",
                "shape": (h, w, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
        if has_infrared:
            ir_h, ir_w = first.infrared_left.shape
            for side in ("left", "right"):
                features[f"observation.images.camera_infrared_{side}"] = {
                    "dtype": "video",
                    "shape": (ir_h, ir_w, 3),
                    "names": ["height", "width", "channels"],
                }
        if has_model_depth:
            features["observation.images.camera_depth_model"] = {
                "dtype": "image",
                "shape": (h, w, 1),
                "names": ["height", "width", "channels"],
                "info": {"is_depth_map": True},
            }
        temporary = directory / (".episode-" + uuid.uuid4().hex)
        try:
            dataset = LeRobotDataset.create(
                f"local/flash-{directory.name}",
                fps,
                features,
                root=temporary,
                robot_type="flash_rgbd",
                use_videos=True,
                image_writer_threads=4,
                rgb_encoder=RGBEncoderConfig(vcodec="h264", preset="veryfast", crf=18),
            )
            overlay_root = temporary / "flash" / "v1" / "overlays"
            overlay_root.mkdir(parents=True)
            processed_root = temporary / "flash" / "v1" / "processed"
            processed_root.mkdir(parents=True)
            for i, row in enumerate(rows):
                frame, _, _, _ = decode_record(read_packet(directory, row["frame"]))
                if frame.depth.shape != (h, w):
                    raise ValueError(f"Camera format changed at frame {row['frame']}")
                episode_frame = {
                    "task": "FLASH live camera tracking",
                    "observation.images.camera": frame.rgb,
                    "observation.images.camera_depth": np.rint(
                        np.nan_to_num(
                            frame.depth.astype(np.float32) * (frame.depth_m_per_unit * 1000),
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                    )
                    .clip(0, 65535)
                    .astype(np.uint16)[:, :, None],
                    "observation.camera.source_frame_id": np.array([frame.frame_id], np.int64),
                    "observation.camera.source_time_s": np.array([frame.source_time_s], np.float64),
                    "observation.camera.available_time_s": np.array([frame.available_time_s], np.float64),
                    "observation.camera.depth_m_per_unit": np.array([frame.depth_m_per_unit], np.float64),
                    "observation.camera.intrinsics": frame.intrinsics.reshape(9),
                    "observation.camera.base_from_camera": frame.base_from_camera.reshape(16),
                }
                if has_before:
                    if frame.depth_before_filters is None and any((frame.depth_processing or {}).values()):
                        raise ValueError(f"Pre-filter depth missing at filtered frame {row['frame']}")
                    before = (
                        frame.depth_before_filters if frame.depth_before_filters is not None else frame.depth
                    )
                    episode_frame["observation.images.camera_depth_before_filters"] = (
                        np.rint(
                            np.nan_to_num(
                                before.astype(np.float32) * (frame.depth_m_per_unit * 1000),
                                nan=0.0,
                                posinf=0.0,
                                neginf=0.0,
                            )
                        )
                        .clip(0, 65535)
                        .astype(np.uint16)[:, :, None]
                    )
                if has_infrared:
                    if frame.infrared_left is None:
                        raise ValueError(f"Infrared pair missing at frame {row['frame']}")
                    for side in ("left", "right"):
                        infrared = getattr(frame, f"infrared_{side}")
                        episode_frame[f"observation.images.camera_infrared_{side}"] = np.repeat(
                            infrared[..., None], 3, axis=2
                        )
                if has_model_depth:
                    model_file = directory / f"{row['frame']:010d}.model_depth.zst"
                    model = (
                        np.frombuffer(
                            zstd.ZstdDecompressor().decompress(model_file.read_bytes()), "<f4"
                        ).reshape(h, w)
                        if model_file.is_file()
                        else np.zeros((h, w), np.float32)
                    )
                    episode_frame["observation.images.camera_depth_model"] = (
                        np.rint(np.nan_to_num(model * 1000, nan=0.0, posinf=0.0, neginf=0.0))
                        .clip(0, 65535)
                        .astype(np.uint16)[:, :, None]
                    )
                dataset.add_frame(episode_frame)
                overlay = directory / f"{row['frame']:010d}.overlay.json"
                if overlay.is_file():
                    os.link(overlay, overlay_root / f"{i:06d}.json")
                if "processed_ready_time_s" in row:
                    for suffix in ("processed.json", "model_depth.zst"):
                        source = directory / f"{row['frame']:010d}.{suffix}"
                        if source.is_file():
                            os.link(source, processed_root / f"{i:06d}.{suffix}")
            dataset.save_episode()
            dataset.finalize()
            shutil.copy2(directory / "config.json", temporary / "flash" / "v1" / "config.json")
            if (directory / "camera_streams.json").is_file():
                shutil.copy2(
                    directory / "camera_streams.json", temporary / "flash" / "v1" / "camera_streams.json"
                )
            write_snapshot(
                temporary / "flash" / "v1" / "camera.json",
                {
                    "width": w,
                    "height": h,
                    "intrinsics": first.intrinsics.tolist(),
                    "depth_m_per_unit": first.depth_m_per_unit,
                },
            )
            write_snapshot(
                temporary / "flash" / "v1" / "source_frames.json",
                {
                    "frames": [
                        {
                            "source_frame_id": row["frame"],
                            "source_time_s": row["time_s"],
                            "depth_processing": row.get("depth_processing"),
                            "depth_filter_revision": row.get("depth_filter_revision", 0),
                        }
                        for row in rows
                    ]
                },
            )
            write_snapshot(
                temporary / "flash" / "v1" / "processed_frames.json",
                {
                    "schema_version": 1,
                    "frames": [
                        {
                            "episode_index": i,
                            "source_frame_id": row["frame"],
                            "source_time_s": row["time_s"],
                            "depth_source": row["processed_depth_source"],
                            "depth_ready_time_s": row["depth_ready_time_s"],
                            "sam_ready_time_s": row.get("sam_ready_time_s"),
                            "ready_time_s": row["processed_ready_time_s"],
                            "depth_file": (
                                f"processed/{i:06d}.model_depth.zst"
                                if row["processed_depth_source"] != "sensor"
                                else None
                            ),
                            "mask_file": (
                                f"processed/{i:06d}.processed.json"
                                if row.get("sam_ready_time_s") is not None
                                else None
                            ),
                        }
                        for i, row in enumerate(rows)
                        if "processed_ready_time_s" in row
                    ],
                },
            )
            temporary.rename(final)
        except Exception:
            # safe-destruct: remove only this failed, generated episode export.
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return final


class RollingHistory:
    def __init__(self, directory, *, seconds=60.0, reserve_bytes=512 * 1024**2):
        if seconds <= 0 or reserve_bytes < 0:
            raise ValueError("Invalid history limits")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.seconds, self.reserve_bytes = seconds, reserve_bytes
        # Enough room for a short compression stall at camera frame rate. A
        # genuine overflow is counted and exposed, never silently subsampled.
        self.queue = Queue(maxsize=64)
        self.rows = []
        self.dropped = 0
        self.error = None
        self.closed = False
        self.thread = Thread(target=self._run, name="flash-evidence-writer", daemon=True)
        self.thread.start()

    def submit(self, packet):
        if self.closed:
            raise RuntimeError("History already closed")
        try:
            self.queue.put_nowait(packet)  # Immutable bytes, not mutable tracker state.
        except Full:
            self.dropped += 1

    def submit_frame(self, frame):
        """Camera-thread handoff, including frames before object selection."""
        if self.closed:
            raise RuntimeError("History already closed")
        try:
            snapshot = replace(
                frame,
                rgb=frame.rgb.copy(),
                depth=frame.depth.copy(),
                depth_before_filters=(
                    frame.depth_before_filters.copy() if frame.depth_before_filters is not None else None
                ),
                infrared_left=frame.infrared_left.copy() if frame.infrared_left is not None else None,
                infrared_right=frame.infrared_right.copy() if frame.infrared_right is not None else None,
            )
            self.queue.put_nowait(("capture", snapshot))
        except Full:
            self.dropped += 1

    def attach_result(self, packet):
        """Add a processed result to its already captured frame by ID."""
        if self.closed:
            raise RuntimeError("History already closed")
        try:
            self.queue.put_nowait(("result", (packet, time.perf_counter())))
        except Full:
            self.error = "Tracking result could not be recorded: history writer queue full"

    def attach_processed_frame(self, frame, depth_source, *, masks=None, complete=True):
        """Record depth and optional SAM output against one captured frame ID.

        Called at the producer boundary, before the pose fitter. The writer
        owns the compressed copy, so model and tracker throughput remain separate.
        """
        if self.closed:
            raise RuntimeError("History already closed")
        frame_id = frame if isinstance(frame, int) else frame.frame_id
        depth = None
        if depth_source != "sensor":
            depth = np.ascontiguousarray(frame.metric_depth(), dtype="<f4").tobytes()
        runs = None
        if masks is not None:
            runs = {}
            for role, mask in masks.items():
                edges = np.flatnonzero(np.diff(np.pad(mask.ravel().astype(np.int8), (1, 1))))
                runs[role] = np.column_stack((edges[::2], edges[1::2] - edges[::2])).tolist()
        event = (frame_id, depth_source, time.perf_counter(), depth, runs, complete)
        try:
            self.queue.put_nowait(("processed", event))
        except Full:
            self.error = "Processed frame could not be recorded: history writer queue full"

    def _index(self):
        return {
            "schema_version": 1,
            "frames": self.rows,
            "retention_seconds": self.seconds,
            "bytes": sum(row["bytes"] for row in self.rows),
            "dropped_frames": self.dropped,
            "error": self.error,
        }

    def _write(self, packet):
        n = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + n])
        encoded = zlib.compress(packet, 1)
        dependencies = set()
        for value in header["objects"].values():
            detail = value.get("diagnostics", {})
            dependencies.update(detail.get("surface_frames_used", []))
            dependencies.update(
                detail.get(key) for key in ("reference_frame", "accepted_frame_before", "image_source_frame")
            )
        dependencies.discard(None)
        row = {
            "frame": header["frame"],
            "time_s": header["time_s"],
            "bytes": len(encoded),
            "dependencies": sorted(dependencies),
            "objects": {
                role: {key: obj.get(key) for key in ("name", "status", "reason")}
                for role, obj in header["objects"].items()
            },
        }
        with locked(self.directory, exclusive=True):
            cutoff = row["time_s"] - self.seconds
            window = [r for r in self.rows + [row] if r["time_s"] >= cutoff]
            needed = {i for r in window for i in r["dependencies"]}
            retained = [r for r in self.rows if r["time_s"] >= cutoff or r["frame"] in needed]
            retained.append(row)
            keep = {r["frame"] for r in retained}
            for old in self.rows:
                if old["frame"] not in keep:
                    # safe-destruct: expire only this rolling buffer's generated frame file.
                    (self.directory / f"{old['frame']:010d}.bin.z").unlink(missing_ok=True)
                    # safe-destruct: expire only this rolling buffer's generated frame file.
                    (self.directory / f"{old['frame']:010d}.bin.zst").unlink(missing_ok=True)
                    # safe-destruct: expire only this rolling buffer's generated overlay file.
                    (self.directory / f"{old['frame']:010d}.overlay.json").unlink(missing_ok=True)
                    # safe-destruct: expire only this rolling buffer's generated model-depth file.
                    (self.directory / f"{old['frame']:010d}.model_depth.zst").unlink(missing_ok=True)
            self.rows = [r for r in retained if r is not row]
            if shutil.disk_usage(self.directory).free - len(encoded) < self.reserve_bytes:
                raise OSError(
                    f"History paused: less than {self.reserve_bytes / 1024**2:g} MiB disk reserve available"
                )
            path = self.directory / f"{row['frame']:010d}.bin.z"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(path)
            self.rows = retained
            available = {r["frame"] for r in retained}
            for r in retained:
                r["context_only"] = r["time_s"] < cutoff
                r["missing_dependencies"] = sorted(set(r["dependencies"]) - available)
            self.error = None
            write_snapshot(self.directory / "index.json", self._index())

    def _write_capture(self, frame):
        packet = capture_packet(frame)
        encoded = zstd.ZstdCompressor(level=3).compress(packet)
        row = {
            "frame": frame.frame_id,
            "time_s": frame.source_time_s,
            "depth_processing": frame.depth_processing,
            "depth_filter_revision": frame.depth_filter_revision,
            "has_before_filters": frame.depth_before_filters is not None,
            "has_infrared": frame.infrared_left is not None,
            "bytes": len(encoded),
            "dependencies": [],
            "objects": {},
        }
        with locked(self.directory, exclusive=True):
            cutoff = row["time_s"] - self.seconds
            expired = [r for r in self.rows if r["time_s"] < cutoff]
            self.rows = [r for r in self.rows if r["time_s"] >= cutoff]
            for old in expired:
                self._remove_frame(old["frame"])
            if shutil.disk_usage(self.directory).free - len(encoded) < self.reserve_bytes:
                write_snapshot(self.directory / "index.json", self._index())
                raise OSError(
                    f"History paused: less than {self.reserve_bytes / 1024**2:g} MiB disk reserve available"
                )
            path = self.directory / f"{frame.frame_id:010d}.bin.zst"
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(encoded)
            temporary.replace(path)
            self.rows.append(row)
            self.error = None
            write_snapshot(self.directory / "index.json", self._index())

    def _remove_frame(self, frame):
        for suffix in (".bin.z", ".bin.zst", ".processed.json", ".overlay.json", ".model_depth.zst"):
            # safe-destruct: remove only generated files for this rolling buffer frame.
            (self.directory / f"{frame:010d}{suffix}").unlink(missing_ok=True)

    def _write_processed(self, event):
        frame_id, source, ready_time_s, depth, masks, complete = event
        with locked(self.directory, exclusive=True):
            row = next((r for r in self.rows if r["frame"] == frame_id), None)
            if row is None:
                return  # Its camera capture expired or was not retained.
            path = self.directory / f"{frame_id:010d}.processed.json"
            previous = json.loads(path.read_text()) if path.is_file() else {}
            if previous and previous["processed_depth_source"] != source:
                raise ValueError("Processed depth source changed within one camera frame")
            update = {**previous, "processed_depth_source": source}
            if "depth_ready_time_s" not in update:
                update["depth_ready_time_s"] = ready_time_s
            if masks is not None:
                update["sam_ready_time_s"] = ready_time_s
                update["masks_rle"] = masks
            if complete:
                update["processed_ready_time_s"] = ready_time_s
            before = path.stat().st_size if path.is_file() else 0
            write_snapshot(path, update)
            row["bytes"] += path.stat().st_size - before
            row["processed_depth_source"] = source
            row["depth_ready_time_s"] = update["depth_ready_time_s"]
            if masks is not None:
                row["sam_ready_time_s"] = ready_time_s
            if complete:
                row["processed_ready_time_s"] = ready_time_s
            if depth is not None:
                model_path = self.directory / f"{frame_id:010d}.model_depth.zst"
                if not model_path.is_file():
                    model_path.write_bytes(zstd.ZstdCompressor(level=3).compress(depth))
                    row["bytes"] += model_path.stat().st_size
            write_snapshot(self.directory / "index.json", self._index())

    def _write_result(self, event):
        packet, ready_time_s = event
        n = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + n])
        header["tracking_ready_time_s"] = ready_time_s
        with locked(self.directory, exclusive=True):
            row = next((r for r in self.rows if r["frame"] == header["frame"]), None)
            if row is None:
                return  # The camera frame expired or was not retained.
            path = self.directory / f"{row['frame']:010d}.overlay.json"
            before = path.stat().st_size if path.is_file() else 0
            write_snapshot(path, header)
            if header.get("depth_source", "sensor") != "sensor":
                depth_start = 4 + n
                depth_bytes = packet[depth_start : depth_start + header["depth_bytes"]]
                model_path = self.directory / f"{row['frame']:010d}.model_depth.zst"
                if not model_path.is_file():
                    model_path.write_bytes(zstd.ZstdCompressor(level=3).compress(depth_bytes))
                    row["bytes"] += model_path.stat().st_size
                row["tracking_depth_source"] = header["depth_source"]
            row["objects"] = {
                role: {key: obj.get(key) for key in ("name", "status", "reason")}
                for role, obj in header.get("objects", {}).items()
            }
            row["tracking_ready_time_s"] = ready_time_s
            row["bytes"] += path.stat().st_size - before
            write_snapshot(self.directory / "index.json", self._index())

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    break
                if isinstance(item, tuple):
                    kind, value = item
                    if kind == "capture":
                        self._write_capture(value)
                    elif kind == "processed":
                        self._write_processed(value)
                    elif kind == "result":
                        self._write_result(value)
                    elif kind == "clear":
                        try:
                            self._clear_files()
                        finally:
                            value.set()
                else:
                    self._write(item)
            except Exception as exc:
                self.dropped += 1
                self.error = str(exc)
                with suppress(OSError), locked(self.directory, exclusive=True):
                    write_snapshot(self.directory / "index.json", self._index())
            finally:
                self.queue.task_done()
        with suppress(OSError), locked(self.directory, exclusive=True):
            write_snapshot(self.directory / "index.json", self._index())

    def close(self):
        self.closed = True
        self.queue.put(None)
        self.thread.join()

    def clear(self):
        """Discard the rolling recording; future frames continue normally."""
        done = Event()
        self.queue.put(("clear", done))
        done.wait()

    def _clear_files(self):
        with locked(self.directory, exclusive=True):
            for row in self.rows:
                # safe-destruct: clear only generated frames in the requested rolling buffer.
                self._remove_frame(row["frame"])
            self.rows = []
            self.dropped = 0
            self.error = None
            write_snapshot(self.directory / "index.json", self._index())
