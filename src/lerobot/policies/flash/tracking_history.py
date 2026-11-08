"""Rolling camera RGB-D with optional tracking results keyed by capture frame.

Capture never waits for the tracker. The writer compresses measured frames on
its own thread; a later result adds metadata without replacing those pixels.
"""

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
        overlay = base.with_suffix(".overlay.json")
        if overlay.exists():
            n = struct.unpack("<I", packet[:4])[0]
            header = json.loads(packet[4 : 4 + n])
            update = json.loads(overlay.read_text())
            header.update(
                {
                    key: value
                    for key, value in update.items()
                    if key
                    not in (
                        "camera",
                        "depth_bytes",
                        "source_frame_id",
                        "base_from_camera",
                        "available_time_s",
                        "calibration_version",
                        "rgb_format",
                        "depth_format",
                        "depth_m_per_unit",
                    )
                }
            )
            metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
            packet = struct.pack("<I", len(metadata)) + metadata + packet[4 + n :]
        return packet


def capture_packet(frame):
    """Lossless camera input in the same scene envelope as processed packets."""
    depth = np.ascontiguousarray(frame.depth)
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
        "depth_format": "sensor_u16" if depth.dtype == np.uint16 else "metric_f32",
        "depth_m_per_unit": frame.depth_m_per_unit,
        "rgb_format": "raw_rgb8",
        "objects": {},
        "scene_group_tracks": [],
        "masks_rle": {},
        "result": None,
    }
    metadata = json.dumps(header, allow_nan=False, separators=(",", ":")).encode()
    return struct.pack("<I", len(metadata)) + metadata + depth.tobytes() + rgb.tobytes()


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
    rgb_bytes = packet[start + header["depth_bytes"] :]
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
                for suffix in (".bin.zst", ".bin.z", ".overlay.json"):
                    name = base + suffix
                    if (directory / name).exists():
                        os.link(directory / name, temporary / name)
            shutil.copy2(directory / "config.json", temporary / "config.json")
            write_snapshot(temporary / "index.json", {**index, "saved_at": time.time()})
        temporary.rename(final)
    except Exception:
        # Only this incomplete export; existing recordings are never touched.
        # safe-destruct: remove only the temporary directory created for this failed save.
        shutil.rmtree(temporary)
        raise
    return identity


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
            snapshot = replace(frame, rgb=frame.rgb.copy(), depth=frame.depth.copy())
            self.queue.put_nowait(("capture", snapshot))
        except Full:
            self.dropped += 1

    def attach_result(self, packet):
        """Add a processed result to its already captured frame by ID."""
        if self.closed:
            raise RuntimeError("History already closed")
        try:
            self.queue.put_nowait(("result", packet))
        except Full:
            self.error = "Tracking result could not be recorded: history writer queue full"

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
        for suffix in (".bin.z", ".bin.zst", ".overlay.json"):
            # safe-destruct: remove only generated files for this rolling buffer frame.
            (self.directory / f"{frame:010d}{suffix}").unlink(missing_ok=True)

    def _write_result(self, packet):
        n = struct.unpack("<I", packet[:4])[0]
        header = json.loads(packet[4 : 4 + n])
        with locked(self.directory, exclusive=True):
            row = next((r for r in self.rows if r["frame"] == header["frame"]), None)
            if row is None:
                return  # The camera frame expired or was not retained.
            path = self.directory / f"{row['frame']:010d}.overlay.json"
            write_snapshot(path, header)
            row["objects"] = {
                role: {key: obj.get(key) for key in ("name", "status", "reason")}
                for role, obj in header.get("objects", {}).items()
            }
            row["bytes"] += path.stat().st_size
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
