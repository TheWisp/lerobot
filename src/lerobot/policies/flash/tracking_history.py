"""Bounded, lossless frame evidence. Retention changes storage, never tracking.

Each compressed file is the same RGB-D scene packet used by video inspection,
including the original frame result. A two-frame queue keeps disk/compression
work off the camera loop. Saved windows hard-link immutable frames under a lock.
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
from pathlib import Path
from queue import Full, Queue
from threading import Thread

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
        return zlib.decompress((directory / f"{frame:010d}.bin.z").read_bytes())


def decode_record(packet):
    """Restore measured inputs and the unchanged pipeline result for inspection."""
    import cv2
    import numpy as np

    from .perception import RGBDFrame

    n = struct.unpack("<I", packet[:4])[0]
    header = json.loads(packet[4 : 4 + n])
    camera = header["camera"]
    start = 4 + n
    depth = (
        np.frombuffer(packet[start : start + header["depth_bytes"]], "<f4")
        .reshape(camera["height"], camera["width"])
        .copy()
    )
    rgb = cv2.imdecode(np.frombuffer(packet[start + header["depth_bytes"] :], np.uint8), cv2.IMREAD_COLOR)
    frame = RGBDFrame(
        header["source_frame_id"],
        cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB),
        depth,
        np.asarray(camera["intrinsics"]),
        1.0,
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
    return frame, masks, header["result"], header.get("image_matches")


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
                raise ValueError("No recorded tracking frames yet")
            for row in index["frames"]:
                name = f"{row['frame']:010d}.bin.z"
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
        self.queue = Queue(maxsize=2)
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

    def _run(self):
        while True:
            packet = self.queue.get()
            try:
                if packet is None:
                    break
                self._write(packet)
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
        self.queue.join()
        with locked(self.directory, exclusive=True):
            for row in self.rows:
                # safe-destruct: clear only generated frames in the requested rolling buffer.
                (self.directory / f"{row['frame']:010d}.bin.z").unlink(missing_ok=True)
            self.rows = []
            self.dropped = 0
            self.error = None
            write_snapshot(self.directory / "index.json", self._index())
