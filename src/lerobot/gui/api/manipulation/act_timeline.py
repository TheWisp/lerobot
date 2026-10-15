"""Fixed-rate Act observation timeline; camera, inference and disk remain asynchronous to the motor bus."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ActTimeline:
    """Sample the camera's latest published RGB-D frame on a fixed clock.

    The RealSense multi-reader accessor does not consume a frame from P2P or the
    point groups. Encoding and disk writes run on another thread. A slow writer
    or a missed tick is recorded explicitly; neither can hold up the motor loop.
    """

    def __init__(self, root: Path, camera: Any, fps: float) -> None:
        self.root, self.camera, self.fps = root, camera, float(fps)
        self.stop = threading.Event()
        self.pending: queue.Queue[tuple[int, np.ndarray, np.ndarray] | None] = queue.Queue(maxsize=90)
        self.rows: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.saved = 0
        self.origin_mono = 0.0
        self.origin_wall = 0.0
        self.sampler: threading.Thread | None = None
        self.writer: threading.Thread | None = None
        self._closed = False

    def start(self) -> None:
        if not callable(getattr(self.camera, "read_newer_color_and_depth", None)):
            raise RuntimeError("camera has no non-consuming timestamped RGB-D reader")
        (self.root / "rgb").mkdir(parents=True, exist_ok=True)
        (self.root / "depth").mkdir(exist_ok=True)
        self.origin_mono, self.origin_wall = time.perf_counter(), time.time()
        self.writer = threading.Thread(target=self._write, name="act-timeline-writer", daemon=True)
        self.sampler = threading.Thread(target=self._sample, name="act-timeline-30hz", daemon=True)
        self.writer.start()
        self.sampler.start()

    def _sample(self) -> None:
        period = 1.0 / self.fps
        slot, last_stamp, last_image = 0, None, None
        try:
            while not self.stop.is_set():
                due = self.origin_mono + slot * period
                if self.stop.wait(max(0.0, due - time.perf_counter())):
                    break
                sampled = time.perf_counter()
                # Keep the timeline's grid when a scheduling stall misses one or
                # more ticks. Those slots visibly hold the last image, with an
                # explicit missed_tick status; no camera frame is invented.
                while sampled - due >= period:
                    self.rows.append(
                        {
                            "slot": slot,
                            "t_mono": due,
                            "t_wall": self.origin_wall + slot * period,
                            "image": last_image,
                            "camera_mono": last_stamp,
                            "status": "missed_tick",
                        }
                    )
                    slot += 1
                    due = self.origin_mono + slot * period
                status = "held_camera_frame"
                try:
                    rgb, depth, stamp = self.camera.read_newer_color_and_depth(None)
                    stamp = float(stamp)
                    if stamp != last_stamp:
                        try:
                            self.pending.put_nowait((slot, np.asarray(rgb).copy(), np.asarray(depth).copy()))
                            last_stamp, last_image, status = stamp, slot, "captured"
                        except queue.Full:
                            status = "writer_full"
                except Exception as exc:
                    status = f"camera_error: {type(exc).__name__}: {exc}"
                    if not self.errors or self.errors[-1] != status:
                        self.errors.append(status)
                self.rows.append(
                    {
                        "slot": slot,
                        "t_mono": due,
                        "t_wall": self.origin_wall + slot * period,
                        "sampled_mono": sampled,
                        "image": last_image,
                        "camera_mono": last_stamp,
                        "status": status,
                    }
                )
                slot += 1
        except Exception as exc:
            logger.exception("Act timeline sampler failed")
            self.errors.append(f"sampler: {type(exc).__name__}: {exc}")

    def _write(self) -> None:
        import cv2

        while True:
            item = self.pending.get()
            if item is None:
                self.pending.task_done()
                return
            slot, rgb, depth = item
            try:
                image = self.root / "rgb" / f"{slot:06d}.jpg"
                measured = self.root / "depth" / f"{slot:06d}.png"
                if not cv2.imwrite(str(image), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                    raise OSError(f"cannot write {image}")
                if not cv2.imwrite(str(measured), depth.astype(np.uint16)):
                    raise OSError(f"cannot write {measured}")
                self.saved += 1
            except Exception as exc:
                logger.exception("Act timeline frame write failed")
                self.errors.append(f"writer slot {slot}: {type(exc).__name__}: {exc}")
            finally:
                self.pending.task_done()

    def close(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("Act timeline already closed")
        self._closed = True
        self.stop.set()
        if self.sampler is not None:
            self.sampler.join(timeout=5.0)
            if self.sampler.is_alive():
                self.errors.append("sampler did not stop")
        try:
            self.pending.put(None, timeout=5.0)
        except queue.Full:
            self.errors.append("writer queue did not drain")
        if self.writer is not None:
            self.writer.join(timeout=30.0)
            if self.writer.is_alive():
                self.errors.append("writer did not finish")
        timeline = {
            "fps": self.fps,
            "origin_mono": self.origin_mono,
            "origin_wall": self.origin_wall,
            "rows": self.rows,
            "saved": self.saved,
            "errors": self.errors,
        }
        (self.root / "timeline.json").write_text(json.dumps(timeline))
        return {
            "fps": self.fps,
            "slots": len(self.rows),
            "saved": self.saved,
            "missed_ticks": sum(r["status"] == "missed_tick" for r in self.rows),
            "errors": list(self.errors),
        }
