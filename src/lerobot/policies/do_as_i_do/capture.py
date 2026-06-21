# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Standalone RealSense capture for the MVP: aligned RGB + metric depth +
intrinsics, with no robot/teleop dependency and no motor motion. Used both for a
single validation frame and for a hand-demo clip.

Kept independent of ``RealSenseCamera`` on purpose: this is a throwaway capture
path, and the working tree's camera class lacks the intrinsics accessor anyway.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np


def reset_devices(settle_s: float = 3.0) -> None:
    """Hardware-reset connected RealSense devices and wait for re-enumeration.

    A wedged USB session (e.g. a prior pipeline that didn't release) shows up as
    ``Frame didn't arrive``; a reset clears it. No-op if no device is present.
    """
    import pyrealsense2 as rs

    ctx = rs.context()
    devs = list(ctx.query_devices())
    if not devs:
        return
    for d in devs:
        try:
            d.hardware_reset()
        except Exception:
            pass
    time.sleep(settle_s)


def capture_frames(
    n: int = 1, warmup: int = 15, width: int = 640, height: int = 480, fps: int = 30, retries: int = 3
):
    """Grab ``n`` aligned (RGB, depth_m) frames + intrinsics from the RealSense.

    Args:
        n: number of frames to return (1 = a single still; >1 = a clip).
        warmup: frames to discard first so auto-exposure settles.
        retries: on ``Frame didn't arrive``, hardware-reset and retry this many times.

    Returns ``(rgbs, depths_m, intr)``: ``rgbs`` is ``(n,H,W,3)`` uint8 RGB,
    ``depths_m`` is ``(n,H,W)`` float32 meters (0 = no return), ``intr`` is the
    color-stream pinhole dict ``{fx,fy,ppx,ppy,width,height}``.

    Precondition: a RealSense device is connected. Blocks ~``(warmup+n)/fps`` s.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            return _capture_once(n, warmup, width, height, fps)
        except RuntimeError as e:
            last_err = e
            reset_devices()
    raise RuntimeError(f"RealSense capture failed after {retries} attempts: {last_err}")


def _capture_once(n: int, warmup: int, width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(cfg)
    try:
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ci = color_stream.get_intrinsics()
        intr = {
            "fx": float(ci.fx),
            "fy": float(ci.fy),
            "ppx": float(ci.ppx),
            "ppy": float(ci.ppy),
            "width": int(ci.width),
            "height": int(ci.height),
        }
        for _ in range(warmup):
            pipeline.wait_for_frames(10000)
        rgbs, depths = [], []
        for _ in range(n):
            frames = align.process(pipeline.wait_for_frames(10000))
            rgbs.append(np.asarray(frames.get_color_frame().get_data(), dtype=np.uint8).copy())
            depth_raw = np.asarray(frames.get_depth_frame().get_data(), dtype=np.uint16)
            depths.append(depth_raw.astype(np.float32) * depth_scale)
        return np.stack(rgbs), np.stack(depths), intr
    finally:
        pipeline.stop()


def save_clip(path: str | Path, rgbs: np.ndarray, depths_m: np.ndarray, intr: dict) -> None:
    """Persist a capture to a single ``.npz`` (RGB + depth_m + intrinsics)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, rgbs=rgbs, depths_m=depths_m, intr=np.array(list(intr.items()), dtype=object))


def load_clip(path: str | Path):
    """Inverse of :func:`save_clip`. Returns ``(rgbs, depths_m, intr)``."""
    d = np.load(path, allow_pickle=True)
    intr = {k: float(v) for k, v in d["intr"]}
    return d["rgbs"], d["depths_m"], intr
