# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Hand/wrist reconstruction. MediaPipe Hands gives 2D landmarks; we ignore its
relative-depth z and instead sample the RealSense METRIC depth at the landmark
pixel, then lift to the robot base frame. Grasp signal = thumb-tip to index-tip
3D distance (open hand large, pinch small).

MediaPipe runs in an isolated venv (subprocess) so it stays off the conda env.
The default paths are throwaway prototype locations on this host.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile

import numpy as np

from .calib import transform_points, unproject_pixel

MP_VENV_PY = "/tmp/do_as_i_do/mp_venv/bin/python"
MP_WORKER = "/tmp/do_as_i_do/hand_worker.py"
MP_MODEL = "/tmp/do_as_i_do/hand_landmarker.task"

# MediaPipe hand landmark indices.
WRIST, THUMB_TIP, INDEX_TIP, MIDDLE_MCP = 0, 4, 8, 9


def _sample_depth(depth_m: np.ndarray, u: float, v: float, win: int = 2) -> float:
    """Median of valid depth in a small window around ``(u, v)`` (robust to holes)."""
    h, w = depth_m.shape
    u, v = int(round(u)), int(round(v))
    patch = depth_m[max(0, v - win) : min(h, v + win + 1), max(0, u - win) : min(w, u + win + 1)]
    vals = patch[patch > 0]
    return float(np.median(vals)) if vals.size else 0.0


def run_mediapipe(
    rgbs: np.ndarray, mp_py: str = MP_VENV_PY, worker: str = MP_WORKER, model: str = MP_MODEL
) -> list[dict]:
    """Detect hand landmarks per frame via the isolated-venv worker.

    Returns one dict per frame: ``{"landmarks": [[x,y,z]*21] | None, "handedness": str}``
    where x,y are normalized image coords in [0,1].
    """
    with tempfile.TemporaryDirectory() as td:
        inp, out = os.path.join(td, "in.npz"), os.path.join(td, "out.json")
        np.savez_compressed(inp, rgbs=rgbs)
        proc = subprocess.run([mp_py, worker, inp, out, model], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"mediapipe worker failed:\n{proc.stderr}")
        with open(out) as f:
            return json.load(f)


def _dist2d(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def reconstruct_hand(
    rgbs: np.ndarray,
    depths_m: np.ndarray,
    intr: dict[str, float],
    T_base_camera: np.ndarray,  # noqa: N803
) -> list[dict | None]:
    """Per-frame wrist/thumb/index in the base frame + grasp distance.

    Returns one entry per frame (``None`` if no hand): ``{wrist, thumb, index}``
    are ``(3,)`` base-frame XYZ (or None if depth missing), ``grasp_dist`` is the
    thumb<->index metric distance in meters, ``handedness`` is Left/Right.
    """
    detections = run_mediapipe(rgbs)
    out: list[dict | None] = []
    for i, det in enumerate(detections):
        lm = det["landmarks"]
        if lm is None:
            out.append(None)
            continue
        h, w = depths_m[i].shape

        def pt3d(idx, _i=i, _h=h, _w=w, _lm=lm):
            u, v = _lm[idx][0] * _w, _lm[idx][1] * _h
            z = _sample_depth(depths_m[_i], u, v)
            if z <= 0:
                return None
            return transform_points(T_base_camera, unproject_pixel(u, v, z, intr)[None])[0]

        wrist, thumb, index = pt3d(WRIST), pt3d(THUMB_TIP), pt3d(INDEX_TIP)
        grasp = float(np.linalg.norm(thumb - index)) if thumb is not None and index is not None else None
        # Pinch ratio: 2D thumb-index gap normalized by hand size (wrist->middle MCP).
        # Depth-free, so robust to the fingertip depth dropouts that made grasp_dist
        # collapse; this is the signal used for grasp *timing*.
        scale = _dist2d(lm[WRIST], lm[MIDDLE_MCP]) or 1e-6
        pinch_ratio = _dist2d(lm[THUMB_TIP], lm[INDEX_TIP]) / scale
        out.append(
            {
                "wrist": wrist,
                "thumb": thumb,
                "index": index,
                "grasp_dist": grasp,
                "pinch_ratio": pinch_ratio,
                "handedness": det.get("handedness"),
            }
        )
    return out
