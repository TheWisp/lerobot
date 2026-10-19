#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measured object-to-gripper relationships and checked demonstration references."""

from __future__ import annotations

import pathlib

import numpy as np

from . import planning, recording, runtime, tracking


def _firm_grip(demo: runtime._Demo) -> int | None:
    """The demo sample its grip became firm, where its hold begins (:func:`core.demo_holding`); None when the
    gripper never stopped short of its command in a stretch the arm follows."""
    holding = planning._holding(demo)
    return None if holding is None else holding["grip"]


def _demo_surface_views(
    demo: runtime._Demo, obj: str, frames: list[tuple[int, int]]
) -> list[tuple[int, dict]]:
    """Re-evaluate recorded hold observations; no camera or tracker state is used."""
    import cv2

    o, out = demo.objects[obj], []
    reference, intr = tracking._object_surface(demo, obj), _recording_intr(demo)
    for f, i in frames:
        _rgb, depth = recording._stream_frame(demo.recording, f)
        mask = np.asarray(o["masks"][f], np.uint8)
        if mask.shape != depth.shape:
            mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        check = tracking._surface_decision(o["deltas"][f], reference, depth, mask, intr)
        out.append((i, {"frame": int(f), **check}))
    return out


def _prepare_demo_references(demo: runtime._Demo) -> str:
    """Validate the few demo poses used to plan, off the event loop before motion.

    Never rewrite recorded deltas. Keeping raw and measured references separate
    prevents the inverse of an old false tilt from undoing a corrected live pose.
    """
    checks, problems = {}, []
    for obj in planning._act_objects(demo):
        f = planning._pose_frame(demo, obj)
        if f is None:
            problems.append(f"{obj}'s demo pose is not visible before this interaction")
            continue
        raw = np.asarray(demo.objects[obj]["deltas"][f], dtype=float)
        try:
            if not demo.objects[obj]["seen"][f]:
                raise ValueError("object was not tracked in this reference frame")
            check = _demo_surface_views(demo, obj, [(f, 0)])[0][1]
        except (OSError, KeyError, ValueError, AssertionError) as exc:
            check = {"state": "rejected", "pose": None, "reason": f"recorded surface unavailable: {exc}"}
        checks[obj] = {**check, "frame": int(f), "raw_pose": raw.tolist()}
        if check.get("pose") is None:
            problems.append(f"{obj} demo reference frame {f}: {check['reason']}")
    demo.reference_constraints = checks
    return "; ".join(problems)


def _demo_control_pose(demo: runtime._Demo, obj: str, frame: int) -> tuple[np.ndarray | None, str]:
    """Read prepared geometry without file IO in the control loop.

    Pure composition/editor callers can still inspect an unprepared recording.
    Every physical runner prepares references first and aborts on any rejection.
    Loaded demos are prepared before preview; a changed selection cannot reuse
    a checked pose from a different frame.
    """
    if not demo.reference_constraints:
        return np.asarray(demo.objects[obj]["deltas"][frame], dtype=float), ""
    check = demo.reference_constraints.get(obj, {})
    if check.get("frame") != frame or check.get("pose") is None:
        return None, f"{obj}'s demo reference is not verified: {check.get('reason', 'selection changed')}"
    return np.asarray(check["pose"], dtype=float), ""


def _recording_intr(demo: runtime._Demo) -> dict[str, float]:
    """The intrinsics of the demo's recorded stream: as the demo kept them, else from the recording's own file."""
    if demo.intr and "fx" in demo.intr:
        return dict(demo.intr)
    k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
    rgb, _depth = recording._stream_frame(demo.recording, 0)
    return {
        "fx": k[0, 0],
        "fy": k[1, 1],
        "cx": k[0, 2],
        "cy": k[1, 2],
        "width": rgb.shape[1],
        "height": rgb.shape[0],
    }
