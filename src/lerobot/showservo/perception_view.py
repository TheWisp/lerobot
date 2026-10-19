# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""A recording drawn with its perception (lerobot.showservo.perception), the same for a demo and an act, so the two
can be laid side by side at the same moment of the demo and compared by eye.

On each frame: the point groups as the act's views paint them (groups_scene.paint_act_view, the act replay's own),
each object's surface where the groups put it, green where the engine's estimator took that pose and red where it
refused it, the fingertip where the arm had it, and in words what each object's frame said and why.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

from . import groups_scene, motion as core
from .perception import Perception, surface
from .recordings import Recording

TAKEN, REFUSED, TIP = (60, 220, 60), (60, 60, 240), (255, 255, 255)  # BGR


def surfaces(recording: Recording, perception: Perception) -> dict[str, np.ndarray]:
    """Each object's surface at its designation, camera frame: its mask over the depth of the frame it was
    designated on (the perception's first)."""
    depth = recording.depth(recording.frame_at(float(perception.t[0])))
    return {n: surface(m, depth, recording.K) for n, m in perception.masks.items()}


def render(
    recording: Recording,
    perception: Perception,
    k: int,
    *,
    shapes: dict[str, np.ndarray] | None = None,
    base_from_camera: np.ndarray | None = None,
    demo_time: float | None = None,
    title: str = "",
) -> np.ndarray:
    """Perception frame ``k`` of ``recording`` drawn (BGR). ``shapes`` are :func:`surfaces`, computed once by the
    caller for a run of frames; ``base_from_camera`` places the fingertip, when the recording does not keep it;
    ``demo_time`` is said, for a frame laid beside the demo's."""
    t = float(perception.t[k])
    i = recording.frame_at(t)
    bgr = cv2.cvtColor(recording.color(i), cv2.COLOR_RGB2BGR)
    picture = groups_scene.recorded_picture(perception.pictures[k], perception.picture(k))
    if picture is not None:
        bgr = groups_scene.paint_act_view(bgr, picture)
    shapes = surfaces(recording, perception) if shapes is None else shapes
    lines = [
        f"{title}  {recording.kind} +{t - float(recording.times[0]):.2f} s"
        + ("" if demo_time is None else f"  demo {demo_time:.2f} s")
    ]
    for name, row in (perception.rows[k].get("objects") or {}).items():
        colour = TAKEN if row["taken"] else REFUSED
        if name in shapes:
            m = perception.motion(name, k)
            _dots(bgr, shapes[name] @ m[:3, :3].T + m[:3, 3], recording.K, colour)
        seen = "" if row.get("seen_through") is None else f", {row['seen_through']:.0%} of it seen through"
        lines.append(f"{name}: {'taken' if row['taken'] else 'refused'}{seen} ({row['why']})")
    t_bc = recording.base_from_camera if recording.base_from_camera is not None else base_from_camera
    tip = _tip_at(recording, t)
    if t_bc is not None and tip is not None:
        p = (np.linalg.inv(t_bc) @ tip)[:3, 3]
        if p[2] > 0:
            u, v = (recording.K @ (p / p[2]))[:2]
            cv2.drawMarker(bgr, (int(u), int(v)), TIP, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    for row_n, text in enumerate(lines):
        _text(bgr, text, (8, 20 + 18 * row_n))
    _text(bgr, "green: pose taken  red: pose refused  white cross: fingertip", (8, bgr.shape[0] - 10))
    return bgr


def pair(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
    """Two drawn frames side by side, the right one scaled to the left one's height."""
    if right.shape[0] != left.shape[0]:
        right = cv2.resize(right, (int(right.shape[1] * left.shape[0] / right.shape[0]), left.shape[0]))
    return np.hstack([left, np.full((left.shape[0], gap, 3), 25, np.uint8), right])


def frame_at(perception: Perception, t: float) -> int:
    """The perception frame nearest wall-clock time ``t``."""
    return int(np.argmin(np.abs(perception.t - t)))


def act_clock(
    targets: list[dict[str, Any]], keypoints: list[dict[str, Any]]
) -> Callable[[float], float | None]:
    """An act's demo time at a wall-clock time, from the targets it gave the arm: each target's demo time where the
    act recorded it (``td``), else its leg's place in the demo, the leg toward key frame L running from the key frame
    before L to L. Going back, and moves to a state, say nothing of where in the demo the act is."""
    if any("td" in x for x in targets):
        known = [
            (float(x["t"]), float(x["td"]))
            for x in targets
            if "td" in x and x.get("step", "").startswith("leg")
        ]
    else:
        marks = sorted(core.key_frames(keypoints), key=lambda m: float(m["t"]))
        span, before = {}, None
        for m in marks:
            span[m["label"]] = (float(before) if before is not None else float(m["t"]), float(m["t"]))
            before = m["t"]
        known, run = [], []
        for x in [*targets, {"step": "", "t": np.inf}]:
            label = x.get("step", "")[len("leg: ") :] if x.get("step", "").startswith("leg: ") else None
            if run and label != run[0][1]:
                t0, t1 = span[run[0][1]]
                for j, (w, _label) in enumerate(run):
                    known.append((w, t0 + (t1 - t0) * (j / max(1, len(run) - 1))))
                run = []
            if label in span:
                run.append((float(x["t"]), label))
    known.sort()
    if not known:
        return lambda _t: None
    w, td = np.array([a for a, _ in known]), np.array([b for _, b in known])

    def clock(t: float) -> float | None:
        if t < w[0] or t > w[-1]:
            return None
        return float(np.interp(t, w, td))

    return clock


def act_time_at(
    clock_points: Callable[[float], float | None], times: np.ndarray, demo_t: float
) -> float | None:
    """The first of ``times`` (an act's, wall clock) at which the act had reached demo time ``demo_t``."""
    for t in times:
        td = clock_points(float(t))
        if td is not None and td >= demo_t:
            return float(t)
    return None


def _tip_at(recording: Recording, t: float) -> np.ndarray | None:
    j = int(np.argmin(np.abs(recording.arm.t - t))) if len(recording.arm.t) else -1
    if j < 0 or abs(float(recording.arm.t[j]) - t) > 2.0 / 30.0:
        return None
    return recording.arm.tip[j]


def _dots(bgr: np.ndarray, points: np.ndarray, K: np.ndarray, colour) -> None:  # noqa: N803
    p = points[points[:, 2] > 0]
    uv = (p @ K.T)[:, :2] / p[:, 2:3]
    h, w = bgr.shape[:2]
    for u, v in uv[::3]:
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(bgr, (int(u), int(v)), 2, colour, -1, cv2.LINE_AA)


def _text(bgr: np.ndarray, s: str, at: tuple[int, int]) -> None:
    cv2.putText(bgr, s, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, s, at, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def write_comparison(
    demo: Recording,
    demo_perception: Perception,
    act: Recording,
    act_perception: Perception,
    act_meta: dict[str, Any],
    out: pathlib.Path,
    times: list[float],
    base_from_camera: np.ndarray | None = None,
) -> pathlib.Path:
    """A sheet of the demo and an act side by side at each of ``times`` (demo seconds), one row each."""
    keypoints = demo.annotations.keypoints if demo.annotations else []
    clock = act_clock(act_meta.get("targets") or [], keypoints)
    shapes_d, shapes_a = surfaces(demo, demo_perception), surfaces(act, act_perception)
    rows = []
    t0 = float(demo.origin if demo.origin is not None else demo.arm.t[0])
    for td in times:
        left = render(
            demo,
            demo_perception,
            frame_at(demo_perception, t0 + td),
            shapes=shapes_d,
            base_from_camera=base_from_camera,
            demo_time=td,
            title="demo",
        )
        at = act_time_at(clock, act_perception.t, td)
        if at is None:
            right = np.full_like(left, 25)
            _text(right, f"the act did not reach demo {td:.2f} s", (8, 20))
        else:
            right = render(
                act, act_perception, frame_at(act_perception, at), shapes=shapes_a, demo_time=td, title="act"
            )
        rows.append(pair(left, right))
    cv2.imwrite(str(out), np.vstack(rows))
    return out
