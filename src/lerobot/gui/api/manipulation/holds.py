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

import asyncio
import json
import pathlib
import time
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from lerobot.showservo import motion as core

from . import acquisition, camera, planning, recording, rendering, runtime, tracking


def _tip_speed(demo: runtime._Demo) -> np.ndarray:
    """The demo's fingertip speed at every sample, m/s, from its neighbours."""
    pos, n = demo.tips[:, :3, 3], len(demo.t)
    before, after = np.maximum(np.arange(n) - 1, 0), np.minimum(np.arange(n) + 1, n - 1)
    return np.linalg.norm(pos[after] - pos[before], axis=1) / np.maximum(demo.t[after] - demo.t[before], 1e-6)


def _firm_grip(demo: runtime._Demo) -> int | None:
    """The demo sample its grip became firm, where its hold begins (:func:`core.demo_holding`); None when the
    gripper never stopped short of its command in a stretch the arm follows."""
    holding = planning._holding(demo)
    return None if holding is None else holding["grip"]


def _hold_window(demo: runtime._Demo, window: str) -> tuple[float, float] | None:
    """The demo's time span a hold is measured over, or None. "grip": from the firm grip until the arm moves again,
    the object gripped and still where it lay, within the follow the hold began in. "carry": from the end of that
    follow to the last line the demo made holding, or the end of the pause it is marked in, after any shift the lift
    caused. None without a hold, or for "carry" without a line made holding."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    holding = planning._holding(demo)
    if holding is None:
        return None
    frames = core.key_frames(demo.keypoints)
    speed, n = _tip_speed(demo), len(demo.t)
    end = frames[holding["begins"]]["t"]
    i_end = int(np.argmin(np.abs(demo.t - end)))
    if window == "grip":
        i = holding["grip"]
        j = i
        while j + 1 <= i_end and speed[j + 1] < core.HOLD_STILL_M_S:
            j += 1
        return float(demo.t[i]), float(demo.t[j])
    held_lines = [
        k["t"]
        for m, k in enumerate(frames)
        if k["move"] == "line" and core.held_at(holding, demo.t, frames, m)
    ]
    if not held_lines:
        return None
    last = max(held_lines)
    # A pause marked at its start goes on after the last pre-place: still the same hold until the arm moves again
    # or the gripper starts to open.
    grip = demo.q_cmd[:, MOTOR_NAMES.index("gripper")]
    j = int(np.argmin(np.abs(demo.t - last)))
    while (
        j + 1 < n
        and speed[j + 1] < core.HOLD_STILL_M_S
        and abs(grip[j + 1] - grip[i_end]) <= core.GRASP_HELD_SHORT
    ):
        j += 1
    return end, max(last, float(demo.t[j]))


def _still_held_frames(
    demo: runtime._Demo, obj: str, window: str = "carry", need_seen: bool = True
) -> list[tuple[int, int]]:
    """Where the demo held the picked object still within a hold ``window`` (:func:`_hold_window`): ``(stream frame,
    demo sample)`` pairs with the fingertip slower than :data:`core.HOLD_STILL_M_S` and the object in its own track;
    nearest the window's end first, at least three frames apart, a few more than a hold needs. Pre: a place is
    marked on a stream demo."""
    span = _hold_window(demo, window)
    if span is None:
        return []
    start, last = span
    o = demo.objects[obj]
    times = recording._stream_times(demo.recording) - demo.t0
    speed = _tip_speed(demo)
    picked: list[tuple[int, int]] = []
    edge = 1e-3  # a frame on the window's edge stays in it, whatever the stream clock's rounding
    inside = np.flatnonzero((times >= start - edge) & (times <= last + edge))
    for f in sorted(inside, key=lambda f: abs(times[f] - last)):
        i = int(np.argmin(np.abs(demo.t - times[f])))
        if speed[i] >= core.HOLD_STILL_M_S or (need_seen and (not o["seen"][f] or not np.any(o["masks"][f]))):
            continue
        if all(abs(int(f) - g) >= 3 for g, _ in picked):
            picked.append((int(f), i))
        if len(picked) >= core.HOLD_VIEWS + 2:
            break
    return picked


async def _demo_hold(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, stopped: Callable[[], bool], window: str = "carry"
) -> tuple[dict | None, str]:
    """``(hold, problem)``: how the demo held the picked object in a hold ``window`` (:func:`_hold_window`), as
    :func:`core.average_hold` reports it. A find of the object's view in the demo on each frame where the demo held
    it still there, seen from the gripper. A measured hold is kept on the demo for the marks, window and calibration
    it was measured with, and so is a failure the worker measured; one it did not answer is asked again next time.
    Pre: a place is marked on a stream demo."""
    span = _hold_window(demo, window)
    o = demo.objects[obj]
    key = json.dumps(
        [obj, int(o["frame"]), window, span, np.round(t_bc, 5).tolist(), "surface-constrained-track"]
    )
    if key in demo.holds:
        return demo.holds[key], ""
    by_track = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, _demo_hold_by_track, demo, obj, t_bc, window
    )
    if by_track is not None:  # a track that does not follow it is read again next time: it costs no worker
        demo.holds[key] = {**by_track, "window": window, "method": "track"}
        return demo.holds[key], ""
    key = json.dumps(
        [obj, int(o["frame"]), window, span, np.round(t_bc, 5).tolist(), "surface-constrained-find"]
    )
    if key in demo.holds:
        kept = demo.holds[key]
        return (None, kept["problem"]) if "problem" in kept else (kept, "")
    frames = _still_held_frames(demo, obj, window)
    if not frames:
        where = (
            "after the gripper closed on it, before the lift"
            if window == "grip"
            else "between the grasp end and the last pre-place"
        )
        if _still_held_frames(demo, obj, window, need_seen=False):
            return None, f"{obj} is hidden in the gripper {where}: its own track does not see it there"
        return None, f"the demo never holds {obj} still in the gripper {where}"
    intr = await asyncio.get_event_loop().run_in_executor(runtime._RENDER_EXECUTOR, _recording_intr, demo)
    holds, failures, answered = [], Counter(), True
    for f, i in frames:
        rgb, depth = await asyncio.get_event_loop().run_in_executor(
            runtime._RENDER_EXECUTOR, recording._stream_frame, demo.recording, f
        )
        log = {"why": "measuring how the demo held it", "came_from": {"demo_frame": int(f)}}
        found = await acquisition._locate(
            obj, rgb, depth, intr, acquisition._deepest_pixel(o["masks"][f], rgb.shape), stopped, log=log
        )
        if stopped():
            return None, "stopped"
        answered = answered and bool(found.get("answered"))
        if found["ok"] and not acquisition._weak(found):
            holds.append(np.linalg.inv(demo.tips[i]) @ t_bc @ found["delta"] @ np.linalg.inv(t_bc))
        else:
            failures[found["reason"] or acquisition._weak(found) or "view not verified"] += 1
    points = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, planning._object_points, demo, obj, t_bc
    )
    avg = core.average_hold(holds, points.mean(axis=0))
    if avg is None:
        why = "; ".join(f"{n} view(s): {reason}" for reason, n in failures.items())
        problem = (
            f"{obj} is not visible enough in the gripper in the demo: {len(holds)} of {len(frames)} still views "
            f"matched the demo's view of it ({why or 'their poses disagree'})"
        )
        if answered:
            demo.holds[key] = {"problem": problem}
        return None, problem
    avg["window"], avg["method"] = window, "find"
    demo.holds[key] = avg
    return avg, ""


def _moves_with_tip(demo: runtime._Demo, obj: str, t_bc: np.ndarray, span: tuple[float, float]) -> bool:
    """Does ``obj``'s own track show it moving with the fingertip over ``span`` (demo seconds): its middle wandering
    under half as far in the fingertip's frame as in the base frame, over the frames its track sees it? An object left
    lying, or a track that lost it and stayed put, wanders the other way."""
    o = demo.objects[obj]
    times = recording._stream_times(demo.recording) - demo.t0
    c = planning._object_points(demo, obj, t_bc).mean(axis=0)
    inv_bc = np.linalg.inv(t_bc)
    in_tip, in_base = [], []
    for f in np.flatnonzero((times >= span[0]) & (times <= span[1])):
        if not o["seen"][f]:
            continue
        m = t_bc @ np.asarray(o["deltas"][f], dtype=float) @ inv_bc
        p = m[:3, :3] @ c + m[:3, 3]
        tip = demo.tips[int(np.argmin(np.abs(demo.t - times[f])))]
        in_tip.append(np.linalg.inv(tip)[:3, :3] @ p + np.linalg.inv(tip)[:3, 3])
        in_base.append(p)
    if len(in_base) < core.HOLD_MIN_VIEWS:
        return False

    def wander(ps: list[np.ndarray]) -> float:
        a = np.stack(ps)
        return float(np.median(np.linalg.norm(a - np.median(a, axis=0), axis=1)))

    return wander(in_tip) < 0.5 * wander(in_base)


def _demo_hold_by_track(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, window: str
) -> dict[str, Any] | None:
    """How the demo held ``obj`` in a hold ``window`` by its own track, as :func:`core.average_hold` reports it: its
    motion from the demo's view on each frame the demo held it still there, seen from the gripper; None when its track
    does not show it moving with the fingertip over the hold (:func:`_moves_with_tip`) or too few still views agree.
    Both reference and live holds must pass the same measured-surface policy;
    comparison against an uncorrected historical tilt would reintroduce it."""
    holding = planning._holding(demo)
    span = _hold_window(demo, window)
    if holding is None or span is None:
        return None
    release = len(demo.t) - 1 if holding["release"] is None else holding["release"]
    if not _moves_with_tip(demo, obj, t_bc, (float(demo.t[holding["grip"]]), float(demo.t[release]))):
        return None
    views = _demo_surface_views(demo, obj, _still_held_frames(demo, obj, window))
    holds = [
        np.linalg.inv(demo.tips[i]) @ t_bc @ np.asarray(check["pose"]) @ np.linalg.inv(t_bc)
        for i, check in views
        if check.get("pose") is not None
    ]
    avg = core.average_hold(holds, planning._object_points(demo, obj, t_bc).mean(axis=0))
    if avg is not None:
        avg["surface_constraints"] = [check for _, check in views]
    return avg


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


def _held_track_views(demo: runtime._Demo, obj: str, since: float) -> list[tuple[float, np.ndarray]]:
    """Accepted, surface-checked poses of ``obj`` read after ``since``, each from the demo's view (camera
    frame): the session object's against its find of the demo's view, any other's as kept."""
    with runtime._state.lock:
        teach, raw = runtime._state.teach, list(runtime._state.track.hold_views)
        target = runtime._state.targets.get(obj)
        other = list(target.hold_views) if target is not None else []
    if obj == (planning._session_object(demo) or ""):
        ref = ((teach.keypoints.get("ref") if teach is not None else None) or {}).get("delta")
        start = max(since, (teach.stamp or 0.0) if teach is not None else 0.0)
        if ref is None:
            return []
        return [(s, np.asarray(d, dtype=float) @ np.asarray(ref, dtype=float)) for s, d in raw if s > start]
    return [(s, m) for s, m in other if s > since]


async def _live_hold_by_track(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, stopped: Callable[[], bool]
) -> tuple[dict | None, str]:
    """``(hold, problem)``: how ``obj`` sits in the gripper by accepted surface-checked poses (:func:`_held_track_views`)
    over the frames read from now, the arm standing still, each seen from the gripper where it stands; as
    :func:`core.average_hold` reports it. Pre: the arm stands still holding ``obj``."""
    from .. import jog

    since, t0 = time.time(), time.monotonic()
    points = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, planning._object_points, demo, obj, t_bc
    )
    views: list[tuple[float, np.ndarray]] = []
    while len(views) < core.HOLD_VIEWS + 2 and time.monotonic() - t0 < runtime.HOLD_TRACK_WAIT_S:
        if stopped():
            return None, "stopped"
        await asyncio.sleep(runtime.ACT_TICK_S)
        views = _held_track_views(demo, obj, since)
    cur = jog.current_tip_and_anchor()
    if cur is None:
        return None, "the arm went away"
    holds = [np.linalg.inv(cur[0]) @ t_bc @ m @ np.linalg.inv(t_bc) for _s, m in views]
    avg = core.average_hold(holds, points.mean(axis=0))
    if avg is None:
        return None, f"{obj}'s track gave {len(views)} views in the gripper, too few that agree"
    return {**avg, "method": "track"}, ""


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


def _held_click(
    points: np.ndarray,
    motion: np.ndarray,
    t_bc: np.ndarray,
    intr: dict[str, float],
    depth_m: np.ndarray,
    tolerance_m: float,
) -> list[int] | None:
    """Click a depth-supported part of the predicted held surface, never its covered centre.

    The prediction proposes a search region, not a pose measurement. Missing depth and nearer occluders cannot
    seed segmentation. Use the existing depth-agreement tolerance; all subsequent matching gates still apply.
    """
    import cv2

    moved = points @ motion[:3, :3].T + motion[:3, 3]
    cam = (moved - t_bc[:3, 3]) @ t_bc[:3, :3]
    cam = cam[np.isfinite(cam).all(axis=1) & (cam[:, 2] > 1e-6)]
    uv = rendering._project_cam(intr, cam)
    h, w = depth_m.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    uv, cam = uv[inside].astype(int), cam[inside]
    surface = np.full((h, w), np.inf, np.float32)
    np.minimum.at(surface, (uv[:, 1], uv[:, 0]), cam[:, 2])
    # The reference samples every fourth pixel. Fill the same five-pixel neighbourhood as the old projected
    # click mask, carrying the nearest surface depth too. Apply the depth test AFTER filling, so padding cannot
    # put the click back onto a foreground finger or a pixel with missing depth.
    surface = cv2.erode(surface, np.ones((5, 5), np.uint8))
    supported = (
        np.isfinite(surface)
        & np.isfinite(depth_m)
        & (depth_m > 0)
        & (np.abs(surface - depth_m) <= tolerance_m)
    )
    return acquisition._deepest_pixel(supported, depth_m.shape)


async def _held_view() -> (
    tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray | None] | None
):
    """A view of the held object as it is now: ``(rgb, depth, intrinsics, fingertip, live mask)``, the fingertip read
    just before the frame and the live track's mask while it tracks, or None without the arm. Pre: the arm stands
    still, or the fingertip is not where the frame shows it."""
    from .. import jog

    cur = jog.current_tip_and_anchor()
    if cur is None:
        return None
    rgb, depth_m, intr = await camera._frame()
    with (
        runtime._state.lock
    ):  # a finger over the predicted middle: the live track's view is the next place to click
        test, state = runtime._state.test, (runtime._state.track.last or {}).get("state")
    mask = test.result.get("live_mask") if state == "tracking" and test is not None else None
    return rgb, depth_m, intr, cur[0], mask


async def _live_hold(
    demo: runtime._Demo,
    obj: str,
    demo_hold: np.ndarray,
    t_bc: np.ndarray,
    stopped: Callable[[], bool],
    views: list[tuple[np.ndarray, np.ndarray, dict[str, float], np.ndarray, np.ndarray | None]] | None = None,
    method: str = "find",
) -> tuple[dict | None, str]:
    """``(hold, problem)``: how the held object sits in the gripper, as :func:`core.average_hold` reports it. Prefer
    the demo's measurement method; after a failed live track, try fresh reference matches within one more track
    wait budget. Both methods measure motion from the SAME designated demo view, so their hold frames agree.
    Finds of its view in the demo on fresh frames (:func:`_held_view`), each seen from the gripper where FK had it
    then, with the click where the demo's hold puts the object; after a view that fails, where the live track had
    it, and back. ``views`` taken earlier are read instead of fresh ones: the grip's, read while the act goes on.
    Pre: without ``views``, the arm stands still with the object gripped."""
    track_problem, refind_until = "", None
    if method == "track":
        if views is not None:
            return None, "not measured: the demo's hold was read from its track, these views are frames"
        tracked, track_problem = await _live_hold_by_track(demo, obj, t_bc, stopped)
        if tracked is not None or stopped():
            return tracked, track_problem
        refind_until = time.monotonic() + runtime.HOLD_TRACK_WAIT_S

    def expired() -> bool:
        return refind_until is not None and time.monotonic() >= refind_until

    points = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, planning._object_points, demo, obj, t_bc
    )
    pending = None if views is None else list(views)
    holds, tries, choice = [], 0, 0
    while len(holds) < core.HOLD_VIEWS and tries < core.HOLD_VIEWS + 2:
        if stopped():
            return None, "stopped"
        if expired():
            break
        view = (pending.pop(0) if pending else None) if pending is not None else await _held_view()
        if view is None:
            if pending is not None:
                break
            return None, "the arm went away"
        tries += 1
        rgb, depth_m, intr, tip, mask = view
        with runtime._state.lock:
            tolerance_m = runtime._state.depth_tol_m
        clicks = [
            (
                "the depth-supported predicted surface",
                _held_click(points, tip @ demo_hold, t_bc, intr, depth_m, tolerance_m),
            )
        ]
        if mask is not None:
            clicks.append(("where the live track has it", acquisition._deepest_pixel(mask, rgb.shape)))
        clicks = [c for c in clicks if c[1] is not None]
        if not clicks:
            return None, f"{obj} has no depth-supported predicted surface or live mask in this view"
        came, click = clicks[choice % len(clicks)]
        log = {"why": "act: the held object, in the gripper", "came_from": {"click": came}}
        found = await acquisition._locate(
            obj, rgb, depth_m, intr, click, lambda: stopped() or expired(), log=log
        )
        if stopped():
            return None, "stopped"
        if found["ok"] and not acquisition._weak(found):
            holds.append(np.linalg.inv(tip) @ t_bc @ found["delta"] @ np.linalg.inv(t_bc))
        else:
            choice += 1  # the other place to click, while there is one
    avg = core.average_hold(holds, points.mean(axis=0))
    if avg is None:
        return None, (
            (f"{track_problem}; " if track_problem else "")
            + f"{obj} is not visible enough in the gripper to place it: {len(holds)} of {tries} views matched the "
            "demo's view of it"
        )
    return {**avg, "method": "find", "track_problem": track_problem}, ""
