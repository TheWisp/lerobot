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

"""Manipulation overlays, pose comparisons and projected geometry."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from lerobot.showservo import motion as core, pose_verification

from . import acquisition, camera, groups, planning, replay, runtime, tracking

router = APIRouter()


@router.get("/act/recovered-preview.jpg")
async def recovered_grasp_preview() -> Response:
    """Show the proposed fingertip paths on the find's own RGB frame, including both measured end assignments.

    This is a geometric projection for operator review, not a collision or reach certificate.
    """
    import cv2

    with runtime._state.lock:
        demo, teach = runtime._state.demo, runtime._state.teach
    ref = {} if teach is None else (teach.keypoints.get("ref") or {})
    constraint = ref.get("surface_constraint") or {}
    if demo is None or teach is None or teach.rgb is None or constraint.get("state") != "recovered":
        raise HTTPException(409, "no recovered grasp pose to preview; find the object again")
    t_bc = camera._t_base_cam()
    bgr = cv2.cvtColor(teach.rgb, cv2.COLOR_RGB2BGR)
    bgr = cv2.copyMakeBorder(bgr, 88, 0, 0, 0, cv2.BORDER_CONSTANT, value=(20, 20, 20))
    image = bgr[88:]
    mask = teach.keypoints.get("mask")
    if mask is not None:
        _outline(image, np.asarray(mask, dtype=bool), (80, 220, 80))
    chosen = np.asarray(ref["delta"], dtype=float)
    poses = [chosen]
    for item in constraint.get("equivalent_poses") or []:
        pose = np.asarray(item.get("pose"), dtype=float)
        if (
            pose.shape == (4, 4)
            and np.isfinite(pose).all()
            and not any(np.allclose(pose, old, atol=1e-8) for old in poses)
        ):
            poses.append(pose)
    colours = [(0, 215, 255), (255, 90, 210)]
    shown = 0
    obj = planning._session_object(demo)
    frames = core.key_frames(demo.keypoints)
    holding = next((h for h in planning._holdings(demo) if h["frame"] == obj), None)
    for i, pose in enumerate(poses[:2]):
        with contextlib.suppress(AssertionError, KeyError, ValueError):
            path = planning._act_preview(demo, np.eye(4), t_bc, preview_pose=pose)
            if path is not None:
                _draw_path(image, t_bc, teach.intr, path, colour=colours[i])
                shown += 1
            if holding is not None and frames[holding["begins"]]["frame"] == obj:
                grasp_t = frames[holding["begins"]]["t"]
                sample = int(np.argmin(np.abs(demo.t - grasp_t)))
                grasp = planning._delta_base(demo, np.eye(4), t_bc, preview_pose=pose) @ demo.tips[sample]
                _draw_tool(
                    image,
                    t_bc,
                    teach.intr,
                    grasp,
                    f"{'selected' if i == 0 else 'other'} jaws",
                    colours[i],
                    arrow_colour=colours[i],
                )
    cv2.putText(
        bgr, "Saved find: green object outline", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 220, 80), 2
    )
    legend = "Yellow: selected path + jaws" + (" | Pink: other end" if len(poses) > 1 else "")
    cv2.putText(bgr, legend, (8, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 2)
    cv2.putText(
        bgr,
        f"2D projection only ({shown} path(s)); reach and collisions unverified",
        (8, 73),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (80, 160, 255),
        2,
    )
    return Response(content=camera._jpeg(bgr), media_type="image/jpeg")


def _outline(bgr: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> None:
    import cv2

    contours, _h = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, colour, 2)


def _draw_tool(
    bgr: np.ndarray,
    t_base_cam: np.ndarray,
    intr: dict[str, float],
    pose: np.ndarray,
    label: str,
    colour: tuple[int, int, int] = (255, 255, 255),
    *,
    arrow_colour: tuple[int, int, int] = (0, 220, 255),
) -> None:
    """The fingertip pose on the image: a cross at the tip, the jaw line through it, the approach as an arrow.

    The tip frame is the wrist link's: the jaws close along its x axis and the
    fingers point along -y, so the line is the jaw direction and the
    arrow points the way the fingers do. Seen from above, a vertical approach
    collapses the arrow to the cross; a leaning one shows it.
    """
    import cv2

    pose = np.asarray(pose, dtype=float)
    p, x, y = pose[:3, 3], pose[:3, 0], pose[:3, 1]
    tip = camera._project(t_base_cam, intr, p)
    if tip is None:
        return
    a, b = (
        camera._project(t_base_cam, intr, p + runtime.JAW_HALF_M * x),
        camera._project(t_base_cam, intr, p - runtime.JAW_HALF_M * x),
    )
    back = camera._project(t_base_cam, intr, p + runtime.APPROACH_M * y)
    if a is not None and b is not None:
        cv2.line(bgr, a, b, colour, 3)
    if back is not None:
        cv2.arrowedLine(bgr, back, tip, arrow_colour, 2, tipLength=0.3)
    cv2.drawMarker(bgr, tip, colour, cv2.MARKER_CROSS, 24, 2)
    cv2.putText(bgr, label, (tip[0] + 10, tip[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)


@router.get("/test.jpg")
async def test_jpeg() -> Response:
    import cv2

    with runtime._state.lock:
        teach, test = runtime._state.teach, runtime._state.test
    if test is None or teach is None:
        raise HTTPException(404, "no test capture")
    bgr = cv2.cvtColor(test.rgb, cv2.COLOR_RGB2BGR)
    r = test.result
    if r.get("ok"):
        if "live_uv" in r:
            for u, v in r["live_uv"]:
                cv2.circle(bgr, (int(u), int(v)), 3, (60, 200, 60), 1)
        if "live_mask" in r:
            _outline(bgr, r["live_mask"], (60, 200, 60) if r.get("mode") != "features" else (255, 0, 255))
        if test.transported is not None:
            _draw_tool(bgr, camera._t_base_cam(), teach.intr, test.transported, "go here")
    else:
        cv2.putText(bgr, r.get("reason", "no match"), (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return Response(content=camera._jpeg(bgr), media_type="image/jpeg")


def _project_cam(intr: dict[str, float], pts: np.ndarray) -> np.ndarray:
    """Camera-frame points to pixels, (N, 2); points behind the camera are dropped."""
    p = np.asarray(pts, dtype=float).reshape(-1, 3)
    p = p[p[:, 2] > 1e-6]
    return np.stack(
        [intr["fx"] * p[:, 0] / p[:, 2] + intr["cx"], intr["fy"] * p[:, 1] / p[:, 2] + intr["cy"]], axis=1
    )


def _draw_frame(bgr: np.ndarray, intr: dict[str, float], delta: np.ndarray, centre_teach: np.ndarray) -> None:
    """The object's frame carried by the motion: origin at the taught centre, axes as taught (the
    camera's at teach time), x red, y green, z blue. A steady frame is a steady pose."""
    import cv2

    r, t = delta[:3, :3], delta[:3, 3]
    origin = r @ centre_teach + t
    pts = np.vstack(
        [origin, origin + runtime.FRAME_AXIS_M * r.T]
    )  # the three carried axes, one per row of r.T
    uv = _project_cam(intr, pts)
    if len(uv) < 4:
        return
    o = (int(uv[0, 0]), int(uv[0, 1]))
    for k, colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(bgr, o, (int(uv[k + 1, 0]), int(uv[k + 1, 1])), colour, 2, cv2.LINE_AA)


def _found_mask(
    intr: dict[str, float], points: np.ndarray, delta: np.ndarray, shape: tuple[int, ...]
) -> np.ndarray | None:
    """Where an object's surface from the demo's view lands in an image of ``shape`` when carried by a find's motion
    (camera frame): its projected points, their gaps closed. None when fewer than three land in the image."""
    import cv2

    d = np.asarray(delta, dtype=float)
    uv = _project_cam(intr, np.asarray(points, dtype=float) @ d[:3, :3].T + d[:3, 3])
    h, w = shape[:2]
    uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)].astype(int)
    if len(uv) < 3:
        return None
    mask = np.zeros((h, w), np.uint8)
    for u, v in uv:
        cv2.circle(mask, (int(u), int(v)), 3, 1, -1)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(
        bool
    )  # the points' gaps only


def _draw_found(
    bgr: np.ndarray,
    intr: dict[str, float],
    points: np.ndarray,
    delta: np.ndarray,
    label: str,
    colour: tuple[int, int, int] = runtime.FOUND_COLOUR,
) -> None:
    """An object where a find placed it: the outline of its surface from the demo's view (:func:`_found_mask`), its
    frame at the surface's centre, and its name. Not tracked, it stays where the find put it until the next find."""
    import cv2

    mask = _found_mask(intr, points, delta, bgr.shape)
    if mask is None:
        return
    _outline(bgr, mask, colour)
    _draw_frame(bgr, intr, np.asarray(delta, dtype=float), np.asarray(points, dtype=float).mean(axis=0))
    ys, xs = np.nonzero(mask)
    top = int(np.argmin(ys))
    cv2.putText(
        bgr,
        label,
        (int(xs[top]) - 20, max(int(ys[top]) - 8, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colour,
        2,
    )


def _draw_track_estimate(bgr, intr, result, arrays, name) -> str:
    """Draw the current raw fit using its own model in teach coordinates, even if following rejected it.

    Neither the current demo nor a previous fit can stand in for missing recorded data. A model is the
    confirmed surface learned so far, including points now occluded; its outline is a prediction, not a visible mask.
    The returned availability says nothing about acceptance by either the following or held-pose gate.
    """
    delta = arrays.get("delta")
    if not result.get("ok") or delta is None:
        return "unavailable"
    delta = np.asarray(delta, dtype=float)
    if delta.shape != (4, 4) or not np.isfinite(delta).all():
        return "unavailable"
    cloud = arrays.get("model_xyz")
    if cloud is None:
        return "available (surface unavailable)"
    cloud = np.asarray(cloud, dtype=float)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return "available (surface unavailable)"
    validity = arrays.get("model_valid")
    availability = "available"
    if validity is None:
        # Older recordings omitted this distinction. Preserve their diagnostic picture without
        # claiming that the tentative/rejected rows in it were a confirmed surface.
        availability += " (surface validity unrecorded)"
    else:
        validity = np.asarray(validity, dtype=bool)
        if validity.shape != (len(cloud),):
            return "available (surface validity invalid)"
        cloud = cloud[validity]
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    cloud = cloud[(cloud @ delta[:3, :3].T + delta[:3, 3])[:, 2] > 0]
    if len(cloud) < 3:
        return "available (surface unavailable)"
    _draw_found(bgr, intr, cloud, delta, f"{name}: raw pose estimate", runtime.POSE_ESTIMATE_COLOUR)
    return availability


def _render_live(rgb, r, result, transported, teach, status) -> bytes:
    """The tracking view: the point groups underneath when acts use them (:func:`_paint_groups`), the mask edge, the
    points that agree, the taught cloud carried by the motion (where the object is believed to be: this frame's view
    when it places the object, otherwise the act's pose of it, its last placing view moved on by the point groups),
    the transported pre-grasp, the object a place goes onto where the act would aim, and a status strip. A blue
    outline also shows the current raw pose's model, independently of the stricter following acceptance."""
    import cv2

    started = time.perf_counter()
    bgr = groups._paint_groups(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    painted = time.perf_counter()
    belief = None  # the object's motion the view draws, and where it comes from
    if result is not None and result.get("ok"):
        belief = (result["delta_cam"], "")
    elif teach is not None:
        with runtime._state.lock:
            held, demo = runtime._state.test, runtime._state.demo
        if held is not None and held.result.get("ok") and held.result.get("delta_cam") is not None:
            name = (planning._session_object(demo) if demo is not None else None) or teach.keypoints.get(
                "concept"
            )
            views = planning._certified_since(None)
            carried = groups._carried(str(name), views[-1][0]) if views else np.eye(4)
            moved = core.motion_summary(carried)["translation_mm"]
            belief = (
                carried @ np.asarray(held.result["delta_cam"], dtype=float),
                f"last view, carried {moved:.0f} mm by the point groups"
                if views and runtime._state.groups_with_acts
                else "last view, held",
            )
    if r.get("mask") is not None:
        _outline(bgr, np.asarray(r["mask"]).astype(bool), (255, 0, 255))
    live = r.get("live_uv")
    if live is not None:
        # White on black reads on any object; a green dot vanished on a green one.
        for u, v in np.asarray(live)[::2]:
            cv2.circle(bgr, (int(u), int(v)), 3, (0, 0, 0), -1)
            cv2.circle(bgr, (int(u), int(v)), 2, (255, 255, 255), -1)
    if belief is not None:
        d = belief[0]
        # The object as known so far (the teach view plus what the tracker has adopted since), carried
        # by the motion: where the tracker believes the whole object is, seen sides and hidden ones.
        cloud = r.get("model_xyz") if result is not None else None
        cloud = np.asarray(teach.keypoints["xyz"] if cloud is None else cloud, dtype=float)
        moved = cloud[::3] @ d[:3, :3].T + d[:3, 3]
        h, w = bgr.shape[:2]
        for u, v in _project_cam(teach.intr, moved):
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(bgr, (int(u), int(v)), 1, (0, 220, 255), -1)
        _draw_frame(bgr, teach.intr, d, np.asarray(teach.keypoints["xyz"], dtype=float).mean(axis=0))
        with contextlib.suppress(HTTPException):
            t_bc = camera._t_base_cam()
            if transported is not None:
                _draw_tool(bgr, t_bc, teach.intr, transported, "start")
            with runtime._state.lock:
                demo = runtime._state.demo
            if demo is not None:
                path = planning._act_preview(demo, d, t_bc)
                if path is not None:
                    _draw_path(bgr, t_bc, teach.intr, path)
    path_drawn = time.perf_counter()
    with runtime._state.lock:
        demo = runtime._state.demo
    estimate = "unavailable"
    if teach is not None:
        name = (
            (planning._session_object(demo) if demo is not None else None)
            or teach.keypoints.get("concept")
            or "object"
        )
        estimate = _draw_track_estimate(bgr, teach.intr, r, r, name)
    others = (
        [] if demo is None else [(n, acquisition._located(demo, n)) for n in planning._other_objects(demo)]
    )
    for onto, located in others:
        if located is None or not located.get("ok") or located.get("delta") is None or teach is None:
            continue
        aim = np.asarray(located["delta"], dtype=float)
        if located.get("stamp") is not None:  # where the act would aim: its last placing view, carried
            aim = groups._carried(onto, float(located["stamp"])) @ aim
        with contextlib.suppress(
            OSError, KeyError, ValueError
        ):  # a demo recording that went away: no outline
            _draw_found(bgr, teach.intr, tracking._view_points(demo, onto), aim, onto)
    objects_drawn = time.perf_counter()
    state = status.get("state") or ""
    strip = (
        f"[{status.get('algo')}] {state} | {status.get('fps') or 0:.0f} fps | {status.get('ms') or 0:.0f} ms"
    )
    if status.get("n_inliers") is not None:
        strip += f" | {status['n_inliers']} of {status.get('n_matches')} agree"
    if status.get("centre_shift_mm") is not None:
        strip += f" | moved {status['centre_shift_mm']:.0f} mm"
    if status.get("arm_turn_deg") is not None:
        strip += f" | gripper turns {status['arm_turn_deg']:.0f} deg, leans {status['arm_lean_deg']:.0f} deg"
    elif status.get("yaw_deg") is not None:
        strip += f" | turned {status['yaw_deg']:.0f} deg"
    elif (status.get("motion") or {}).get("rotation_deg") is not None:
        strip += f" | rotated {status['motion']['rotation_deg']:.0f} deg"
    if status.get("card_points"):
        strip += f" | card {status['card_points']}"
    if status.get("reason"):
        strip += f" | {status['reason']}"
    if belief is not None and belief[1]:
        strip += f" | pose: {belief[1]}"
    colour = {"tracking": (60, 230, 60), "occluded": (0, 200, 255)}.get(state, (0, 0, 255))
    cv2.rectangle(bgr, (0, 0), (bgr.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(bgr, strip, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)
    badges = [_find_badge((teach.keypoints.get("ref") if teach is not None else None) or {})]
    for onto, located in others:
        if located is None:
            continue
        with runtime._state.lock:
            target = runtime._state.targets.get(onto)
            tracked = None if target is None else target.last.get("state")
        badges.append(_find_badge(located, f"{onto}, {tracked or 'not tracked'}"))
    y = 32
    for badge in badges:  # each on its own line: the strip above already runs off the frame's edge
        if badge is None:
            continue
        text, badge_colour = badge
        (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(bgr, (0, y), (w + 16, y + 26), (0, 0, 0), -1)
        cv2.putText(bgr, text, (8, y + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, badge_colour, 2, cv2.LINE_AA)
        y += 28
    replay._replay_text(
        bgr,
        f"Blue raw P2P outline: {estimate}; control pose: "
        f"{'accepted' if result is not None and result.get('ok') else 'rejected'}",
        (8, y + 18),
        runtime.POSE_ESTIMATE_COLOUR,
    )
    checks = status.get("pose_verification")
    encoded = camera._jpeg(_pose_verification_panel(bgr, checks, len(checks)) if checks else bgr)
    finished = time.perf_counter()
    status["render_stages_ms"] = {
        "groups": round((painted - started) * 1000, 1),
        "held_and_path": round((path_drawn - painted) * 1000, 1),
        "other_objects": round((objects_drawn - path_drawn) * 1000, 1),
        "text_and_jpeg": round((finished - objects_drawn) * 1000, 1),
    }
    return encoded


def _pose_verification_panel(bgr: np.ndarray, checks: list[dict] | None, slots: int) -> np.ndarray:
    """Keep diagnostic text outside the camera image so object outlines stay visible.

    Replay reserves the same number of rows for every frame, including frames
    without diagnostics, so exports retain a fixed size.
    """
    panel = np.zeros((16 + 48 * slots, bgr.shape[1], 3), dtype=bgr.dtype)
    _draw_pose_verification(panel, checks, 18)
    return np.vstack([bgr, panel])


def _draw_pose_verification(bgr: np.ndarray, checks: list[dict] | None, y: int) -> None:
    """A visible warning independent of following acceptance; unknown is not agreement."""
    if checks is None:
        replay._replay_text(bgr, "Pose verification: not recorded", (8, y), (180, 180, 180), 0.4)
        return
    for line, conflict in pose_verification.lines(checks):
        replay._replay_text(bgr, line, (8, y), (30, 100, 255) if conflict else (200, 200, 200), 0.42)
        y += 16


def _find_badge(ref: dict[str, Any], label: str = "find") -> tuple[str, tuple[int, int, int]] | None:
    """The live view's line about a find of the demo's view: strong in green, weak in orange with what to do; None when
    there is no such find or its strength is unknown. ``label`` names it: the find the track starts from, or the
    object a place goes onto."""
    strong, _share = core.find_strength(ref.get("inliers"), ref.get("card_points"))
    if not ref.get("ok") or strong is None:
        return None
    counts = f"{ref['inliers']} of {ref['card_points']} points"
    if strong:
        return f"{label}: strong, {counts}", (60, 230, 60)
    return f"{label}: weak, {counts}; turn it closer to how it lay in the demo", (0, 165, 255)


def _draw_path(
    bgr: np.ndarray,
    t_bc: np.ndarray,
    intr: dict[str, float],
    tips: np.ndarray,
    *,
    colour: tuple[int, int, int] = (255, 200, 0),
) -> None:
    """The fingertip path on the image, start as a dot, end as a square."""
    import cv2

    stride = max(1, len(tips) // 200)
    pts = [p for p in (camera._project(t_bc, intr, tip[:3, 3]) for tip in tips[::stride]) if p is not None]
    for a, b in zip(pts, pts[1:], strict=False):
        cv2.line(bgr, a, b, colour, 2)
    if pts:
        cv2.circle(bgr, pts[0], 5, colour, -1)
        x, y = pts[-1]
        cv2.rectangle(bgr, (x - 4, y - 4), (x + 4, y + 4), colour, -1)
