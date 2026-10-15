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

"""Tracking acceptance, measured-surface constraints and accepted pose history."""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import pathlib
import time
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from lerobot.showservo import motion as core, pose_verification, surface_constraint, symmetry

from . import acquisition, camera, planning, recording, rendering, runtime, worker

logger = logging.getLogger(__name__)
router = APIRouter()


def _seen_with_depth(
    depths: runtime._PointDepths,
    r: dict[str, Any],
    key: str,
    session: Any,
    depth_m: np.ndarray | None,
    intr: dict[str, float] | None,
    frames: list[tuple[float, np.ndarray]],
    stamp: float,
) -> tuple[int, int, dict[int, np.ndarray], np.ndarray] | None:
    """The depth check on one object of a tracked frame (:func:`core.depth_seen`): how many of its tracks are seen with
    a depth that agrees with where they were when a view last placed it, moved since as most of them agree, starting
    from the point groups' motion of it (``frames``, stamped like ``stamp``, when this frame was read). ``key`` names its track arrays in ``r`` with a ``{}`` for idx,
    uv and vis; ``session`` is the tracker session they come from, whose start drops the places kept from another.
    Post: ``(count, judged, now, move)`` for :func:`_depth_accept`, or None when the check is off, or the frame has no
    depth or no tracks to judge."""
    with runtime._state.lock:
        on, tol = runtime._state.depth_check, runtime._state.depth_tol_m
    names = [key.format(n) for n in ("idx", "uv", "vis")]
    if not on or depth_m is None or intr is None or any(r.get(n) is None for n in names):
        return None
    if session != depths.session:
        depths.session, depths.t, depths.at = session, None, {}
    carry = core.carried_between(frames, depths.t, stamp) if depths.t is not None else np.eye(4)
    return core.depth_seen(*(r[n] for n in names), depth_m, intr, depths.at, carry, tol)


def _depth_accept(
    depths: runtime._PointDepths,
    seen: tuple[int, int, dict[int, np.ndarray], np.ndarray] | None,
    stamp: float,
) -> None:
    """A view placed the object: its tracks are expected where it saw them from now on, the others moved with it
    (:func:`core.depth_expect`)."""
    if seen is not None:
        _count, _judged, now, move = seen
        depths.at, depths.t = core.depth_expect(depths.at, now, move), stamp


def _apply_others(
    r: dict[str, Any],
    shape: tuple[int, ...],
    depth_m: np.ndarray | None = None,
    intr: dict[str, float] | None = None,
    stamp: float | None = None,
    decisions: dict[str, dict] | None = None,
) -> None:
    """The other objects of a tracked frame's session (``others``, ``other_delta_i``, ``other_mask_i``, and with
    Point2Pose ``other_fit_uv_i``, ``other_fit_inlier_i``, ``other_model_i``): each followed object's share moves its
    last find (:func:`_apply_other`)."""
    with runtime._state.lock:
        targets = list(runtime._state.targets.values())
    for target in targets:
        _apply_other(
            target,
            r,
            shape,
            depth_m,
            intr,
            stamp,
            None
            if decisions is None
            else decisions.get(target.obj, {"pose": None, "reason": "geometry decision missing"}),
        )


def _apply_other(
    target: runtime._TargetTrack,
    r: dict[str, Any],
    shape: tuple[int, ...],
    depth_m: np.ndarray | None,
    intr: dict[str, float] | None,
    stamp: float | None,
    decision: dict | None = None,
) -> None:
    """``target``'s share of a tracked frame moves its last find (its motion since its find, times the find's motion
    from the demo's view, and ``stamp``, when the frame was read) when the view places it; anything else leaves the
    find where it was. It places it when Point2Pose has it, enough of the tracks it began with are seen
    (:func:`core.find_trusted`), and the act's one rule holds (:func:`core.view_places`): ``trust_share`` of its
    tracks seen now (with the depth check, seen with a depth that agrees: :func:`_seen_with_depth`), and its fit
    points, lifted by ``depth_m``, pinning its middle (its key points, carried). Otherwise it stays where it was last
    seen."""
    with runtime._state.lock:
        found = runtime._state.located.get(target.obj) if target.obj is not None else None
    if target.obj is None or found is None or target.anchor is None:
        return
    i = next((k for k, o in enumerate(r.get("others") or ()) if o.get("name") == target.obj), None)
    if i is None:
        target.last = {"state": "not in the session"}
        return
    share = r["others"][i]
    target.n_points = target.n_points or int(share.get("n_tracks") or 0)
    trusted, why = (
        (False, "lost")
        if share.get("lost")
        else core.find_trusted(int(share.get("n_visible") or 0), target.n_points)
    )
    n_tracks = int(share.get("n_tracks") or 0)
    seen = int(share.get("n_visible") or 0) / max(1, n_tracks)
    now_t = time.time() if stamp is None else stamp
    with runtime._state.lock:
        run, need, tol = runtime._state.run, runtime._state.trust_share, runtime._state.depth_tol_m
        frames = (
            list(runtime._state.groups_feed.frames.get(target.obj, ()))
            if runtime._state.groups_with_acts
            else []
        )
    depth = None
    if trusted:
        points = middle = None
        if depth_m is not None and intr is not None and f"other_delta_{i}" in r:
            points = _fit_points_3d(r.get(f"other_fit_uv_{i}"), r.get(f"other_fit_inlier_{i}"), depth_m, intr)
            model = np.asarray(r.get(f"other_model_{i}", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
            d = np.asarray(r[f"other_delta_{i}"], dtype=float)
            middle = d[:3, :3] @ model.mean(axis=0) + d[:3, 3] if len(model) else None
        depth = _seen_with_depth(
            target.depths, r, f"other_track_{{}}_{i}", share.get("session"), depth_m, intr, frames, now_t
        )
        counted = seen if depth is None else depth[0] / max(1, depth[1])
        trusted, why = core.view_places(counted, points, middle, need)
        if depth is not None and not trusted and seen >= need and counted < need:
            why = (
                f"only {counted:.0%} of its points are seen with a depth that agrees within {tol * 1000:.0f} mm; "
                f"a view counts from {need:.0%}"
            )
    if trusted and decision is not None and decision.get("pose") is None:
        trusted, why = False, decision.get("reason", "surface pose rejected")
    entry = None
    if (
        run is not None and f"other_delta_{i}" in r
    ):  # the act's record: what the place object's track said, each frame
        d = np.asarray(r[f"other_delta_{i}"], dtype=float) @ target.anchor
        entry = {
            "t": time.time(),
            "object": target.obj,
            "lost": bool(share.get("lost")),
            "n_visible": share.get("n_visible"),
            "n_tracks": share.get("n_tracks"),
            "trusted": bool(trusted),
            "seen": round(seen, 3),
            "trust_share": need,
            "delta": d.round(5).tolist(),  # the whole motion: its translation alone swings with a small turn
            "surface_constraint": decision,
        }
        if depth is not None:  # the share of its tracks seen with a depth that agrees
            entry["depth_seen"] = round(depth[0] / max(1, depth[1]), 3)
        run.meta.setdefault("target_track", []).append(entry)
    if not (share.get("ok") and trusted and f"other_delta_{i}" in r):
        target.last = {
            "state": "untrusted" if share.get("ok") else "lost",
            "reason": why,
            **({"surface_constraint": decision} if decision is not None else {}),
        }
        return
    mask = None if r.get(f"other_mask_{i}") is None else np.asarray(r[f"other_mask_{i}"]).astype(bool)
    d = np.asarray(r[f"other_delta_{i}"], dtype=float)
    view = d @ target.anchor if decision is None else np.asarray(decision["pose"], dtype=float)
    target.hold_views.append((now_t, view.copy()))
    del target.hold_views[: -runtime.TRACK_HISTORY_MAX]
    _depth_accept(target.depths, depth, now_t)
    with runtime._state.lock:
        found["delta"] = view
        found["stamp"] = stamp
        found["tracked_at"] = time.time()
        if mask is not None:  # where the act's own find clicks it next
            found["mask"] = mask
    target.last = {
        "state": "tracking",
        "n_visible": share.get("n_visible"),
        **({"surface_constraint": decision} if decision is not None else {}),
    }
    if mask is not None:
        acquisition._remember_seen(target.obj, mask, shape, "tracked by Point2Pose")


def _compose_motion(
    result: dict[str, Any], r: dict[str, Any], teach: runtime._Teach, flat: bool, t_bc: np.ndarray | None
) -> np.ndarray | None:
    """Turn the worker's raw fit into the motion the arm uses, in ``result``, and return the transported pre-grasp.

    P2P proposals now pass measured-surface arbitration before this function.
    A supported plane constrains tilt and normal offset; its unobserved tangent
    motion remains the proposal's. Keep that decision ahead of the old opt-in
    resting prior: a real observed tilt must never be forced level. The older
    unconditional face/table/footprint priors regressed the nine reference videos
    and remain off by default. This bounded geometry policy is evaluated
    separately; it does not inherit their benchmark results. ``axis_source``
    distinguishes "measured surface", "fit", opted-in "surface", and "table".
    """
    result["delta_cam"] = np.asarray(r["delta"], dtype=float)
    centroid = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
    ft, ff = r.get("face_teach"), r.get("face_find")
    if core.face_usable(ft) and core.face_usable(ff):  # reported, never applied
        a, b = np.asarray(ft["normal"], dtype=float), np.asarray(ff["normal"], dtype=float)
        result["face_tilt_deg"] = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))))
        result["face_planarity"] = min(ft["planarity"], ff["planarity"])
    ta, tb = r.get("table_teach"), r.get("table_find")
    if r.get("algo") == "depth":
        result.update({"axis_source": "table", "yaw_deg": r.get("yaw_deg"), "symmetric": r.get("symmetric")})
    elif r.get("surface_constraint", {}).get("state") in ("corrected", "consistent"):
        result["axis_source"] = "measured surface"
    elif flat and ta is not None and tb is not None:
        comp = core.compose_with_face(result["delta_cam"], ta, tb, centroid)
        result["delta_cam"] = comp["delta"]
        result.update(
            {
                "axis_source": "surface",
                "yaw_deg": comp["yaw_deg"],
                "surface_tilt_deg": comp["face_tilt_deg"],
                "fit_axis_tilt_deg": comp["fit_axis_tilt_deg"],
            }
        )
    else:
        result["axis_source"] = "fit"
    if t_bc is None:
        return None
    result.update(camera._arm_motion(t_bc, result["delta_cam"]))
    if teach.tip_pose is None:
        return None
    return core.transport_pose(t_bc, result["delta_cam"], teach.tip_pose)


@router.post("/options")
async def options(body: runtime.OptionsBody) -> dict:
    """Run-time options, each changed only when given: ``flat`` opts into the resting prior (see
    :func:`_compose_motion`), off by default; ``trust_share`` is the share of the place object's tracked points that
    must be seen for its track to be followed (see :func:`_apply_others`); ``depth_check`` counts a tracked point as seen only while its depth agrees within ``depth_tol_mm`` with where it was
    when a view last placed its object, on by default (see :func:`_seen_with_depth`); ``groups_with_acts`` runs acts
    with the point groups, which carry an object no view places with what it rests on, on by default (see
    :attr:`_State.groups_with_acts`)."""
    with runtime._state.lock:
        if body.flat is not None:
            runtime._state.flat = bool(body.flat)
        if body.trust_share is not None:
            runtime._state.trust_share = float(body.trust_share)
        if body.depth_check is not None or body.depth_tol_mm is not None:
            if body.depth_check is not None:
                runtime._state.depth_check = bool(body.depth_check)
            if body.depth_tol_mm is not None:
                runtime._state.depth_tol_m = float(body.depth_tol_mm) / 1000.0
            # Places kept under another setting do not count: the next view that places each object starts them over.
            runtime._state.track.depths = runtime._PointDepths()
            for target in runtime._state.targets.values():
                target.depths = runtime._PointDepths()
        if body.groups_with_acts is not None:
            runtime._state.groups_with_acts = bool(body.groups_with_acts)
        return {
            "flat": runtime._state.flat,
            "trust_share": runtime._state.trust_share,
            "depth_check": runtime._state.depth_check,
            "depth_tol_mm": runtime._state.depth_tol_m * 1000.0,
            "groups_with_acts": runtime._state.groups_with_acts,
        }


# ── live tracking: the worker follows the card frame after frame; the arm may follow the pose ──


class TrackBody(BaseModel):
    algo: str = "p2p"  # the one tracker; the DINO modes stay as a comparison
    follow: bool = False
    hover_mm: float = 20.0


def _view_points(demo: runtime._Demo, obj: str) -> np.ndarray:
    """The designated object's surface on the frame it was clicked on, camera frame (N, 3): the points a find's motion
    carries. Read once per designation and kept on the demo. Pre: the object is done on a stream demo."""
    o = demo.objects[obj]
    key = (int(o["frame"]), int(np.count_nonzero(o["mask"])))
    kept = demo.view_points.get(obj)
    if kept is None or kept[0] != key:
        kept = demo.view_points[obj] = (key, planning._object_points(demo, obj, np.eye(4)))
    return kept[1]


def _reference_surface(recording_path: str, frame: int, mask: np.ndarray) -> np.ndarray:
    """Dense interior depth in the designation's camera frame, never the tracker map."""
    _rgb, depth = recording._stream_frame(recording_path, frame)
    k = np.loadtxt(pathlib.Path(recording_path) / "cam_K.txt")
    intr = {"fx": k[0, 0], "fy": k[1, 1], "cx": k[0, 2], "cy": k[1, 2]}
    return surface_constraint.depth_points(depth, mask, intr)


def _object_surface(demo: runtime._Demo, obj: str) -> np.ndarray:
    o = demo.objects[obj]
    key = (int(o["frame"]), np.asarray(o["mask"], bool).tobytes())
    kept = demo.surface_points.get(obj)
    if kept is None or kept[0] != key:
        kept = demo.surface_points[obj] = (key, _reference_surface(demo.recording, key[0], o["mask"]))
    return kept[1]


def _teach_surface(teach: runtime._Teach) -> np.ndarray:
    if teach.keypoints.get("mask") is None:
        return np.asarray(teach.keypoints.get("xyz", np.empty((0, 3))))
    return surface_constraint.depth_points(teach.depth_m, teach.keypoints["mask"], teach.intr)


def _surface_decision(
    pose, reference, depth, mask, intr, check=None, recover_axis=False, end_swap=False
) -> dict:
    """A failed measurement is an explicit rejection, including unexpected errors."""
    try:
        return surface_constraint.resolve(
            pose,
            reference,
            depth,
            mask,
            intr,
            noise_m=core.POINT_NOISE_M,
            tol_m=core.ACT_REACH_TOL_M,
            tol_deg=core.ACT_REACH_TOL_DEG,
            surface=check,
            recover_axis=recover_axis,
            end_swap=end_swap,
        )
    except Exception as exc:
        logger.exception("Surface pose constraint failed")
        return {
            "state": "rejected",
            "pose": None,
            "observed_dof": 0,
            "reason": f"surface constraint error: {type(exc).__name__}: {exc}",
        }


def _constrain_find_result(job: runtime._Job, result: dict) -> dict:
    """Check acquisition as well as tracking; return raw and control poses separately."""
    key, reference = None, None
    try:
        if result.get("ref_ok") and result.get("ref_delta") is not None:
            key = "ref_delta"
            reference = _reference_surface(
                job.extra["ref_recording"],
                int(job.extra["ref_frame"]),
                result.get("ref_mask", job.arrays["ref_mask"]),
            )
        elif job.kind == "find" and result.get("ok") and result.get("delta") is not None:
            with runtime._state.lock:
                teach = runtime._state.teach
            if teach is not None:
                key, reference = "delta", _teach_surface(teach)
        if key is None:
            return result
        spec = symmetry.normalise((getattr(job, "extra", None) or {}).get("ref_symmetry"))
        decision = _surface_decision(
            result[key],
            reference,
            job.depth_m,
            result.get("mask"),
            job.intr,
            recover_axis=key == "ref_delta" and spec["kind"] == "continuous" and spec["axis"] == "long_axis",
            end_swap=bool(spec.get("end_swap")),
        )
    except (OSError, KeyError, ValueError, AssertionError) as exc:
        decision = {"state": "rejected", "pose": None, "reason": f"reference surface unavailable: {exc}"}
    if key is None:
        return result
    out = {**result, f"raw_{key}": np.asarray(result[key]).copy(), "surface_constraint": decision}
    if decision["pose"] is None:
        out["ref_ok" if key == "ref_delta" else "ok"] = False
        out["ref_reason" if key == "ref_delta" else "reason"] = decision["reason"]
    else:
        out[key] = np.asarray(decision["pose"], dtype=float)
    return out


def _fit_points_3d(uv: Any, inlier: Any, depth_m: np.ndarray, intr: dict[str, float]) -> np.ndarray | None:
    """A tracker view's fit points in 3D, camera frame: its inlier pixels lifted by the frame's own depth, a pixel
    without depth left out; None when the tracker reported none."""
    if uv is None:
        return None
    px = np.asarray(uv, dtype=float).reshape(-1, 2)
    if inlier is not None and len(inlier) == len(px):
        px = px[np.asarray(inlier).astype(bool)]
    u, v = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
    inside = (u >= 0) & (u < depth_m.shape[1]) & (v >= 0) & (v < depth_m.shape[0])
    u, v = u[inside], v[inside]
    z = np.asarray(depth_m, dtype=float)[v, u]
    u, v, z = u[z > 0], v[z > 0], z[z > 0]
    return np.stack([(u - intr["cx"]) * z / intr["fx"], (v - intr["cy"]) * z / intr["fy"], z], axis=1)


def _view_places(
    r: dict[str, Any], job: runtime._Job, teach: runtime._Teach
) -> tuple[bool, str, tuple[int, int, dict[int, np.ndarray], np.ndarray] | None]:
    """The act's one rule (:func:`core.view_places`) for a view of the object picked: the share of its tracks the
    view sees (with the depth check, those whose depth agrees: :func:`_seen_with_depth`), and its fit points against
    its middle where the pose held puts it (the teach's points, carried). Post: ``(ok, reason, depth)``, ``depth`` the
    check's result for :func:`_depth_accept` once the view is taken, None when it did not run."""
    with runtime._state.lock:
        need, held, tol = runtime._state.trust_share, runtime._state.test, runtime._state.depth_tol_m
        name = str((teach.keypoints.get("ref") or {}).get("object") or teach.keypoints.get("concept"))
        frames = (
            list(runtime._state.groups_feed.frames.get(name, ())) if runtime._state.groups_with_acts else []
        )
    pose = held.result.get("delta_cam") if held is not None else None
    pose = np.asarray(r["delta"] if pose is None else pose, dtype=float)
    taught = np.asarray(teach.keypoints.get("xyz", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    middle = pose[:3, :3] @ taught.mean(axis=0) + pose[:3, 3] if len(taught) else None
    points = _fit_points_3d(r.get("fit_uv"), r.get("fit_inlier"), job.depth_m, job.intr)
    n_tracks = int(r.get("n_tracks") or 0)
    share = int(r.get("n_matches") or 0) / n_tracks if n_tracks else 1.0
    depth = (
        _seen_with_depth(
            runtime._state.track.depths,
            r,
            "track_{}",
            r.get("session"),
            job.depth_m,
            job.intr,
            frames,
            job.created,
        )
        if n_tracks
        else None
    )
    counted = share if depth is None else depth[0] / max(1, depth[1])
    ok, why = core.view_places(counted, points, middle, need)
    if depth is not None and not ok and share >= need and counted < need:
        why = (
            f"only {counted:.0%} of its points are seen with a depth that agrees within {tol * 1000:.0f} mm; "
            f"a view counts from {need:.0%}"
        )
    return ok, why, depth


def _verify_track_poses(
    job: runtime._Job, r: dict, teach: runtime._Teach, accepted: bool, log: bool = True
) -> list[dict]:
    """Compare and constrain off the event loop, before any control pose is accepted."""
    if r.get("algo") != "p2p":
        return []
    try:
        return _compare_track_poses(job, r, teach, accepted, log)
    except Exception as exc:
        logger.exception("Pose verification unavailable at %.6f", job.created)
        missing = {"state": "unavailable", "reason": f"diagnostic error: {type(exc).__name__}: {exc}"}
        return [
            {
                "object": "pose verification",
                "t_frame": job.created,
                "disagreement": False,
                "groups": missing,
                "surface": missing,
                "corrected_groups": missing,
                "constraint": {"state": "rejected", "pose": None, "reason": missing["reason"]},
            }
        ]


def _compare_track_poses(
    job: runtime._Job, r: dict, teach: runtime._Teach, accepted: bool, log: bool = True
) -> list[dict]:
    """Compare raw P2P and prepare a separate control proposal on the render executor.

    Compare the frame's RAW P2P pose, not a stale accepted pose. Keep its exact
    frame time, each source's state and the acceptance decision together in the
    record. Groups' fixed reference does not follow P2P re-pinning.
    """
    with runtime._state.lock:
        demo, tr = runtime._state.demo, runtime._state.track
        groups_on = runtime._state.groups_with_acts
        group_frames = {n: list(fs) for n, fs in runtime._state.groups_feed.frames.items()}
        floors = dict(runtime._state.groups_feed.floor)
        drawing = runtime._state.groups_feed.drawing or {}
        targets = dict(runtime._state.targets)
    name = str(
        (planning._session_object(demo) if demo else None) or teach.keypoints.get("concept") or "object"
    )
    sources = [
        (
            name,
            r.get("delta") if r.get("ok") else None,
            teach.keypoints.get("xyz"),
            _teach_surface(teach),
            r.get("mask"),
            accepted,
            (r.get("session"), teach.stamp, teach.at),
        )
    ]
    for i, share in enumerate(r.get("others") or ()):
        other = share.get("name")
        target = targets.get(other)
        if other is None or target is None or target.anchor is None:
            continue
        pose = r.get(f"other_delta_{i}") if share.get("ok") and not share.get("lost") else None
        pose = None if pose is None else np.asarray(pose) @ target.anchor
        reference = geometry = None
        if demo is not None and other in demo.objects:
            with contextlib.suppress(OSError, KeyError, ValueError):
                reference = _view_points(demo, other)
                geometry = _object_surface(demo, other)
        sources.append(
            (
                other,
                pose,
                reference,
                geometry,
                r.get(f"other_mask_{i}"),
                target.last.get("state") == "tracking",
                (share.get("session"), target.anchor.tobytes()),
            )
        )
    checks = []
    for name, pose, reference, geometry, mask, used, session in sources:
        started = time.perf_counter()
        ref = np.asarray(reference if reference is not None else np.empty((0, 3)), dtype=float).reshape(-1, 3)
        group, timing = None, {}
        fs = group_frames.get(name, []) if groups_on else []
        context = (drawing.get("objects") or {}).get(name) or {}
        group_info = (drawing.get("groups") or {}).get(str(context.get("group"))) or {}
        if fs:
            nearest = min(fs, key=lambda f: abs(f[0] - job.created))[0]
            timing = {"nearest_frame_t": nearest, "time_skew_s": abs(nearest - job.created)}
            # Use interpolation only when bracketed, or a sample within the
            # existing frame/pose synchronization allowance. Never compare stale
            # groups to current P2P as if both were observed now.
            if timing["time_skew_s"] <= runtime.HISTORY_MATCH_S and group_info.get("supported", True):
                group = core._groups_pose_at(fs, job.created)
        tracker = tr.verification.setdefault(name, pose_verification.GroupReference())
        compared = tracker.compare(
            pose,
            group if len(ref) else None,
            ref.mean(axis=0) if len(ref) else np.zeros(3),
            job.created,
            (session, floors.get(name)),
            tol_m=core.ACT_REACH_TOL_M,
            tol_deg=core.ACT_REACH_TOL_DEG,
        )
        if not len(ref):
            compared = {"state": "unavailable", "reason": "reference centre missing"}
        check = {
            "object": name,
            "t_frame": job.created,
            "accepted": bool(used),
            "p2p_pose": None if pose is None else np.asarray(pose).tolist(),
            "groups": {
                **compared,
                **timing,
                "context": {k: context.get(k) for k in ("group", "own", "why", "n_seen", "n_grouped")},
            },
            "surface": pose_verification.surface(
                pose,
                ref,
                job.depth_m,
                mask,
                job.intr,
                noise_m=core.POINT_NOISE_M,
                tol_m=core.ACT_REACH_TOL_M,
                tol_deg=core.ACT_REACH_TOL_DEG,
            ),
        }
        check["disagreement"] = any(check[k]["state"] == "disagreement" for k in ("groups", "surface"))
        surface_decision = _surface_decision(pose, geometry, job.depth_m, mask, job.intr, check["surface"])
        corrected_tracker = tr.corrected_verification.setdefault(
            name, pose_verification.GroupMotionReference()
        )
        corrected_group = corrected_tracker.compare(
            surface_decision.get("pose"),
            group if len(ref) else None,
            ref.mean(axis=0) if len(ref) else np.zeros(3),
            job.created,
            (session, floors.get(name)),
            tol_m=core.ACT_REACH_TOL_M,
            tol_deg=core.ACT_REACH_TOL_DEG,
        )
        check["corrected_groups"] = corrected_group
        check["disagreement"] |= corrected_group["state"] == "disagreement"
        check["constraint"] = pose_verification.with_group_consistency(
            surface_decision,
            check["groups"],
            ref.mean(axis=0) if len(ref) else np.zeros(3),
            tol_m=core.ACT_REACH_TOL_M,
            tol_deg=core.ACT_REACH_TOL_DEG,
            held=context.get("why") == "held",
            corrected_group_check=corrected_group,
        )
        check["elapsed_ms"] = (time.perf_counter() - started) * 1000
        checks.append(check)
    if log:
        _log_pose_conflicts(checks, tr)
    return checks


def _log_pose_conflicts(checks: list[dict], tr: runtime._Track) -> None:
    conflicts = {
        (c["object"], key)
        for c in checks
        for key in ("groups", "surface", "corrected_groups")
        if c[key]["state"] == "disagreement"
    }
    for check in checks:
        for key in ("groups", "surface", "corrected_groups"):
            if (check["object"], key) in conflicts - tr.verification_conflicts:
                logger.warning(
                    "Pose disagreement at %.6f: %s %s vs %s; accepted=%s; control=%s; %s",
                    check["t_frame"],
                    check["object"],
                    "surface-constrained pose" if key == "corrected_groups" else "P2P",
                    "groups" if key == "corrected_groups" else key,
                    check["accepted"],
                    check.get("constraint", {}).get("state", "diagnostic only"),
                    check[key],
                )
    tr.verification_conflicts = conflicts


async def _apply_track_result(job: runtime._Job) -> None:
    """One tracked frame: a certified, trusted fit becomes the live pose (and the jog's target when
    following); an occluded or lost frame leaves the last pose in place and only changes the status."""
    from .. import jog

    r = job.result or {}
    tr = runtime._state.track
    epoch = job.extra.get("track_epoch")
    with runtime._state.lock:
        stale = tr.job != job.id if epoch is None else tr.epoch != epoch or not tr.on
        teach = runtime._state.teach
        flat = bool(job.extra.get("geometry", runtime._state.flat))
    if stale or teach is None:
        return
    checks = await asyncio.get_running_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, _verify_track_poses, job, r, teach, False, False
    )
    job.verified_at = time.time()
    # Teach/track can be replaced while the executor is measuring this frame.
    with runtime._state.lock:
        stale = tr.job != job.id if epoch is None else tr.epoch != epoch or not tr.on
        if runtime._state.track is not tr or stale or runtime._state.teach is not teach:
            return
    missing = {"state": "rejected", "pose": None, "reason": "surface decision unavailable"}
    decisions = (
        {c["object"]: c.get("constraint", missing) for c in checks} if r.get("algo") == "p2p" else None
    )
    primary = checks[0].get("constraint", missing) if checks else missing
    proposal = r
    if decisions is not None and primary.get("pose") is not None:
        proposal = {**r, "delta": np.asarray(primary["pose"], dtype=float), "surface_constraint": primary}
    status: dict[str, Any] = {
        k: r.get(k)
        for k in (
            "software",
            "ok",
            "state",
            "algo",
            "ms",
            "n_matches",
            "n_inliers",
            "rms_m",
            "scale",
            "reason",
            "card_points",
            "card_grew",
            "n_tracks",  # the session's tracks of the object, which only grow, and the model points it adopted
            "n_model_points",
            "worker_stages_ms",
            "geometry_stages_ms",
        )
    }
    result: dict[str, Any] | None = None
    transported = None
    if r.get("ok"):
        trusted, why = core.find_trusted(int(r.get("n_inliers", 0)), int(teach.keypoints["n_points"]))
        if trusted:
            trusted, why, depth = _view_places(proposal, job, teach)
            if depth is not None:
                status["depth_seen"] = round(depth[0] / max(1, depth[1]), 3)
            if trusted and decisions is not None and primary.get("pose") is None:
                trusted, why = False, primary["reason"]
            if trusted:
                _depth_accept(tr.depths, depth, job.created)
        if trusted:
            result = {
                k: v
                for k, v in r.items()
                # The fitted points are the recording's, not the live readout's: arrays the state cannot serve. The
                # session's other objects are applied on their own (:func:`_apply_others`).
                if k
                not in (
                    "mask",
                    "uv",
                    "xyz",
                    "model_xyz",
                    "model_valid",
                    "delta",
                    "face_teach",
                    "face_find",
                    "fit_uv",
                    "fit_inlier",
                    "track_idx",
                    "track_uv",
                    "track_vis",
                    "others",
                )
                and not k.startswith("other_")
            }
            result["mode"] = "features"
            if r.get("mask") is not None:
                result["live_mask"] = np.asarray(r["mask"]).astype(bool)
            try:
                t_bc = camera._t_base_cam()
            except HTTPException:
                t_bc = None
            if decisions is not None:
                result["surface_constraint"] = primary
            transported = _compose_motion(result, proposal, teach, flat, t_bc)
            status.update(
                {
                    k: result.get(k)
                    for k in (
                        "axis_source",
                        "yaw_deg",
                        "face_tilt_deg",
                        "arm_turn_deg",
                        "arm_lean_deg",
                    )
                }
            )
            status["motion"] = core.motion_summary(result["delta_cam"])
            d = result["delta_cam"]
            c = np.asarray(teach.keypoints["xyz"]).mean(axis=0)
            status["centre_shift_mm"] = float(np.linalg.norm(d[:3, :3] @ c + d[:3, 3] - c) * 1000.0)
            if transported is not None:
                status["transported_tip_mm"] = (transported[:3, 3] * 1000.0).tolist()
                if tr.follow:
                    pose = transported.copy()
                    pose[2, 3] += tr.hover_mm / 1000.0
                    try:
                        jog.set_target_pose(pose)
                    except RuntimeError as e:
                        status["follow_error"] = str(e)
            with runtime._state.lock:
                runtime._state.test = runtime._Test(
                    at=time.strftime("%H:%M:%S"), rgb=job.rgb, result=result, transported=transported
                )
            name = (teach.keypoints.get("ref") or {}).get("object")
            if name and result.get("live_mask") is not None:
                acquisition._remember_seen(name, result["live_mask"], job.rgb.shape, "tracked by Point2Pose")
        else:
            status.update(ok=False, state="untrusted", reason=why)
    _apply_others(
        r, job.rgb.shape, job.depth_m, job.intr, job.created, decisions
    )  # the same step's other objects: the one placed onto
    now = time.perf_counter()
    if tr.t_prev:
        tr.fps = 0.8 * tr.fps + 0.2 / max(now - tr.t_prev, 1e-3)
    tr.t_prev = now
    status["fps"] = tr.fps
    for i, check in enumerate(checks):
        target = runtime._state.targets.get(check["object"])
        check["accepted"] = (
            result is not None if i == 0 else target is not None and target.last.get("state") == "tracking"
        )
        check["raw_p2p_used"] = bool(
            check["accepted"] and check.get("constraint", {}).get("state") == "unchanged"
        )
        if check["accepted"] and (check.get("groups") or {}).get("context", {}).get("why") != "held":
            tracker = tr.corrected_verification.get(check["object"])
            if tracker is not None:
                tracker.accept()
    _log_pose_conflicts(checks, tr)
    status["pose_verification"] = checks
    overlay = await asyncio.get_event_loop().run_in_executor(
        runtime._RENDER_EXECUTOR, rendering._render_live, job.rgb, r, result, transported, teach, status
    )
    job.rendered_at = time.time()
    if (
        job.result_received_at is not None
        and job.result_decoded_at is not None
        and job.verified_at is not None
    ):
        status["timing_ms"] = {
            "worker_round_trip": round(1000 * (job.result_received_at - job.created), 1),
            "decode": round(1000 * (job.result_decoded_at - job.result_received_at), 1),
            "verify": round(1000 * (job.verified_at - job.result_decoded_at), 1),
            "render": round(1000 * (job.rendered_at - job.verified_at), 1),
            **job.worker_timing_ms,
        }
    with runtime._state.lock:
        stale = tr.job != job.id if epoch is None else tr.epoch != epoch or not tr.on
        if runtime._state.track is not tr or stale or runtime._state.teach is not teach:
            return
        held = runtime._state.test.result.get("delta_cam") if runtime._state.test is not None else None
        tr.model = {
            "t": job.created,
            "model_xyz": r.get("model_xyz"),
            "model_valid": r.get("model_valid"),
            "raw_delta_cam": r.get("delta"),  # the worker's fit, even when the view rejects it
            "delta_cam": None if result is None else result["delta_cam"],  # this frame's, when trusted
            "held_delta_cam": held,  # the last trusted one, before any carrying by point groups
            "teach_xyz": teach.keypoints.get("xyz"),
            "mask": r.get("mask"),
            "rgb": job.rgb,
            "depth_m": job.depth_m,
            "intr": job.intr,
        }
        tr.last, tr.overlay = status, overlay
        if tr.job == job.id:
            tr.job = None
        tr.history.append((job.created, result is not None, None if result is None else result["delta_cam"]))
        if result is not None:
            tr.hold_views.append((job.created, np.asarray(result["delta_cam"], dtype=float).copy()))
            del tr.hold_views[: -runtime.TRACK_HISTORY_MAX]
        del tr.history[: -runtime.TRACK_HISTORY_MAX]
        rec = runtime._state.recording
        if rec is not None and runtime._state.stream is None:
            rec["frames"].append(
                (time.time(), job.rgb[::2, ::2].copy())
            )  # half size: a demo's video is a record, not evidence
        run, step = runtime._state.run, runtime._state.act.step
    if run is not None:
        run.frame(job, r, status, step, None if result is None else result["delta_cam"], transported)
    if epoch is None and tr.done is not None:
        tr.done.set()


async def _track_pump() -> None:
    """Feed fresh frames; overlap one worker inference with one ordered server result, never build a backlog."""
    tr = runtime._state.track
    slots = tr.slots
    assert slots is not None
    session_epoch = tr.epoch
    last_camera_stamp = None
    try:
        while True:
            with runtime._state.lock:
                on, teach, running, flat = (
                    tr.on and tr.epoch == session_epoch,
                    runtime._state.teach,
                    runtime._state.worker.running,
                    runtime._state.flat,
                )
            if not on:
                return
            if teach is None or teach.keypoints.get("mode") != "features":
                tr.last = {"state": "stopped", "reason": "teach by concept first"}
                return
            if not running:
                tr.last = {"state": "stopped", "reason": "the worker is not running"}
                return
            await slots.acquire()
            with runtime._state.lock:
                if not tr.on or tr.epoch != session_epoch:
                    slots.release()
                    return
            try:
                timed = await camera.timed_frame(last_camera_stamp)
                if timed is None:
                    rgb, depth_m, intr = await camera._frame()
                    camera_stamp = None
                else:
                    rgb, depth_m, intr, camera_stamp = timed
                    last_camera_stamp = camera_stamp
            except (HTTPException, TimeoutError) as e:
                slots.release()
                # A frame that cannot be read now is waited out: a track that stopped here stayed stopped once the
                # camera was back, while the live view went on showing its last frame as tracking.
                tr.last = {"state": "waiting", "reason": e.detail if isinstance(e, HTTPException) else str(e)}
                await asyncio.sleep(runtime.TRACK_FRAME_RETRY_S)
                continue
            with runtime._state.lock:
                if not tr.on or tr.epoch != session_epoch:
                    slots.release()
                    return
                epoch = tr.epoch
            assert tr.done is not None
            tr.done.clear()
            job = worker._queue_job(
                "track",
                teach.keypoints["concept"],
                rgb,
                depth_m,
                intr,
                algo=tr.algo,
                compress=False,
                camera_stamp_mono=camera_stamp,
                # P2P supplies the pose. Its surface authority is checked by _verify_track_poses on the
                # server; the worker's full footprint/face diagnostic is only needed for flat control.
                extra={"geometry": flat, "pipeline": True, "track_epoch": epoch},
            )
            with runtime._state.lock:
                tr.job = job.id
            try:
                await asyncio.wait_for(tr.done.wait(), timeout=runtime.TRACK_JOB_TIMEOUT_S)
            except TimeoutError:
                tr.last = {"state": "waiting", "reason": "the worker did not answer"}
                # Keep this one request in flight. Retrying would queue more frames behind a hung worker;
                # a late answer still resumes the stream, while stop cancels the wait.
                await tr.done.wait()
    finally:
        with runtime._state.lock:
            if tr.epoch == session_epoch:
                tr.on, tr.job, tr.task = False, None, None


@router.post("/track/start")
async def track_start(body: TrackBody) -> dict:
    """Start following the taught object live with ``algo``; ``follow`` makes the jog walk to the hover above it."""
    from .. import showservo

    if body.algo not in core.TRACK_ALGOS:
        raise HTTPException(422, f"algo must be one of {core.TRACK_ALGOS}")
    with runtime._state.lock:
        teach = runtime._state.teach
        running = runtime._state.worker.running
    if teach is None or teach.keypoints.get("mode") != "features":
        raise HTTPException(409, "teach by concept first")
    if not running:
        raise HTTPException(409, "start the worker first")
    if showservo.live_camera() is None:
        raise HTTPException(409, "start a live camera session first")
    tr = runtime._state.track
    with runtime._state.lock:
        tr.algo, tr.follow, tr.hover_mm = body.algo, body.follow, body.hover_mm
    _begin_track()
    return {"status": "tracking", "algo": tr.algo, "follow": tr.follow}


def _begin_track() -> bool:
    """Start the live track with the current settings if it is not running. Post: True when it is
    running after the call. Called on the event loop (an endpoint, or the worker-result handler)."""
    tr = runtime._state.track
    with runtime._state.lock:
        already = tr.on
        if not already:
            tr.on, tr.last, tr.fps, tr.t_prev, tr.overlay = True, {"state": "starting"}, 0.0, 0.0, None
            tr.done = asyncio.Event()
            tr.slots = asyncio.Semaphore(2)
            tr.epoch += 1
    if not already:
        tr.task = asyncio.create_task(_track_pump())
    return True


@router.post("/track/stop")
async def track_stop() -> dict:
    tr = runtime._state.track
    with runtime._state.lock:
        tr.on = False
        tr.epoch += 1
        done, task = tr.done, tr.task
    if done is not None:
        done.set()
    if task is not None:
        task.cancel()
    return {"status": "stopped"}


@router.post("/track/options")
async def track_options(body: TrackBody) -> dict:
    """Switch the algorithm, the following and the hover while tracking runs; the next frame uses them."""
    if body.algo not in core.TRACK_ALGOS:
        raise HTTPException(422, f"algo must be one of {core.TRACK_ALGOS}")
    tr = runtime._state.track
    with runtime._state.lock:
        tr.algo, tr.follow, tr.hover_mm = body.algo, body.follow, body.hover_mm
    return {"algo": tr.algo, "follow": tr.follow, "hover_mm": tr.hover_mm}


@router.get("/track/model.npz")
async def track_model() -> Response:
    """The live session's model of the object it follows, as of its newest frame: every point it has adopted (the
    session's init frame), the raw fit, this frame's motion if trusted and the last trusted one, the teach's points,
    the live mask, and the frame (colour, depth in metres, intrinsics fx, fy, cx, cy), to check what the model took in."""
    with runtime._state.lock:
        m = runtime._state.track.model
    if m is None:
        raise HTTPException(404, "no tracking frame yet")
    intr = m["intr"] or {}
    arrays = {k: np.asarray(v) for k, v in m.items() if v is not None and k != "intr"}
    arrays["intr"] = np.array([intr.get(k, np.nan) for k in ("fx", "fy", "cx", "cy")], dtype=float)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return Response(content=buf.getvalue(), media_type="application/octet-stream")


@router.get("/track/live.jpg")
async def track_live() -> Response:
    with runtime._state.lock:
        overlay = runtime._state.track.overlay
    if overlay is None:
        raise HTTPException(404, "no tracking frame yet")
    return Response(content=overlay, media_type="image/jpeg")
