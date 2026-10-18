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

"""Playback and exports of recorded act evidence."""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import json
import pathlib
import subprocess
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from . import camera, groups, planning, recording, rendering, runtime, tracking

router = APIRouter()


def _replay_run(trial: int) -> tuple[pathlib.Path, dict[str, Any]]:
    """Trial row ``trial``'s recorded act: its folder and its record. Raises HTTPException when there is none. Called
    off the event loop."""
    rows = recording._load_trials()
    if not 0 <= trial < len(rows):
        raise HTTPException(404, "no such trial")
    run = rows[trial].get("run")
    f = None if not run else pathlib.Path(run) / "act.json"
    if f is None or not f.exists():
        raise HTTPException(404, "this act was not recorded")
    mtime = f.stat().st_mtime
    kept = runtime._replay_meta.get(str(run))
    if kept is None or kept[0] != mtime:
        kept = runtime._replay_meta[str(run)] = (mtime, json.loads(f.read_text()))
        # Durations are derivable from old motor traces too. Cache the summary
        # in memory; never rewrite the original experimental evidence.
        motor_file = pathlib.Path(run) / "arm.npz"
        if "pauses" not in (kept[1].get("control_cadence") or {}) and motor_file.exists():
            with np.load(motor_file, allow_pickle=False) as motor:
                if all(key in motor for key in ("t", "q_cmd", "q_obs")):
                    kept[1]["control_cadence"] = recording._control_cadence(
                        dict(motor), kept[1].get("targets", [])
                    )
    return pathlib.Path(run), kept[1]


def _replay_timeline(run: pathlib.Path) -> dict:
    path = run / "camera" / "timeline.json"
    if not path.exists():
        raise HTTPException(404, "this act has no fixed-rate camera timeline")
    timeline = json.loads(path.read_text())
    if not timeline.get("rows") or not timeline.get("saved"):
        raise HTTPException(404, "this act has no usable fixed-rate camera timeline")
    return timeline


def _replay_text(
    bgr: np.ndarray, s: str, org: tuple[int, int], colour=(255, 255, 255), scale: float = 0.45
) -> None:
    import cv2

    cv2.putText(bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(bgr, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def _replay_objects(meta: dict[str, Any]) -> list[str]:
    """The act's objects as its record names them: the one whose find started the session, then the others joining
    it (``objects``); for an act recorded before that list was kept, its find's object (or "picked") and the object a
    place went onto (``target``)."""
    if meta.get("objects"):
        return [str(n) for n in meta["objects"]]
    first = (meta.get("find") or {}).get("object") or "picked"
    onto = (meta.get("target") or {}).get("object")
    return [first, *([onto] if onto else [])]


def _replay_draw(
    run: pathlib.Path,
    meta: dict[str, Any],
    i: int,
    points: dict[str, np.ndarray | None],
    *,
    camera_bgr: np.ndarray | None = None,
    at: float | None = None,
    mono_at: float | None = None,
    base: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Frame ``i`` of the act recorded in ``run`` (``meta`` its act.json) with what the act had then, every value as the
    run recorded it, each of the act's objects (:func:`_replay_objects`) in its colour of REPLAY_COLOURS: its mask
    edge and tracked points (filled seen, hollow not), the session object's raw pose estimate applied to its recorded
    model (blue, independently of following acceptance), and for each object joining the session where the act held it
    (drawn like the live view's outline of it, from ``points``, its surface from the demo's view, by name) and what
    carried it in the point groups; the fingertip and, while the act walked the arm, where it was told to go, and what
    each object's view was judged on; underneath, the point groups painted as the live view painted them
    (:func:`_paint_picture`); and, for an act the engine ran, each object where the engine believed it at the newest
    camera frame it had then (its demo surface from ``base``, base frame, by name). What the act did not record is
    said, not rebuilt. Raises HTTPException for a frame the record does not have."""
    import cv2

    frames = meta.get("frames") or []
    if not 0 <= i < len(frames):
        raise HTTPException(404, "no such frame")
    f, d = frames[i], run / "frames"
    bgr = camera_bgr.copy() if camera_bgr is not None else cv2.imread(str(d / f"{i:06d}.jpg"))
    if bgr is None:
        raise HTTPException(404, "the frame's image is missing")
    at = float(f["t_frame"] if at is None else at)
    intr, t0 = meta["intr"], float(meta.get("t_started") or f["t_frame"])
    t_bc = np.asarray(meta["t_bc"], dtype=float) if meta.get("t_bc") else None
    z: Any = np.load(d / f"{i:06d}.npz", allow_pickle=False) if (d / f"{i:06d}.npz").exists() else {}
    names = _replay_objects(meta)
    colour = {n: runtime.REPLAY_COLOURS[k % len(runtime.REPLAY_COLOURS)] for k, n in enumerate(names)}
    first, others = names[0], names[1:]
    # The other_*_k arrays' objects: as the frame names them, or for an act recorded before, the one place object.
    in_session = f.get("others") if f.get("others") is not None else others[:1]
    # Where the act held each other object: held_k by the frame's ``held``, or an earlier act's onto_held.
    held = {h["object"]: (z.get(f"held_{k}"), h) for k, h in enumerate(f.get("held") or [])}
    if "onto_held" in z and others:
        held[others[0]] = (z["onto_held"], f.get("onto_held_from") or {})
    recorded_groups = f.get("groups") or {}
    wired = bool(recorded_groups) and "groups_links" in z
    if recorded_groups and "groups_points" in z:
        # The picture as the live view painted it (_paint_picture): its tint, its wireframe, or for an act recorded
        # before the wireframes its tracks as dots.
        picture = {k: recorded_groups.get(k) for k in ("base", "quiet", "surfaces", "scale", "depth_range")}
        picture["points"] = np.asarray(z["groups_points"])
        if wired:
            picture.update(
                links=np.asarray(z["groups_links"]),
                joins=np.asarray(z["groups_joins"]) if "groups_joins" in z else None,
                depth=np.asarray(z["groups_depth"]),
            )
        bgr = groups._paint_picture(bgr, picture)
    mask = cv2.imread(str(d / f"{i:06d}_mask.png"), cv2.IMREAD_UNCHANGED)
    if mask is not None:
        rendering._outline(bgr, mask > 0, colour[first])
    tracks = [("track_uv", "track_vis", colour[first])]
    for k, name in enumerate(in_session):
        c = colour.get(name, runtime.REPLAY_COLOURS[-1])
        if f"other_mask_{k}" in z:
            rendering._outline(bgr, np.asarray(z[f"other_mask_{k}"], dtype=bool), c)
        tracks.append((f"other_track_uv_{k}", f"other_track_vis_{k}", c))
    for uv_key, vis_key, c in tracks:
        if uv_key in z and vis_key in z:
            for (u, v), seen in zip(np.asarray(z[uv_key]), np.asarray(z[vis_key]).astype(bool), strict=True):
                cv2.circle(bgr, (int(u), int(v)), 3, c, -1 if seen else 1, cv2.LINE_AA)
    estimate = rendering._draw_track_estimate(bgr, intr, f, z, first)
    held_lines = []
    for name in others:
        pose, src = held.get(name, (None, {}))
        if pose is None:
            held_lines.append(
                f"{name} held where: not recorded"
                + ("" if "held" in f else " (an act from before it was kept)")
            )
            continue
        moved = f", the tracker's view of +{src['tracked_at'] - t0:.1f} s" if src.get("tracked_at") else ""
        line = f"{name} held where: {src.get('find_id') or 'its find'}{moved}, carried by the point groups"
        if points.get(name) is not None:
            rendering._draw_found(bgr, intr, points[name], np.asarray(pose, dtype=float), name)
        else:
            line += " (load the act's demo to draw it)"
        held_lines.append(line)
    carried_lines = []
    for name in others:
        if not (recorded_groups and "groups_points" in z):
            carried_lines.append(f"{name} by the point groups: not recorded")
            continue
        o = (recorded_groups.get("objects") or {}).get(name) or {}
        g = o.get("group")
        st = (recorded_groups.get("groups") or {}).get(str(g)) or {}
        own = "yes" if o.get("own") else "no" + (f" ({o['why']})" if o.get("why") else "")
        carried_lines.append(
            f"{name} by the point groups: group {g} ({st.get('members')} tracks, fitted on {st.get('n_fit')}, "
            f"rms {st.get('rms_mm')} mm{'' if st.get('supported', True) else ', unsupported'}); own points placed it: "
            f"{own}, {o.get('n_seen')} of its tracks seen"
        )
    if t_bc is not None and (run / "arm.npz").exists():
        arm = np.load(run / "arm.npz")
        if len(arm["t"]):
            tip = None
            if mono_at is not None and "t_mono" in arm:
                times = arm["t_mono"]
                j = int(np.searchsorted(times, mono_at, side="left"))
                if 0 < j < len(times) and 0 < times[j] - times[j - 1] <= 2.0 / 30.0:
                    from lerobot.showservo import motion as core

                    alpha = float((mono_at - times[j - 1]) / (times[j] - times[j - 1]))
                    tip = core.interp_rigid(arm["tip_obs"][j - 1], arm["tip_obs"][j], alpha)
            else:
                nearest = int(np.argmin(np.abs(arm["t"] - at)))
                if camera_bgr is None or abs(float(arm["t"][nearest]) - at) <= 2.0 / 30.0:
                    tip = arm["tip_obs"][nearest]
            if tip is not None:
                rendering._draw_tool(bgr, t_bc, intr, tip, "")
        told = [x for x in meta.get("targets") or [] if "pose" in x and x["t"] <= at]
        if told and at - told[-1]["t"] < 1.0:
            p = camera._project(t_bc, intr, np.asarray(told[-1]["pose"], dtype=float)[:3, 3])
            if p is not None:
                cv2.circle(bgr, p, 10, runtime.REPLAY_TOLD, 2, cv2.LINE_AA)
    if camera_bgr is not None:
        _draw_timeline_groups(bgr, meta, at, intr)
    engine_line = _draw_engine_estimates(
        bgr, meta, float(f.get("t_capture_mono", 0.0)) if mono_at is None else mono_at, base or {}
    )
    lines = [
        f"+{at - t0:.1f} s  frame {i} of {len(frames)}  {f['step']}",
        f"{first}: {f.get('state')}, following {'accepted' if f.get('used') else 'rejected'}  "
        f"{f.get('n_matches')} of {f.get('n_tracks')} points seen  depth agrees {f.get('depth_seen')}",
    ]
    if camera_bgr is not None:
        lines.insert(
            0,
            f"CAMERA timeline; last P2P source age "
            f"{max(0.0, at - float(f.get('t_capture', f['t_frame']))):.2f} s "
            f"(result available {max(0.0, at - float(f.get('t_result', f['t_frame']))):.2f} s ago)",
        )
    for name in (
        others
    ):  # each object's own track entries; an earlier act's carry no name and are the one place object's
        track = [
            e
            for e in meta.get("target_track") or []
            if e.get("object", others[0]) == name and abs(e["t"] - f["t_frame"]) < 0.5
        ]
        if track:
            e = min(track, key=lambda e: abs(e["t"] - f["t_frame"]))
            lines.append(
                f"{name}: {'trusted' if e['trusted'] else 'not trusted'}{', lost' if e['lost'] else ''}  "
                f"{e['n_visible']} of {e['n_tracks']} points seen  depth agrees {e.get('depth_seen')}"
            )
    stamp = f.get("t_result", f["t_frame"])
    selections = meta.get("hold_selections") or [meta.get("hold_selection") or {}]
    selection = next(
        (
            s
            for s in reversed(selections)
            if s.get("at", float("inf")) <= stamp < s.get("until", float("inf"))
        ),
        {},
    )
    if selection.get("evidence") == "predicted" and f.get("t_result", f["t_frame"]) >= selection["at"]:
        lines.insert(0, f"HOLD: {selection['source']}; assumes no slip; no post-lift visual verification")
    elif selection.get("evidence") == "partly measured":
        lines.insert(0, "HOLD: position measured against held demo views; rotation remains predicted")
    elif selection.get("source") == "held-view registration":
        lines.insert(0, "HOLD: measured against held demo views")
    lines += held_lines + carried_lines + ([engine_line] if engine_line else [])
    for k, s in enumerate(lines):
        _replay_text(bgr, s, (8, 18 + 17 * k))
    legend = [(f"{first}: mask edge, points (filled seen)", colour[first])]
    legend += [(f"{name}: mask edge, points", colour[name]) for name in others]
    legend += [
        (
            f"{first} pose estimate: {estimate}; fit inliers {f.get('fit_inliers')} / {f.get('fit_points')}"
            " (outline includes hidden surface)",
            runtime.POSE_ESTIMATE_COLOUR,
        ),
        ("other objects where the act held them", runtime.FOUND_COLOUR),
        ("fingertip (white cross), told to go (red ring)", (255, 255, 255)),
        (
            "point groups: each a wireframe by depth (bar: near red, far blue); tinted: surfaces moving differently"
            if wired
            else "point groups: tracks by group (the world faint); tinted: surfaces moving differently",
            (200, 200, 200),
        ),
    ]
    if (meta.get("engine") or {}).get("estimates"):
        legend.append(
            (
                "each object where the engine believed it (a held one where the gripper carries it)",
                runtime.REPLAY_ENGINE,
            )
        )
    for k, (s, c) in enumerate(legend):
        _replay_text(bgr, s, (8, bgr.shape[0] - 10 - 15 * (len(legend) - 1 - k)), c, 0.4)
    if any("pose_verification" in frame for frame in frames):
        slots = max(len(names), max(len(frame.get("pose_verification") or []) for frame in frames))
        bgr = rendering._pose_verification_panel(bgr, f.get("pose_verification"), slots)
    return bgr


def _draw_engine_estimates(
    bgr: np.ndarray, meta: dict[str, Any], mono_at: float, base: dict[str, np.ndarray]
) -> str | None:
    """Each object where the act engine believed it at its newest camera frame by ``mono_at`` (the act's
    ``engine.estimates``): its demo surface (``base``, base frame) moved by that pose, as dots. The line saying what
    each was believed by; None for an act without them."""
    import cv2

    estimates = (meta.get("engine") or {}).get("estimates") or []
    k = bisect.bisect_right([e["t"] for e in estimates], mono_at + 1e-6) - 1
    if k < 0 or mono_at - estimates[k]["t"] > runtime.REPLAY_ENGINE_MAX_AGE_S:
        return (
            None
            if not estimates
            else f"engine: none of a frame in the {runtime.REPLAY_ENGINE_MAX_AGE_S:.1f} s before"
        )
    t_bc, intr = np.asarray(meta["t_bc"], dtype=float), meta["intr"]
    cam_from_base = np.linalg.inv(t_bc)
    said = []
    for name, e in estimates[k]["objects"].items():
        said.append(f"{name} by {e['by']}")
        points = base.get(name)
        if points is None:
            continue
        motion = cam_from_base @ np.asarray(e["motion"], dtype=float)
        cam = points @ motion[:3, :3].T + motion[:3, 3]
        cam = cam[cam[:, 2] > 0]
        for x, y, zc in cam[:: max(1, len(cam) // 300)]:
            u, v = int(round(intr["fx"] * x / zc + intr["cx"])), int(round(intr["fy"] * y / zc + intr["cy"]))
            if 0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]:
                cv2.circle(bgr, (u, v), 2, runtime.REPLAY_ENGINE, -1, cv2.LINE_AA)
    drawn = "" if base else " (load the act's demo to draw them)"
    return f"engine, {mono_at - estimates[k]['t']:.2f} s before: " + ", ".join(said) + drawn


def _draw_timeline_groups(bgr: np.ndarray, meta: dict[str, Any], at: float, intr: dict) -> None:
    """Show each group's newest measured centre, independently of the slower P2P overlay."""
    import cv2

    frames = (meta.get("groups") or {}).get("frames") or {}
    for k, name in enumerate(_replay_objects(meta)):
        samples = frames.get(name) or []
        i = bisect.bisect_right([row[0] for row in samples], at) - 1
        if i < 0:
            continue
        row = samples[i]
        age = at - float(row[0])
        if age > runtime.GROUPS_DRAW_MAX_AGE_S:
            continue
        xyz = np.asarray(row[1:], dtype=float).reshape(4, 4)[:3, 3]
        if xyz[2] <= 0:
            continue
        u = int(round(intr["fx"] * xyz[0] / xyz[2] + intr["cx"]))
        v = int(round(intr["fy"] * xyz[1] / xyz[2] + intr["cy"]))
        if not (0 <= u < bgr.shape[1] and 0 <= v < bgr.shape[0]):
            continue
        colour = runtime.REPLAY_COLOURS[k % len(runtime.REPLAY_COLOURS)]
        cv2.drawMarker(bgr, (u, v), colour, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
        _replay_text(
            bgr, f"group {name} ({age:.2f}s)", (min(u + 11, bgr.shape[1] - 160), max(v - 8, 16)), colour, 0.4
        )


def _replay_points(meta: dict[str, Any], demo: runtime._Demo | None) -> dict[str, np.ndarray | None]:
    """Each object joining the act's session, by name: its surface from the demo's view, for drawing where the act
    held it, when the act's demo is the one loaded; otherwise none."""
    out: dict[str, np.ndarray | None] = {}
    if demo is None or demo.name != meta.get("demo"):
        return out
    for name in _replay_objects(meta)[1:]:
        if name in demo.objects:
            with contextlib.suppress(
                OSError, KeyError, ValueError
            ):  # a demo recording that went away: no outline
                out[name] = tracking._view_points(demo, name)
    return out


def _replay_base_points(meta: dict[str, Any], demo: runtime._Demo | None) -> dict[str, np.ndarray]:
    """Each of the act's objects, by name: its surface at its demo pose, base frame (the frame the engine's poses move),
    when the act's demo is the one loaded; otherwise none."""
    out: dict[str, np.ndarray] = {}
    if demo is None or demo.name != meta.get("demo") or not meta.get("t_bc"):
        return out
    for name in _replay_objects(meta):
        if name in demo.objects:
            with contextlib.suppress(OSError, KeyError, ValueError):
                out[name] = np.asarray(
                    planning._object_points(demo, name, np.asarray(meta["t_bc"], dtype=float))
                )
    return out


def _replay_jpeg(trial: int, i: int, demo: runtime._Demo | None, pair: bool = False) -> bytes:
    """Fixed-clock slot ``i`` of an act, alone or beside its camera image, as a JPEG."""
    import cv2

    run, meta = _replay_run(trial)
    points = _replay_points(meta, demo)
    timeline = _replay_timeline(run)
    rows = timeline["rows"]
    if not 0 <= i < len(rows):
        raise HTTPException(404, "no such frame")
    picture = _timeline_pair(run, meta, rows[i], points, _replay_base_points(meta, demo))
    drawn = picture if pair else picture[:, picture.shape[1] // 2 :]
    ok, buf = cv2.imencode(".jpg", drawn)
    assert ok, "a drawn frame encodes"
    return buf.tobytes()


def _timeline_pair(
    run: pathlib.Path,
    meta: dict[str, Any],
    row: dict,
    points: dict[str, np.ndarray | None],
    base: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """One fixed-clock replay slot: its actual camera image and the asynchronous estimates known by then."""
    import cv2

    image = row.get("image")
    raw = None if image is None else cv2.imread(str(run / "camera" / "rgb" / f"{image:06d}.jpg"))
    if raw is None:
        images = sorted((run / "camera" / "rgb").glob("*.jpg"))
        first = cv2.imread(str(images[0])) if images else None
        if first is None:
            raise HTTPException(404, "the Act has no camera image")
        raw = np.zeros_like(first)
    at = float(row["t_wall"])
    frames = meta.get("frames") or []
    i = bisect.bisect_right([float(f.get("t_result", f["t_frame"])) for f in frames], at) - 1
    if i >= 0:
        # Project the arm at the image's host publication time, not the later
        # replay slot. The slot still determines which async results were known.
        image_mono = row.get("camera_mono") or row["t_mono"]
        drawn = _replay_draw(
            run, meta, i, points, camera_bgr=raw, at=at, mono_at=float(image_mono), base=base
        )
    else:
        drawn = raw.copy()
        _draw_timeline_groups(drawn, meta, at, meta["intr"])
        _replay_text(drawn, "P2P result pending", (8, 18))
        if any("pose_verification" in f for f in frames):
            slots = max(
                len(_replay_objects(meta)), max(len(f.get("pose_verification") or []) for f in frames)
            )
            drawn = rendering._pose_verification_panel(drawn, None, slots)
    if row.get("status") not in ("captured", "held_camera_frame"):
        _replay_text(drawn, f"timeline: {row['status']}", (8, 35), (0, 150, 255))
    if row.get("camera_mono") is not None:
        age = max(0.0, float(row["t_mono"]) - float(row["camera_mono"]))
        _replay_text(drawn, f"camera age {age:.2f}s", (8, 52), (200, 200, 200))
    raw = cv2.copyMakeBorder(raw, 0, drawn.shape[0] - raw.shape[0], 0, 0, cv2.BORDER_CONSTANT)
    return np.hstack([raw, drawn])


def _timeline_video_file(
    run: pathlib.Path,
    meta: dict[str, Any],
    points: dict[str, np.ndarray | None],
    timeline: dict,
    base: dict[str, np.ndarray] | None = None,
) -> pathlib.Path:
    """Encode the fixed-rate observation timeline, never substituting P2P's variable result cadence."""
    rows = timeline["rows"]
    fps = float(timeline["fps"])
    picture = _timeline_pair(run, meta, rows[0], points, base)
    h, w = picture.shape[:2]
    out = run / "replay.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert proc.stdin is not None
        for row in rows:
            picture = _timeline_pair(run, meta, row, points, base)
            proc.stdin.write(np.ascontiguousarray(picture).tobytes())
        _out, err = proc.communicate(timeout=runtime.REPLAY_VIDEO_TIMEOUT_S)
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    if proc.returncode != 0:
        raise HTTPException(500, f"the video encoder failed: {err.decode(errors='replace')[-300:]}")
    return out


def _replay_video_file(
    run: pathlib.Path,
    meta: dict[str, Any],
    points: dict[str, np.ndarray | None],
    base: dict[str, np.ndarray] | None = None,
) -> pathlib.Path:
    """Export only the fixed-rate observation timeline as H.264."""
    return _timeline_video_file(run, meta, points, _replay_timeline(run), base)


def _replay_video(trial: int, demo: runtime._Demo | None) -> pathlib.Path:
    """Trial ``trial``'s act as a video (:func:`_replay_video_file`). Called off the event loop."""
    run, meta = _replay_run(trial)
    return _replay_video_file(run, meta, _replay_points(meta, demo), _replay_base_points(meta, demo))


@router.get("/replay")
async def replay(trial: int) -> dict:
    """An act's recording for its replay: each frame's time and step, and whether the act recorded where it held the
    objects joining its session."""
    run, meta = await asyncio.get_running_loop().run_in_executor(runtime._REPLAY_EXECUTOR, _replay_run, trial)
    frames = meta.get("frames") or []
    timeline = await asyncio.get_running_loop().run_in_executor(
        runtime._REPLAY_EXECUTOR, _replay_timeline, run
    )
    rows = timeline["rows"]
    times = [float(row["t_wall"]) for row in rows]
    results = [float(f.get("t_result", f["t_frame"])) for f in frames]
    listed = []
    for i in range(len(rows)):
        j = bisect.bisect_right(results, times[i]) - 1
        listed.append(
            {
                "i": i,
                "t": round(i / float(timeline["fps"]), 3),
                "step": frames[j]["step"] if j >= 0 else "camera",
            }
        )
    disagreements = sorted(
        {
            min(len(rows) - 1, bisect.bisect_left(times, results[k]))
            for k, f in enumerate(frames)
            if any(c.get("disagreement") for c in f.get("pose_verification") or ())
        }
    )
    return {
        "trial": trial,
        "n": len(listed),
        "frames": listed,
        "fps": timeline["fps"],
        "result": meta.get("result"),
        "control_cadence": meta.get("control_cadence"),
        "held_recorded": any(
            "onto_held_from" in f or any(len(h) > 1 for h in f.get("held") or []) for f in frames
        ),
        "pose_verification_recorded": any("pose_verification" in f for f in frames),
        "pose_disagreement_frames": disagreements,
    }


@router.get("/replay/video.mp4")
async def replay_video(trial: int) -> Response:
    """An act's replay as an MP4 to keep: every frame as the camera saw it beside it with what the replay draws on it,
    at the act's own pace."""
    from fastapi.responses import FileResponse

    with runtime._state.lock:
        demo = runtime._state.demo
    path = await asyncio.get_running_loop().run_in_executor(
        runtime._REPLAY_EXPORT_EXECUTOR, _replay_video, trial, demo
    )
    return FileResponse(path, media_type="video/mp4", filename=f"act_{path.parent.name}_replay.mp4")


@router.get("/replay/frame.jpg")
async def replay_frame(trial: int, i: int, pair: bool = False) -> Response:
    with runtime._state.lock:
        demo = runtime._state.demo
    jpg = await asyncio.get_running_loop().run_in_executor(
        runtime._REPLAY_EXECUTOR, _replay_jpeg, trial, i, demo, pair
    )
    return Response(content=jpg, media_type="image/jpeg")
