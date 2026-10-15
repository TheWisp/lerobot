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

"""Object acquisition and its evidence, reference registration and reacquisition."""

from __future__ import annotations

import asyncio
import contextlib
import json
import pathlib
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from lerobot.showservo import motion as core, symmetry

from . import camera, groups, planning, rendering, runtime, tracking, worker

router = APIRouter()


@router.post("/teach/capture")
async def teach_capture(body: runtime.TeachBody) -> dict:
    """Teach the object: by concept through the worker (SAM3 + DINO), or by a drawn box. Before jogging in."""
    if body.mode == "features":
        click = [int(v) for v in body.click] if body.click else None
        if click is not None and len(click) != 2:
            raise HTTPException(422, "click is x, y")
        concept = body.concept.strip() or ("clicked object" if click else "")
        if not concept:
            raise HTTPException(422, "a concept is required, e.g. 'yellow block', or click the object")
        job_id = await _start_teach(click, concept, body.ref_object, log={"why": "clicked on the page"})
        return {"pending": True, "job": job_id, "mode": "features"}
    if len(body.box) != 4:
        raise HTTPException(422, "box is x0, y0, x1, y1")
    x0, y0, x1, y1 = body.box
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    rgb, depth_m, intr = await camera._frame()
    # Texture first; a plain object falls back to its shape above the table.
    kp, texture_reason = None, ""
    try:
        kp = core.keypoints_in_box(rgb, depth_m, intr, box)
        if int(kp["valid"].sum()) < core.MIN_TEXTURE_POINTS:
            texture_reason = f"only {int(kp['valid'].sum())} textured points"
            kp = None
    except ValueError as e:
        texture_reason = str(e)
    if kp is None:
        try:
            kp = core.shape_teach(depth_m, intr, box, rgb)
        except ValueError as e:
            raise HTTPException(422, f"{texture_reason}; and {e}") from e
    else:
        kp["mode"] = "texture"
        # Keep the shape model too: weak texture that re-matches badly falls back to it at Find.
        try:
            kp["shape"] = core.shape_teach(depth_m, intr, box, rgb)
        except ValueError:
            kp["shape"] = None
    with runtime._state.lock:
        runtime._state.teach = runtime._Teach(
            at=time.strftime("%H:%M:%S"), box=box, rgb=rgb, depth_m=depth_m, intr=intr, keypoints=kp
        )
        runtime._state.test = None
    return _teach_info(kp)


async def _start_teach(
    click: list[int] | None,
    concept: str,
    ref_object: str,
    more: dict[str, np.ndarray] | None = None,
    log: dict[str, Any] | None = None,
) -> str:
    """Queue a features teach of the object at ``click`` on the live frame; with ``ref_object``, also its find
    against the demo's view of that object. The live Point2Pose session starts over with it, keeping the demo's other
    objects it follows, and with ``more`` (name -> mask, found just before on the live view) joining from the start:
    one restart for both. ``log`` goes to the find log with the teach's result (:func:`_log_find`). Post: the job's id,
    the teach pending and the live pose cleared. Raises HTTPException when
    the worker is not running or ``ref_object`` is not tracked in the demo."""
    if not runtime._state.worker.running:
        raise HTTPException(409, "start the worker first")
    rgb, depth_m, intr = await camera._frame()
    with runtime._state.lock:
        mode = runtime._state.track.algo  # the Point2Pose mode the teach anchors, when one is selected
    ref = None
    if ref_object:
        with runtime._state.lock:
            demo = runtime._state.demo
        ref = None if demo is None else demo.objects.get(ref_object)
        if ref is None or ref.get("status") != "done" or demo.recording is None:
            raise HTTPException(409, f"{ref_object!r} is not a tracked object of the current demo")
        concept = ref_object
    job = worker._queue_job("teach", concept, rgb, depth_m, intr, algo=mode, click=click)
    job.log = log
    if ref is not None:
        job.extra = {
            "ref_recording": demo.recording,
            "ref_frame": int(ref["frame"]),
            "ref_object": ref_object,
            "ref_symmetry": symmetry.normalise(ref.get("symmetry")),
            "scene": _session_names(demo, but=ref_object),
            "more": list(more or ()),
        }
        job.arrays = {
            "ref_mask": np.asarray(ref["mask"], dtype=bool),
            **{f"more_mask_{i}": np.asarray(m, dtype=bool) for i, m in enumerate((more or {}).values())},
        }
    with runtime._state.lock:
        runtime._state.teach_job = job.id
        runtime._state.test = None
    return job.id


def _deepest_pixel(mask: np.ndarray | None, shape: tuple[int, ...]) -> list[int] | None:
    """``[x, y]`` in an image of ``shape`` at the point deepest inside ``mask``, where a click lands on the
    object rather than on its edge; None for an empty mask."""
    import cv2

    if mask is None or not np.any(mask):
        return None
    m = np.asarray(mask, dtype=np.uint8)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return [int(x * shape[1] / m.shape[1]), int(y * shape[0] / m.shape[0])]


def _reacquisition_clicks(mask: np.ndarray | None, shape: tuple[int, ...]) -> list[list[int]]:
    """Up to three separated interior prompts, starting with the deepest point.

    A point prompt can segment an object's button instead of the whole object.
    Multiplying edge clearance by distance from earlier prompts explores other
    parts of the SAME last observed outline. Three bounds worker cost; every
    proposal still needs the normal reference registration and strength checks.
    """
    import cv2

    if mask is None or not np.any(mask):
        return []
    m = np.asarray(mask, dtype=np.uint8)
    depth = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    yy, xx = np.indices(m.shape)
    separation = np.full(m.shape, np.inf)
    score = depth.copy()
    clicks = []
    for _ in range(3):
        if not np.any(score > 0):
            break
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        click = [int(x * shape[1] / m.shape[1]), int(y * shape[0] / m.shape[0])]
        if click not in clicks:
            clicks.append(click)
        separation = np.minimum(separation, np.hypot(xx - x, yy - y))
        score = depth * separation
    return clicks


async def _find_afresh(
    obj: str, stopped: Callable[[], bool], more: dict[str, np.ndarray] | None = None, need_track: bool = True
) -> str:
    """Find ``obj`` again where it was last seen, as a click there would, and wait for the restarted tracker to
    certify a view. A track kept since an earlier find goes on adding points and never drops one, and its pose
    drifts with them; a find starts it over from the demo's view of the object.

    Pre: ``stopped()`` says whether the act was stopped. Post: "" when found and tracked since, the teach and
    the live pose fresh; otherwise why not, with nothing moved.
    """
    with runtime._state.lock:
        test, teach = runtime._state.test, runtime._state.teach
    if teach is None:
        return f"click {obj} in the camera view to find it"
    seen = test.result.get("live_mask") if test is not None and test.result.get("ok") else None
    outline = seen if seen is not None else teach.keypoints.get("mask")
    clicks = _reacquisition_clicks(outline, teach.rgb.shape)
    if not clicks:
        return f"click {obj} in the camera view to find it"
    came_from = {
        "outline": "the live view" if seen is not None else "the last find",
        **(_outline_stats(outline) or {}),
    }
    t0 = time.monotonic()
    failures = []
    for attempt, click in enumerate(clicks, start=1):
        if stopped():
            return "stopped"
        try:
            job_id = await _start_teach(
                click,
                obj,
                obj,
                more,
                log={"why": "act: found afresh", "came_from": came_from, "attempt": attempt},
            )
        except HTTPException as e:
            return str(e.detail)
        while True:
            with runtime._state.lock:
                pending, teach = runtime._state.teach_job == job_id, runtime._state.teach
            if not pending:
                break
            if stopped():
                return "stopped"
            if time.monotonic() - t0 > runtime.ACT_STEP_TIMEOUT_S:
                return f"finding {obj} took over {runtime.ACT_STEP_TIMEOUT_S:.0f} s"
            await asyncio.sleep(runtime.ACT_TICK_S)
        ref = (teach.keypoints.get("ref") if teach is not None else None) or {}
        if ref.get("ok"):
            break
        failures.append(
            ref.get("reason") or f"{obj} is not where it was last seen: click it in the camera view"
        )
        if time.monotonic() - t0 > runtime.ACT_STEP_TIMEOUT_S:
            return f"finding {obj} took over {runtime.ACT_STEP_TIMEOUT_S:.0f} s"
    else:
        return "; ".join(dict.fromkeys(failures))
    if problem := planning._recovered_pose_problem(obj, ref):
        return problem
    if core.find_strength(ref.get("inliers"), ref.get("card_points"))[0] is False:
        return (
            f"a weak find: {obj} matched {ref['inliers']} of the demo view's {ref['card_points']} points; "
            "turn it closer to how it lay in the demo, then press Act"
        )
    if not need_track:  # the find's own view is the object's pose until the restarted track sees it
        return ""
    found, t0 = time.time(), time.monotonic()
    while not planning._certified_since(found):
        if stopped():
            return "stopped"
        if time.monotonic() - t0 > runtime.ACT_STEP_TIMEOUT_S:
            return f"the tracker has not seen {obj} since finding it"
        await asyncio.sleep(runtime.ACT_TICK_S)
    return ""


async def _locate(
    obj: str,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    intr: dict[str, float],
    click: list[int],
    stopped: Callable[[], bool],
    track: bool = False,
    log: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find the demo's object ``obj`` under ``click`` on this frame against the demo's view of it, leaving the live
    track alone: the object a place goes onto, and the held object in the gripper. A find the worker was asked for is
    kept in the find log (:func:`_log_find`) with ``log``, why it was asked and where ``click`` came from.

    With ``track``, the object joins the live Point2Pose session from this frame (:func:`_store_located` follows it).
    ``reference`` overrides only the recorded view used for this measurement; its transform is never installed
    in the live tracker. It may request fresh segmentation of that saved view, returned as ``reference_mask``.
    Post: ``object``, ``ok``, ``delta`` (the motion from the demo's view, camera frame) or None, ``inliers``,
    ``card_points``, ``turn_deg``, ``reason``, ``mask`` (what SAM3 cut out at the click, or None), ``view`` (the demo
    and the frame of the view it was matched against), ``at``, ``stamp`` (when the frame ``delta`` is from was read:
    the caller reads it just before), ``answered`` (the worker answered) and, with ``track``, ``n_points`` and
    ``tracking``, and ``find_id`` (its row in the find log, or None); ``ok`` False with the reason when the object is
    not tracked in the demo, the worker is off or slow, or ``stopped()``.
    """
    with runtime._state.lock:
        demo = runtime._state.demo
    o = None if demo is None else demo.objects.get(obj)
    now = time.time()
    out: dict[str, Any] = {"object": obj, "ok": False, "delta": None, "mask": None, "at": now, "stamp": now}
    if o is None or o.get("status") != "done" or demo.recording is None:
        return {**out, "reason": f"{obj!r} is not a tracked object of the current demo"}
    if reference is not None and track:
        raise ValueError("a held-view reference cannot replace the live track's designation")
    ref = o if reference is None else reference
    out["view"] = [demo.name, int(ref["frame"])]  # the demo view its motion is from
    if not runtime._state.worker.running:
        return {**out, "reason": "start the worker first"}
    job = worker._queue_job("locate", obj, rgb, depth_m, intr, click=[int(click[0]), int(click[1])])
    job.extra = {
        "ref_recording": demo.recording,
        "ref_frame": int(ref["frame"]),
        "ref_object": obj,
        "ref_symmetry": symmetry.normalise(o.get("symmetry")),
    }
    if ref.get("refresh_mask"):
        job.extra["ref_refresh_mask"] = True
    if track:  # into the live session, beside the objects it already follows
        job.extra.update(track=True, scene=_session_names(demo, but=obj))
    job.arrays = {"ref_mask": np.asarray(ref["mask"], dtype=bool)}
    t0 = time.monotonic()
    while job.result is None:
        if stopped():
            return {**out, "reason": "stopped"}
        if time.monotonic() - t0 > runtime.ACT_STEP_TIMEOUT_S:
            reason = f"finding {obj} took over {runtime.ACT_STEP_TIMEOUT_S:.0f} s"
            fid = _log_find("locate", obj, click, rgb, depth_m, None, {"ok": False, "reason": reason}, log)
            return {**out, "reason": reason, "find_id": fid}
        await asyncio.sleep(runtime.ACT_TICK_S)
    r = job.result
    found = bool(r.get("ok") and r.get("ref_ok") and r.get("ref_delta") is not None)
    out = {
        **out,
        "ok": found,
        "delta": np.asarray(r["ref_delta"], dtype=float) if found else None,
        "inliers": r.get("ref_inliers"),
        "card_points": r.get("ref_card_points"),
        "turn_deg": r.get("ref_turn_deg"),
        "reason": "" if found else (r.get("ref_reason") or r.get("reason") or f"{obj} was not found"),
        "mask": None if r.get("mask") is None else np.asarray(r["mask"]).astype(bool),
        "answered": True,  # the worker's own answer, found or not: a measurement
        "n_points": r.get("n_points"),
        "tracking": bool(r.get("tracking")),
        "surface_constraint": r.get("surface_constraint"),
        "raw_delta": r.get("raw_ref_delta"),
        "reference_mask": r.get("ref_mask"),
    }
    result = {
        **_find_info(out),
        "tracking": out["tracking"],
        "view": out["view"],
        "surface_constraint": out["surface_constraint"],
        "raw_delta": None if out["raw_delta"] is None else np.asarray(out["raw_delta"]).tolist(),
        "delta": None if out["delta"] is None else out["delta"].tolist(),
    }
    out["find_id"] = _log_find("locate", obj, click, rgb, depth_m, out["mask"], result, log)
    return out


def _outline_stats(mask: np.ndarray | None) -> dict[str, Any] | None:
    """An outline as the logs keep it: its pixels, its box ``[x0, y0, x1, y1]`` and its middle, the click a find there
    would use; None for no outline."""
    if mask is None or not np.any(mask):
        return None
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.nonzero(m)
    return {
        "px": int(len(xs)),
        "box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "middle": _deepest_pixel(m, m.shape),
    }


def _write_find(
    d: pathlib.Path,
    row: dict[str, Any],
    rgb: np.ndarray | None,
    depth_m: np.ndarray | None,
    cut: np.ndarray | None,
) -> None:
    import cv2

    with contextlib.suppress(OSError):
        d.mkdir(parents=True, exist_ok=True)
        fid = row["id"]
        if rgb is not None:
            cv2.imwrite(str(d / f"{fid}.jpg"), cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR))
        if depth_m is not None:
            depth_mm = np.round(np.nan_to_num(np.asarray(depth_m, dtype=np.float64)) * 1000.0)
            cv2.imwrite(str(d / f"{fid}_depth.png"), np.clip(depth_mm, 0, 65535).astype(np.uint16))
        if cut is not None:
            cv2.imwrite(str(d / f"{fid}_cut.png"), np.asarray(cut, dtype=np.uint8) * 255)
        with (d / runtime.FINDS_LOG).open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")


def _log_find(
    kind: str,
    obj: str,
    click: list[int] | None,
    rgb: np.ndarray | None,
    depth_m: np.ndarray | None,
    cut: np.ndarray | None,
    result: dict[str, Any],
    log: dict[str, Any] | None,
) -> str | None:
    """Keep one find of the saved demo's object ``obj`` beside the demo (:data:`FINDS_DIR`): why it was asked, the click
    and where that click came from, the outline the find cut out there and what it matched, with the frame it ran on
    unless that is one of the demo's own (``log["came_from"]["demo_frame"]``). An act's record lists the finds made
    while it ran. Written off the event loop; a demo not saved keeps none. Post: the find's id in the log, or None.
    Pre: ``_state.lock`` not held."""
    with runtime._state.lock:
        demo, act = runtime._state.demo, runtime._state.act
        acting = act.on
    if demo is None or demo.root is None:
        return None
    log = log or {}
    came_from = log.get("came_from") or {}
    now = time.time()
    fid = f"{time.strftime('%Y%m%d_%H%M%S', time.localtime(now))}_{int(now % 1 * 1000):03d}_{obj}_{kind}"
    row = {
        "id": fid,
        "at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
        "object": obj,
        "kind": kind,
        "why": log.get("why", ""),
        "click": None if click is None else [int(click[0]), int(click[1])],
        "came_from": came_from,
        "cut": _outline_stats(cut),
        "during_act": acting,
        **result,
    }
    own_frame = "demo_frame" not in came_from
    if acting:
        act.finds.append(fid)
    runtime._FIND_LOG_EXECUTOR.submit(
        _write_find,
        pathlib.Path(demo.root) / runtime.FINDS_DIR,
        row,
        rgb if own_frame else None,
        depth_m if own_frame else None,
        cut,
    )
    return fid


def _write_seen(path: pathlib.Path, obj: str, click: list[int], trail: dict[str, Any] | None = None) -> None:
    """Merge ``obj``'s last-seen click into ``path``; a write that fails leaves the old one. With ``trail``, the write is
    also appended to the demo's last-seen log (:data:`SEEN_LOG`)."""
    try:
        seen = json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        seen = {}
    seen[obj] = {"click": click, "at": time.strftime("%Y-%m-%d %H:%M:%S")}
    with contextlib.suppress(OSError):
        path.write_text(json.dumps(seen))
    if trail is not None:
        with contextlib.suppress(OSError):
            (path.parent / runtime.FINDS_DIR).mkdir(parents=True, exist_ok=True)
            with (path.parent / runtime.FINDS_DIR / runtime.SEEN_LOG).open("a") as f:
                f.write(json.dumps({"object": obj, **seen[obj], **trail}) + "\n")


def _read_seen(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except (OSError, ValueError):
        return {}


def _remember_seen(obj: str, mask: np.ndarray | None, shape: tuple[int, ...], source: str) -> None:
    """Keep where ``obj`` of the current saved demo was last seen live, ``mask`` on a frame of ``shape``, as the click a
    find there would use, so a later load finds it there without one. At most every :data:`LAST_SEEN_EVERY_S` per
    object, written off the event loop; each write is logged with ``source`` (what the outline came from) and the
    outline."""
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None or demo.root is None or mask is None:
        return
    now = time.monotonic()
    if now - runtime._last_seen_written.get(obj, -1e9) < runtime.LAST_SEEN_EVERY_S:
        return
    click = _deepest_pixel(mask, shape)
    if click is None:
        return
    runtime._last_seen_written[obj] = now
    trail = {"source": source, "frame_shape": list(shape[:2]), "outline": _outline_stats(mask)}
    runtime._SEEN_EXECUTOR.submit(
        _write_seen, pathlib.Path(demo.root) / runtime.LAST_SEEN_FILE, obj, click, trail
    )


def _start_refind(demo: runtime._Demo) -> None:
    """Run :func:`_refind_last_seen` for a demo just loaded, kept until it ends."""
    task = asyncio.create_task(_refind_last_seen(demo))
    runtime._refinds.add(task)
    task.add_done_callback(runtime._refinds.discard)


async def _refind_last_seen(demo: runtime._Demo) -> None:
    """After ``demo`` is loaded: once the worker and the camera are up, find its objects again where they were last
    seen, as the operator's clicks there would: the others first (:func:`_other_objects`), then the session's own
    (:func:`_session_object`) by a teach whose Point2Pose session starts with all of them, one start for all. One not
    found there waits for a click, as before."""
    from .. import showservo

    if demo.root is None:
        return
    seen = await asyncio.get_running_loop().run_in_executor(
        runtime._SEEN_EXECUTOR, _read_seen, pathlib.Path(demo.root) / runtime.LAST_SEEN_FILE
    )
    picked, others = planning._session_object(demo), planning._other_objects(demo)
    if not ((picked in seen) or any(o in seen for o in others)):
        return
    t0 = time.monotonic()
    while True:  # the worker and the camera are often started after the load, and the load's own teach first
        with runtime._state.lock:
            current, running, busy = (
                runtime._state.demo is demo,
                runtime._state.worker.running,
                runtime._state.teach_job is not None,
            )
            ready = any("worker ready" in line for line in runtime._state.worker.log)
        if not current or time.monotonic() - t0 > runtime.REFIND_WAIT_S:
            return
        if running and ready and not busy and showservo.live_camera() is not None:
            break
        await asyncio.sleep(0.5)

    def gone() -> bool:
        return runtime._state.demo is not demo

    with runtime._state.lock:
        teach, teach_job = runtime._state.teach, runtime._state.teach_job
    found_picked = teach is not None and (teach.keypoints.get("ref") or {}).get("object") == picked
    teach_picked = picked in seen and not found_picked and teach_job is None
    founds: dict[str, dict[str, Any]] = {}
    for onto in others:  # first, so the session's own object starts it with them
        if onto not in seen or _located(demo, onto) is not None:
            continue
        rgb, depth_m, intr = await camera._frame()
        found = await _locate(
            onto,
            rgb,
            depth_m,
            intr,
            list(seen[onto]["click"]),
            gone,
            track=not teach_picked,
            log={"why": "demo loaded: found where it was last seen", "came_from": {"last_seen": seen[onto]}},
        )
        if gone():
            return
        _store_located(onto, found)
        founds[onto] = found
    if teach_picked:
        more = {n: f["mask"] for n, f in founds.items() if f["ok"] and f.get("mask") is not None} or None
        try:
            log = {
                "why": "demo loaded: found where it was last seen",
                "came_from": {"last_seen": seen[picked]},
            }
            job_id = await _start_teach(list(seen[picked]["click"]), picked, picked, more, log=log)
        except HTTPException:
            return
        t0 = time.monotonic()
        while (
            runtime._state.teach_job == job_id
            and not gone()
            and time.monotonic() - t0 < runtime.ACT_STEP_TIMEOUT_S
        ):
            await asyncio.sleep(runtime.ACT_TICK_S)
        with runtime._state.lock:
            teach = runtime._state.teach
        if more and teach is not None and (teach.keypoints.get("ref") or {}).get("ok") and not gone():
            for n in more:
                _store_located(n, {**founds[n], "tracking": True})


def _session_names(demo: runtime._Demo, but: str | None = None) -> list[str]:
    """The demo's objects the live session keeps following, but ``but``: every object a key frame is held in."""
    return [n for n in planning._act_objects(demo) if n != but]


def _store_located(obj: str, found: dict[str, Any]) -> None:
    """Keep ``found`` as the last find of ``obj``; when it joined the live session, follow it from now on
    (:func:`_apply_others`), and stop following an earlier find of it. Called on the event loop."""
    if not found.get("ok"):
        # A failed reference match can still have a SAM mask (including the
        # whole tray when a stale click lands on background). It is evidence
        # in the find log, never a live object outline or a future click seed.
        found = {**found, "mask": None, "tracking": False}
    if found.get("ok") and found.get("mask") is not None:
        _remember_seen(obj, found["mask"], found["mask"].shape, f"find {found.get('find_id')}")
        groups._groups_follow({obj: found["mask"]})
    with runtime._state.lock:
        runtime._state.located[obj] = found
        if found.get("ok") and found.get("tracking"):
            runtime._state.targets[obj] = runtime._TargetTrack(
                obj=obj, anchor=np.asarray(found["delta"], dtype=float)
            )
        else:
            runtime._state.targets.pop(obj, None)


def _located(demo: runtime._Demo, obj: str) -> dict[str, Any] | None:
    """The last locate of ``obj`` against this demo's view of it, or None. One made against another demo, or against
    an earlier designation of the object, gives a motion from a different view."""
    o = demo.objects.get(obj)
    with runtime._state.lock:
        found = runtime._state.located.get(obj)
    if found is None or o is None or found.get("view") != [demo.name, int(o["frame"])]:
        return None
    return found


def _located_info(found: dict[str, Any]) -> dict[str, Any]:
    """A locate as the page shows it: like a find, without its arrays; ``track`` is its track's newest state."""
    with runtime._state.lock:
        target = runtime._state.targets.get(found.get("object"))
    return {
        **{k: found.get(k) for k in ("object", "ok", "inliers", "card_points", "turn_deg", "reason", "at")},
        "track": target.last.get("state") if target is not None else None,
        **dict(
            zip(
                ("strong", "share"),
                core.find_strength(found.get("inliers"), found.get("card_points")),
                strict=True,
            )
        ),
    }


def _weak(found: dict[str, Any]) -> str:
    """Why a find of the demo's view is too weak to act on, or "" when it is not weak."""
    if core.find_strength(found.get("inliers"), found.get("card_points"))[0] is not False:
        return ""
    return (
        f"a weak find: {found['object']} matched {found['inliers']} of the demo view's {found['card_points']} points; "
        "turn it closer to how it lay in the demo"
    )


class LocateBody(BaseModel):
    click: list[int]
    object: str


@router.post("/locate")
async def locate(body: LocateBody) -> dict:
    """Find a designated object of the demo where the operator clicked it, against the demo's view of it, without
    starting a track: the object a place goes onto. Each act finds it again where this left it."""
    if len(body.click) != 2:
        raise HTTPException(422, "click is x, y")
    with runtime._state.lock:
        demo = runtime._state.demo
    if demo is None or demo.objects.get(body.object, {}).get("status") != "done":
        raise HTTPException(409, f"{body.object!r} is not a tracked object of the current demo")
    if not runtime._state.worker.running:
        raise HTTPException(409, "start the worker first")
    rgb, depth_m, intr = await camera._frame()
    found = await _locate(
        body.object,
        rgb,
        depth_m,
        intr,
        body.click,
        lambda: False,
        track=True,
        log={"why": "clicked on the page"},
    )
    _store_located(body.object, found)
    return _located_info(found)


async def _locate_afresh(obj: str, stopped: Callable[[], bool], track: bool = True) -> str:
    """Find the place's object again where it was last found (where its track has it), as a click there would; with
    ``track``, it joins the live session from there. Post: "" with ``_state.located[obj]`` fresh and strong;
    otherwise why not."""
    with runtime._state.lock:
        last = runtime._state.located.get(obj)
    if last is not None and not last.get("ok"):
        return f"click {obj} in the camera view; its previous find did not match the demo"
    mask = None if last is None else last.get("mask")
    click = None if mask is None else _deepest_pixel(mask, mask.shape)
    if click is None:
        return f"click {obj} in the camera view to find it"
    rgb, depth_m, intr = await camera._frame()
    came_from = {
        "kept_outline": _outline_stats(mask),
        "kept_ok": last.get("ok"),
        "kept_find": last.get("find_id"),
        "found_at": last.get("at"),
        "tracked_at": last.get("tracked_at"),  # when the tracker last replaced the outline, if it has
    }
    log = {"why": "act: found again where it was kept", "came_from": came_from}
    found = await _locate(obj, rgb, depth_m, intr, click, stopped, track=track, log=log)
    _store_located(obj, found)
    if not found["ok"]:
        return found["reason"] or f"{obj} is not where it was last seen: click it in the camera view"
    if problem := planning._recovered_pose_problem(obj, found):
        return problem
    return _weak(found) + (", then press Act" if _weak(found) else "")


def _find_info(ref: dict[str, Any]) -> dict[str, Any]:
    """A find as the page shows it: what it matched, of how many of the demo view's points, and whether that is strong."""
    strong, share = core.find_strength(ref.get("inliers"), ref.get("card_points"))
    return {
        **{k: ref.get(k) for k in ("object", "ok", "inliers", "turn_deg", "reason", "card_points")},
        "strong": strong,
        "share": share,
    }


def _teach_info(kp: dict[str, Any]) -> dict[str, Any]:
    if kp["mode"] == "features":
        return {
            "mode": "features",
            "concept": kp["concept"],
            "n_points": int(kp["n_points"]),
            "radius_mm": float(kp["radius_mm"]),
            "shape_class": kp["shape_class"],
            "yaw_observable": bool(kp["yaw_observable"]),
            "face_planarity": None if not kp.get("face") else kp["face"]["planarity"],
            "face_usable": core.face_usable(kp.get("face")),
            "ref": None if not kp.get("ref") else _find_info(kp["ref"]),
        }
    if kp["mode"] == "texture":
        return {
            "mode": "texture",
            "n_keypoints": int(len(kp["uv"])),
            "n_with_depth": int(kp["valid"].sum()),
            "shape_fallback": kp.get("shape") is not None,
        }
    return {
        "mode": "shape",
        "n_points": int(kp["n_points"]),
        "height_mm": float(kp["height_m"] * 1000.0),
        "colour_cue": kp.get("colour") is not None,
    }


@router.get("/teach.jpg")
async def teach_jpeg() -> Response:
    import cv2

    with runtime._state.lock:
        teach = runtime._state.teach
    if teach is None:
        raise HTTPException(404, "nothing taught")
    bgr = cv2.cvtColor(teach.rgb, cv2.COLOR_RGB2BGR)
    x0, y0, x1, y1 = teach.box
    cv2.rectangle(bgr, (x0, y0), (x1, y1), (0, 220, 255), 2)
    if teach.keypoints["mode"] == "texture":
        for (u, v), ok in zip(teach.keypoints["uv"], teach.keypoints["valid"], strict=True):
            cv2.circle(bgr, (int(u), int(v)), 3, (60, 200, 60) if ok else (0, 0, 255), 1)
    elif teach.keypoints["mode"] == "features":
        rendering._outline(bgr, teach.keypoints["mask"], (255, 0, 255))
        for u, v in teach.keypoints["uv"][::3]:
            cv2.circle(bgr, (int(u), int(v)), 2, (0, 220, 255), -1)
    else:
        rendering._outline(bgr, teach.keypoints["mask"], (60, 200, 60))
    if teach.tip_pose is not None:
        with contextlib.suppress(HTTPException):  # no arm connected: the image still shows the object
            rendering._draw_tool(bgr, camera._t_base_cam(), teach.intr, teach.tip_pose, "pre-grasp")
    return Response(content=camera._jpeg(bgr), media_type="image/jpeg")


@router.post("/test/capture")
async def test_capture() -> dict:
    """Find the taught object in a fresh frame and transport the pre-grasp by its motion."""
    with runtime._state.lock:
        teach = runtime._state.teach
    if teach is None:
        raise HTTPException(409, "teach first")
    if teach.tip_pose is None:
        raise HTTPException(409, "mark the pre-grasp first")
    t_bc = camera._t_base_cam()
    rgb, depth_m, intr = await camera._frame()
    check = await camera._camera_check_async(rgb)
    if teach.keypoints["mode"] == "features":
        if not runtime._state.worker.running:
            raise HTTPException(409, "start the worker first")
        job = worker._queue_job("find", teach.keypoints["concept"], rgb, depth_m, intr)
        job.camera_check = check
        with runtime._state.lock:
            runtime._state.find_job = job.id
        return {"pending": True, "job": job.id, "mode": "features", "camera_check": check}
    if teach.keypoints["mode"] == "texture":
        result = core.register(teach.keypoints, rgb, depth_m, intr)
        if result.get("ok") and runtime._state.flat:
            snap = core.snap_to_table_yaw(
                result["delta_cam"],
                camera._table_normal_cam(t_bc),
                np.asarray(teach.keypoints["xyz"])[teach.keypoints["valid"]].mean(axis=0),
            )
            result["delta_cam"] = snap["delta"]
            result.update(
                {"yaw_deg": snap["yaw_deg"], "tilt_discarded_deg": snap["tilt_deg"], "snapped": True}
            )
        if not result.get("ok") and teach.keypoints.get("shape") is not None:
            texture_reason = result.get("reason", "texture failed")
            result = core.shape_register(teach.keypoints["shape"], depth_m, intr, rgb)
            result["fallback_from"] = f"texture ({texture_reason})"
    else:
        result = core.shape_register(teach.keypoints, depth_m, intr, rgb)
    result["camera_check"] = check
    transported = None
    if result.get("ok"):
        transported = core.transport_pose(t_bc, result["delta_cam"], teach.tip_pose)
        result.update(camera._arm_motion(t_bc, result["delta_cam"]))
    with runtime._state.lock:
        runtime._state.test = runtime._Test(
            at=time.strftime("%H:%M:%S"), rgb=rgb, result=result, transported=transported
        )
    info = {k: v for k, v in result.items() if k not in runtime._HEAVY}
    if result.get("ok"):
        info["motion"] = core.motion_summary(result["delta_cam"])
        info["transported_tip_mm"] = (transported[:3, 3] * 1000.0).tolist()
    return info


def _log_teach(job: runtime._Job, r: dict[str, Any], used: bool) -> str | None:
    """A teach in the find log (:func:`_log_find`): what it cut out at its click and, for one of the demo's objects,
    its find against the demo's view of it; ``used`` False for a teach replaced before it came back. Post: its id."""
    ref = job.extra.get("ref_object")
    result: dict[str, Any] = {"taught": bool(r.get("ok")), "used": used, "n_points": r.get("n_points")}
    if ref:
        find = {
            "object": ref,
            "ok": bool(r.get("ok") and r.get("ref_ok")),
            "inliers": r.get("ref_inliers"),
            "turn_deg": r.get("ref_turn_deg"),
            "reason": r.get("ref_reason") or r.get("reason") or "",
            "card_points": r.get("ref_card_points"),
        }
        result.update(_find_info(find))
        result["surface_constraint"] = r.get("surface_constraint")
        for key in ("raw_ref_delta", "ref_delta"):
            if r.get(key) is not None:
                result[key] = np.asarray(r[key]).tolist()
    else:
        result.update(ok=bool(r.get("ok")), reason=r.get("reason") or "")
    mask = None if r.get("mask") is None else np.asarray(r["mask"]).astype(bool)
    return _log_find("teach", ref or job.concept, job.click, job.rgb, job.depth_m, mask, result, job.log)


def _apply_teach_result(job: runtime._Job) -> None:
    r = job.result or {}
    with runtime._state.lock:
        used = runtime._state.teach_job == job.id
    find_id = _log_teach(job, r, used)
    with runtime._state.lock:
        if runtime._state.teach_job != job.id:
            return
        runtime._state.teach_job = None
        if not r.get("ok"):
            runtime._state.teach = None
            runtime._state.worker.log.append(f"teach failed: {r.get('reason')}")
            return
        kp = {
            "mode": "features",
            "concept": job.concept,
            "mask": r["mask"].astype(bool),
            "uv": r["uv"],
            "xyz": r["xyz"],
            "n_points": int(r["n_points"]),
            "radius_mm": float(r["radius_mm"]),
            "shape_class": r["shape_class"],
            "yaw_observable": bool(r["yaw_observable"]),
            "face": r.get("face"),
        }
        if job.extra.get("ref_object"):
            kp["ref"] = {
                "object": job.extra["ref_object"],
                "ok": bool(r.get("ref_ok")),
                "delta": None if r.get("ref_delta") is None else np.asarray(r["ref_delta"], dtype=float),
                "inliers": r.get("ref_inliers"),
                "turn_deg": r.get("ref_turn_deg"),
                "reason": r.get("ref_reason", ""),
                "card_points": r.get("ref_card_points"),
                "surface_constraint": r.get("surface_constraint"),
            }
        runtime._state.teach = runtime._Teach(
            at=time.strftime("%H:%M:%S"),
            box=(0, 0, 0, 0),
            rgb=job.rgb,
            depth_m=job.depth_m,
            intr=job.intr,
            keypoints=kp,
            stamp=job.created,
        )
        if runtime._state.demo is not None and len(runtime._state.demo.tips):
            # The demo's first pose is the pose that gets transported and drawn; a demo may be applied
            # to a newly taught object on purpose.
            runtime._state.teach.tip_pose = runtime._state.demo.tips[0].copy()
            runtime._state.teach.gripper = float(runtime._state.demo.grippers[0])
        runtime._state.test = None
        running = runtime._state.worker.running
    if job.extra.get("ref_object") and r.get("ref_ok"):  # one of the demo's objects, which the act follows
        _remember_seen(
            job.extra["ref_object"], np.asarray(r["mask"]).astype(bool), job.rgb.shape, f"find {find_id}"
        )
        groups._groups_follow({job.extra["ref_object"]: np.asarray(r["mask"]).astype(bool)})
    # A taught object is tracked from that moment: the guided flow has no separate "start tracking".
    from .. import showservo

    if running and showservo.live_camera() is not None:
        tracking._begin_track()


def _apply_find_result(job: runtime._Job) -> None:
    r = job.result or {}
    with runtime._state.lock:
        if runtime._state.find_job != job.id:
            return
        runtime._state.find_job = None
        teach = runtime._state.teach
    if teach is None or teach.tip_pose is None:
        return
    result: dict[str, Any] = {
        k: v
        for k, v in r.items()
        if k not in ("mask", "uv", "xyz", "delta", "raw_delta", "raw_ref_delta", "face_teach", "face_find")
    }
    result["mode"] = "features"
    result["camera_check"] = job.camera_check
    if r.get("mask") is not None:
        result["live_mask"] = r["mask"].astype(bool)
    transported = None
    if r.get("ok"):
        trusted, why = core.find_trusted(int(r.get("n_inliers", 0)), int(teach.keypoints["n_points"]))
        if not trusted:
            result["ok"], result["reason"] = False, why
    if result.get("ok"):
        try:
            t_bc = camera._t_base_cam()
        except HTTPException as e:
            result["ok"], result["reason"] = False, e.detail
        else:
            with runtime._state.lock:
                flat = runtime._state.flat
            transported = tracking._compose_motion(result, r, teach, flat, t_bc)
    with runtime._state.lock:
        runtime._state.test = runtime._Test(
            at=time.strftime("%H:%M:%S"), rgb=job.rgb, result=result, transported=transported
        )
