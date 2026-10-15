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

"""Environment group observations and conditional pose propagation."""

from __future__ import annotations

import asyncio
import collections
import json
import subprocess
import threading
import time
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from lerobot.showservo import motion as core

from . import acquisition, camera, runtime

router = APIRouter()


@router.get("/groups/view")
async def groups_view() -> Response:
    """The point groups' live view alone on a page, full width; the Approach tab's Groups panel shows the same
    stream with Start and Finish. A relay until the groups run in the worker and draw on the tab's own camera
    view."""
    return Response(content=runtime.GROUPS_PAGE, media_type="text/html")


async def _relay_stream(addr: tuple[str, int]):
    """The view's MJPEG stream with the view's own response headers stripped, over asyncio streams: nothing here
    blocks the loop. While the view is not there (not started yet, restarting) the relay waits and connects again,
    so the page's stream need not be reloaded."""
    while True:
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(*addr), timeout=2.0)
        except (TimeoutError, OSError):
            await asyncio.sleep(1.0)
            continue
        try:
            writer.write(b"GET /stream HTTP/1.0\r\nHost: 127.0.0.1\r\n\r\n")
            await writer.drain()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b""):
                    break
            while True:
                chunk = await reader.read(1 << 16)
                if not chunk:
                    break
                yield chunk
        finally:
            writer.close()
        await asyncio.sleep(0.5)


@router.get("/groups/stream")
async def groups_stream() -> StreamingResponse:
    return StreamingResponse(
        _relay_stream(runtime.GROUPS_VIEW), media_type="multipart/x-mixed-replace; boundary=frame"
    )


async def _groups_call(method: str, path: str, body: dict[str, Any] | None = None) -> Any:
    """One request to the view over asyncio streams: its JSON answer, or None while it cannot be reached (not started,
    loading the tracker, gone) or does not answer in time."""
    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection(*runtime.GROUPS_VIEW), timeout=2.0)
    except (TimeoutError, OSError):
        return None
    data = b"" if body is None else json.dumps(body).encode()
    head = f"{method} {path} HTTP/1.0\r\nHost: 127.0.0.1\r\n"
    if body is not None:
        head += f"Content-Type: application/json\r\nContent-Length: {len(data)}\r\n"
    try:
        writer.write(head.encode() + b"\r\n" + data)
        await writer.drain()
        raw = await asyncio.wait_for(reader.read(), timeout=5.0)
    except (TimeoutError, OSError):
        return None
    finally:
        writer.close()
    try:
        return json.loads(raw.partition(b"\r\n\r\n")[2])
    except ValueError:
        return None


async def _groups_recording() -> dict[str, Any] | None:
    """What the view says of its recording ({"recording", "frames", "last"}), or None while it cannot be reached:
    still loading the tracker, or gone."""
    return await _groups_call("GET", "/record/status")


@router.get("/groups/status")
async def groups_status() -> dict:
    """The view's process and its recording: {"running", "ready", "recording", "frames", "last", "log",
    "with_acts"}. ``ready`` is false while the view is up but not yet answering (the tracker loading); ``last`` is
    where the finished recording went; ``log`` the view's last lines, which say why it stopped when it stopped by
    itself; ``with_acts`` whether acts run with it (:attr:`_State.groups_with_acts`)."""
    g = runtime._state.groups
    rec = await _groups_recording()
    with runtime._state.lock:
        out: dict[str, Any] = {"running": g.running or rec is not None, "ready": rec is not None, **g.last}
        out["log"] = g.log[-5:]
        out["with_acts"] = runtime._state.groups_with_acts
    out.update(rec or {})
    out.setdefault("recording", None)
    out.setdefault("frames", 0)
    out.setdefault("last", None)
    return out


@router.post("/groups/start")
async def groups_start(request: Request) -> dict:
    """Start the view: the camera's frames, TAPIR on the GPU, the groups drawn as a stream, and from the first
    frame a recording (colour, depth, the groups' state) that Finish closes."""
    from .. import showservo

    if showservo.live_camera() is None:
        raise HTTPException(409, "start the camera first")
    with runtime._state.lock:
        runtime._state.server_url = str(request.base_url).rstrip("/")
    _start_groups(runtime._state.server_url, record=True)
    return {"status": "started"}


def _start_groups(server: str, record: bool = False) -> None:
    """Start the view on the camera of the GUI server at ``server``, keeping colour, depth and its drawing on disk
    with ``record`` (the Groups panel's Start, for the offline replay; about 4 GB an hour, so not for a view the act's
    flow starts). Raises HTTPException when it runs already or cannot start."""
    g = runtime._state.groups
    with runtime._state.lock:
        if g.running:
            raise HTTPException(409, "the groups view is already running")
        cmd = [
            runtime.P2P_PYTHON,
            str(runtime._GROUPS_SCRIPT),
            "--live",
            *(["--record"] if record else []),
            "--server",
            server,
            "--port",
            str(runtime.GROUPS_VIEW[1]),
            "--pips",
            "2",
        ]
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=runtime.P2P_REPO
            )
        except OSError as e:
            raise HTTPException(409, f"the view cannot start: {e}") from e
        g.proc, g.log, g.last = proc, [], {}

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            with runtime._state.lock:
                g.log.append(line.rstrip("\n"))
                del g.log[:-200]

    threading.Thread(target=pump, daemon=True, name="pregrasp-groups").start()


async def _designate_in_groups(
    objects: dict[str, list[int]], server: str | None, stopped: Callable[[], bool]
) -> tuple[str, float]:
    """The view running with ``objects`` (name -> a pixel on it) designated in it, and the feed taking its frames.
    ``(why not, since)``: "" and a time after which every frame the view draws has them, or why not. A view not
    running is started on the camera of the GUI server at ``server``; without one it must be running already. Each
    object's frames from before are dropped: a new designation is a new frame of the object."""
    g = runtime._state.groups
    with runtime._state.lock:
        running = g.running
    if not running and await _groups_call("GET", "/objects") is None:
        if server is None:
            return "start the point groups view first", 0.0
        try:
            _start_groups(server)
        except HTTPException as e:
            return f"the point groups view cannot start: {e.detail}", 0.0
    _ensure_groups_feed()
    with runtime._state.lock:  # one started here is watched for exiting; one started elsewhere only times out
        proc = g.proc if g.running else None
    t0, ticket = time.monotonic(), None
    while True:
        if stopped():
            return "stopped", 0.0
        if ticket is None:
            answer = await _groups_call("POST", "/objects", objects)
            ticket = None if answer is None else answer.get("ticket")
        else:
            answer = await _groups_call("GET", "/objects")
            if answer is not None and int(answer.get("served") or 0) >= ticket:
                # Served: each frame the view reads from now on is drawn after they were designated.
                since = time.time()
                done = answer.get("designated") or {}
                failed = {n: (done.get(n) or {}).get("reason") or "not designated" for n in objects}
                failed = {n: why for n, why in failed.items() if not (done.get(n) or {}).get("ok")}
                if failed:
                    return "the point groups did not take " + "; ".join(
                        f"{n}: {w}" for n, w in failed.items()
                    ), 0.0
                feed = runtime._state.groups_feed
                with runtime._state.lock:
                    for name in objects:
                        feed.frames.pop(name, None)
                        feed.floor[name] = since
                return "", since
        if proc is not None and proc.poll() is not None:
            with runtime._state.lock:
                log = g.log[-3:]
            return "the point groups view stopped: " + " | ".join(log), 0.0
        if time.monotonic() - t0 > runtime.GROUPS_READY_S:
            return (
                f"the point groups view did not take {', '.join(objects)} in {runtime.GROUPS_READY_S:.0f} s",
                0.0,
            )
        await asyncio.sleep(0.25)


def _ensure_groups_feed() -> None:
    """The feed's pump running (:func:`_groups_feed_pump`). Called on the event loop."""
    feed = runtime._state.groups_feed
    with runtime._state.lock:
        if feed.task is not None and not feed.task.done():
            return
        feed.task = asyncio.create_task(_groups_feed_pump())


async def _groups_feed_pump() -> None:
    """While acts use the point groups: every frame the view draws, into the feed (each object's pose, from the frame
    after its designation; a frame counts once), and the newest frame's picture. A view that is not there yet, or
    restarting, is waited for."""
    feed = runtime._state.groups_feed
    try:
        while True:
            with runtime._state.lock:
                if not runtime._state.groups_with_acts:
                    return
                since = feed.since
            answer = await _groups_call("GET", f"/poses?since={since!r}")
            if answer:
                with runtime._state.lock:
                    for stamp, poses in answer.get("frames") or ():
                        stamp = float(stamp)
                        if stamp <= feed.since:
                            continue
                        for name, m in poses.items():
                            if stamp > feed.floor.get(name, 0.0):
                                feed.frames.setdefault(
                                    name, collections.deque(maxlen=runtime.GROUPS_FRAMES_KEPT)
                                ).append((stamp, np.asarray(m, dtype=float).reshape(4, 4)))
                        feed.since, feed.fresh = stamp, time.monotonic()
                    if answer.get("drawing"):
                        feed.drawing = answer["drawing"]
            await asyncio.sleep(runtime.GROUPS_POLL_S)
    finally:
        with runtime._state.lock:
            if feed.task is asyncio.current_task():
                feed.task = None


def _carried(
    name: str,
    since: float,
    symmetry_line: tuple[np.ndarray, np.ndarray] | None = None,
    end_swap: bool = False,
) -> np.ndarray:
    """How the point groups moved ``name`` since the frame read at ``since``, camera frame (:func:`core.carried_motion`);
    no motion without them."""
    with runtime._state.lock:
        frames = (
            list(runtime._state.groups_feed.frames.get(name, ())) if runtime._state.groups_with_acts else []
        )
    return core.carried_motion(frames, since, symmetry_line, end_swap)


def _paint_picture(bgr: np.ndarray, d: dict) -> np.ndarray:
    """A point groups' picture on a camera frame as the act's views paint it, live (:func:`_paint_groups`) and in a
    replay (:func:`_replay_draw`) alike (lerobot.showservo.groups_scene.paint_groups): the surfaces that move
    differently tinted, each group's wireframe by depth with its scale (a picture from before the wireframes: its
    tracks as dots, the still world's faint, those in no group not at all). The act's view draws its own pose of
    each object over this, so not the groups' outlines. Pre: called off the event loop."""
    from lerobot.showservo import groups_scene

    return groups_scene.paint_groups(bgr, d, radius=2, world_radius=1, free=False, outlines=False)


def _paint_groups(bgr: np.ndarray) -> np.ndarray:
    """The point groups' newest picture on a camera frame (:func:`_paint_picture`), while acts use them and it is no
    older than GROUPS_DRAW_MAX_AGE_S. Pre: called off the event loop."""
    with runtime._state.lock:
        on, d = runtime._state.groups_with_acts, runtime._state.groups_feed.drawing
    if not on or d is None or time.time() - float(d.get("stamp") or 0.0) > runtime.GROUPS_DRAW_MAX_AGE_S:
        return bgr
    return _paint_picture(bgr, d)


async def _feed_held(name: str, tip_obj: np.ndarray, t_bc: np.ndarray) -> None:
    """While the gripper holds ``name``: its pose by the arm's joints, the fingertip times how it sat in the fingers
    when they closed (``tip_obj``, or a callback returning the current accepted attachment), sent to the point groups every GROUPS_HELD_S (GroupTracker.hold). The fingers hide
    it from the camera, and its hidden points stay with what it lay on; the arm does not lose it."""
    from .. import jog

    inv_bc = np.linalg.inv(t_bc)
    while True:
        cur = jog.current_tip_and_anchor()
        if cur is not None:
            attachment = tip_obj() if callable(tip_obj) else tip_obj
            pose = inv_bc @ np.asarray(cur[0], dtype=float) @ attachment
            await _groups_call("POST", "/objects/held", {name: np.round(pose, 6).ravel().tolist()})
        await asyncio.sleep(runtime.GROUPS_HELD_S)


def _groups_follow(objects: dict[str, np.ndarray | None]) -> None:
    """Designate in the point groups the objects the act's tracker just found (name -> their mask), so the camera
    view shows them carried from now on; the view is started first when it is not running. Runs beside the caller;
    a failure is kept as the feed's reason. Only while acts use the point groups. Called on the event loop."""
    with runtime._state.lock:  # an act designates its objects itself, before the arm moves
        on, server, acting = runtime._state.groups_with_acts, runtime._state.server_url, runtime._state.act.on
    clicks = {n: acquisition._deepest_pixel(m, np.shape(m)) for n, m in objects.items() if m is not None}
    clicks = {n: c for n, c in clicks.items() if c is not None}
    try:
        asyncio.get_running_loop()
    except RuntimeError:  # a synchronous caller, off the event loop: nothing can run beside it
        return
    if not on or acting or not clicks:
        return

    async def run() -> None:
        why, _since = await _designate_in_groups(clicks, server, lambda: False)
        with runtime._state.lock:
            runtime._state.groups_feed.reason = why

    task = asyncio.create_task(run())
    runtime._groups_follows.add(task)
    task.add_done_callback(runtime._groups_follows.discard)


def _end(proc: subprocess.Popen) -> None:
    """A SIGTERM lets the view close its recording and the camera's; a view that does not go is killed."""
    proc.terminate()
    try:
        proc.wait(10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(5.0)


@router.post("/groups/stop")
async def groups_stop() -> dict:
    """Finish: the view's recording is closed and the view stopped, the camera's recording and the GPU freed with
    it. Answers where the recording went and how many frames it holds."""
    g = runtime._state.groups
    with runtime._state.lock:
        running = g.running
    if not running:
        if await _groups_recording() is not None:
            raise HTTPException(409, "the groups view was not started here; stop it where it was started")
        raise HTTPException(409, "the groups view is not running")
    last = await _finish_groups()
    assert last is not None, "it was running a moment ago"
    return last


async def _finish_groups() -> dict[str, Any] | None:
    """Finish the view started here: its recording closed, the view stopped, the camera's shared frames and the GPU
    freed. What it said of its recording, or None when it was not running."""
    g = runtime._state.groups
    with runtime._state.lock:
        proc = g.proc if g.running else None
    if proc is None:
        return None
    rec = await _groups_recording() or {}
    await asyncio.get_event_loop().run_in_executor(runtime._GROUPS_EXECUTOR, _end, proc)
    with runtime._state.lock:  # a view that was killed rather than stopped leaves the camera's frames shared
        share, runtime._state.camera_share = runtime._state.camera_share, None
    if share is not None:
        await camera._stop_share(share)
    last = {
        "recording": None,
        "frames": rec.get("frames", 0),
        "last": rec.get("recording") or rec.get("last"),
    }
    with runtime._state.lock:
        g.last = last
    return last
