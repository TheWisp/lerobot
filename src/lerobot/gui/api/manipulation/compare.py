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

"""A demo and one of its acts side by side at the same moment of the demo, each drawn from its perception by the
same code (lerobot.showservo.perception_view): ``/compare?demo=<name>&act=<act folder>``.

A recording not yet processed by the pipeline as it is now is processed on request, one at a time, off the event
loop: minutes on the GPU each. Reading and drawing happen on their own pool too.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import pathlib
import threading
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from lerobot.showservo import motion as core, perception, perception_view
from lerobot.showservo.recordings import Recording, open_recording

from . import camera, demos

router = APIRouter()

_DRAW_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-compare"
)
_PROCESS_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="manipulation-perception"
)
_lock = threading.Lock()
_processing: dict[str, str] = {}  # a recording's folder -> "queued", "running", "done" or why it failed
_loaded: dict[str, tuple[str, Recording, perception.Perception | None, dict]] = {}  # folder -> (version, ...)


def _folders(demo: str, act: str | None) -> tuple[pathlib.Path, pathlib.Path | None]:
    """The demo's folder and its act's, by name: only a demo of the demos folder and one of its own acts."""
    root = demos._demos_root().resolve()
    d = (root / demo).resolve()
    if d.parent != root or not d.is_dir():
        raise HTTPException(404, f"no demo {demo!r}")
    if act is None:
        return d, None
    a = (d / "acts" / act).resolve()
    if a.parent != d / "acts" or not a.is_dir():
        raise HTTPException(404, f"no act {act!r} of {demo!r}")
    return d, a


def _load(folder: pathlib.Path) -> tuple[Recording, perception.Perception | None, dict]:
    """The recording, its perception by the pipeline as it is now (None while not processed) and its objects' surfaces;
    kept until the pipeline changes. Pre: off the event loop."""
    key, version = str(folder), perception.version()
    with _lock:
        hit = _loaded.get(key)
    if hit is not None and hit[0] == version and hit[2] is not None:
        return hit[1], hit[2], hit[3]
    rec = open_recording(folder)
    p = perception.load(rec)
    shapes = perception_view.surfaces(rec, p) if p is not None else {}
    with _lock:
        _loaded[key] = (version, rec, p, shapes)
    return rec, p, shapes


def _state(folder: pathlib.Path) -> dict[str, Any]:
    rec, p, _ = _load(folder)
    with _lock:
        running = _processing.get(str(folder))
    return {
        "processed": p is not None,
        "processing": running if running not in (None, "done") else None,
        "frames": len(rec),
    }


def _info(d: pathlib.Path, a: pathlib.Path | None) -> dict[str, Any]:
    demo, _, _ = _load(d)
    keypoints = demo.annotations.keypoints if demo.annotations else []
    span = float(demo.arm.t[-1] - (demo.origin if demo.origin is not None else demo.arm.t[0]))
    return {
        "version": perception.version(),
        "span": [0.0, round(span, 3)],
        "key_frames": [{"t": float(m["t"]), "label": m["label"]} for m in core.key_frames(keypoints)],
        "demo": _state(d),
        "act": None if a is None else _state(a),
    }


@router.get("/compare/info")
async def compare_info(demo: str, act: str | None = None) -> dict:
    """What the page needs: the demo's span and key frames, and whether each side is processed or processing."""
    d, a = _folders(demo, act)
    return await asyncio.get_running_loop().run_in_executor(_DRAW_EXECUTOR, _info, d, a)


class ProcessBody(BaseModel):
    demo: str
    act: str | None = None


def _process(folder: pathlib.Path) -> None:
    with _lock:
        _processing[str(folder)] = "running"
    try:
        perception.process(open_recording(folder), log=lambda _s: None)
        outcome = "done"
    except Exception as e:  # noqa: BLE001  the page says why; the next request may try again
        outcome = f"failed: {e}"
    with _lock:
        _processing[str(folder)] = outcome
        _loaded.pop(str(folder), None)


@router.post("/compare/process")
async def compare_process(body: ProcessBody) -> dict:
    """Process each side the pipeline as it is now has not, one after the other, beside the server."""
    d, a = _folders(body.demo, body.act)
    queued = []
    for folder in (d, a):
        if folder is None:
            continue
        state = await asyncio.get_running_loop().run_in_executor(_DRAW_EXECUTOR, _state, folder)
        if state["processed"] or state["processing"] is not None:
            continue
        with _lock:
            _processing[str(folder)] = "queued"
        _PROCESS_EXECUTOR.submit(_process, folder)
        queued.append(folder.name)
    return {"queued": queued}


def _frame(d: pathlib.Path, a: pathlib.Path | None, t: float) -> bytes:
    import cv2

    demo, pd, shapes_d = _load(d)
    if pd is None:
        raise HTTPException(409, "the demo is not processed yet")
    t_bc = None
    act = pa = None
    if a is not None:
        act, pa, shapes_a = _load(a)
        t_bc = act.base_from_camera
    if t_bc is None:
        try:
            t_bc = camera._t_base_cam()
        except Exception:  # noqa: BLE001  no calibration here: the demo is drawn without its fingertip
            t_bc = None
    origin = demo.origin if demo.origin is not None else float(demo.arm.t[0])
    left = perception_view.render(
        demo,
        pd,
        perception_view.frame_at(pd, origin + t),
        shapes=shapes_d,
        base_from_camera=t_bc,
        demo_time=t,
        title=f"demo {d.name}",
    )
    if a is None:
        out = left
    elif pa is None:
        right = np.full_like(left, 25)
        perception_view._text(right, "this act is not processed yet", (8, 20))
        out = perception_view.pair(left, right)
    else:
        meta = json.loads((a / "act.json").read_text())
        keypoints = demo.annotations.keypoints if demo.annotations else []
        clock = perception_view.act_clock(meta.get("targets") or [], keypoints)
        at = perception_view.act_time_at(clock, pa.t, t)
        if at is None:
            right = np.full_like(left, 25)
            perception_view._text(right, f"the act did not reach demo {t:.2f} s", (8, 20))
        else:
            right = perception_view.render(
                act, pa, perception_view.frame_at(pa, at), shapes=shapes_a, demo_time=t, title=f"act {a.name}"
            )
        out = perception_view.pair(left, right)
    ok, jpg = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return jpg.tobytes()


@router.get("/compare/frame.jpg")
async def compare_frame(demo: str, t: float, act: str | None = None) -> Response:
    """The demo at demo time ``t`` and, beside it, the act where it was at the same moment of the demo."""
    d, a = _folders(demo, act)
    jpg = await asyncio.get_running_loop().run_in_executor(_DRAW_EXECUTOR, _frame, d, a, t)
    return Response(content=jpg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/compare")
async def compare_page(demo: str, act: str | None = None) -> HTMLResponse:
    _folders(demo, act)
    return HTMLResponse(PAGE)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>demo and act</title>
<style>
body{margin:0;background:#111;color:#ddd;font:14px sans-serif}
#controls{display:flex;gap:8px;align-items:center;padding:8px}
#controls button{width:40px}
#slider{flex:1}
#under{padding:0 8px 8px;min-height:1.4em}
img{width:100%;display:block}
</style></head><body>
<div id="controls"><button id="prev">&#9664;</button><input id="slider" type="range" min="0" max="1" step="0.0333"
 value="0"><button id="next">&#9654;</button></div>
<div id="under"><span id="time"></span> <span id="status"></span> <button id="process" hidden>Process</button></div>
<img id="frame" alt="">
<script>
const q = new URLSearchParams(location.search);
const base = "/api/pregrasp/compare";
const args = () => "demo=" + encodeURIComponent(q.get("demo")) + (q.get("act") ? "&act=" + encodeURIComponent(q.get("act")) : "");
const slider = document.getElementById("slider"), img = document.getElementById("frame");
let marks = [], pending = null;
function show() {
  const t = Number(slider.value);
  const mark = marks.filter(m => m.t <= t + 1e-6).pop();
  document.getElementById("time").textContent = "demo " + t.toFixed(2) + " s" + (mark ? "  after " + mark.label : "");
  clearTimeout(pending);
  pending = setTimeout(() => { img.src = base + "/frame.jpg?" + args() + "&t=" + t.toFixed(3); }, 60);
}
function step(d) { slider.value = Math.min(Number(slider.max), Math.max(0, Number(slider.value) + d * Number(slider.step))); show(); }
document.getElementById("prev").onclick = () => step(-1);
document.getElementById("next").onclick = () => step(1);
document.addEventListener("keydown", e => { if (e.key === "ArrowLeft") step(-1); if (e.key === "ArrowRight") step(1); });
slider.oninput = show;
document.getElementById("process").onclick = async () => {
  await fetch(base + "/process", {method: "POST", headers: {"content-type": "application/json"},
    body: JSON.stringify({demo: q.get("demo"), act: q.get("act")})});
  poll();
};
async function poll() {
  const info = await (await fetch(base + "/info?" + args())).json();
  marks = info.key_frames; slider.max = info.span[1];
  const sides = [["demo", info.demo], ["act", info.act]].filter(([, s]) => s);
  const missing = sides.filter(([, s]) => !s.processed);
  const busy = sides.filter(([, s]) => s.processing);
  document.getElementById("process").hidden = !missing.length || busy.length > 0;
  document.getElementById("status").textContent = busy.length
    ? busy.map(([n, s]) => n + " " + s.processing).join(", ")
    : missing.length ? missing.map(([n]) => n).join(" and ") + " not processed by pipeline " + info.version : "";
  if (busy.length) setTimeout(poll, 2000);
  show();
}
poll();
</script></body></html>"""
