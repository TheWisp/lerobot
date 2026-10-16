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

"""Read-only manipulation session status for the GUI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from lerobot.showservo import motion as core

from . import acquisition, camera, demos, runtime

router = APIRouter()


@router.get("/state")
async def state() -> dict:
    from .. import jog, showservo

    s = runtime._state
    with s.lock:
        teach, test = s.teach, s.test
    w = s.worker
    with s.lock:
        runtime._expire_jobs_locked(s)
        teach_pending = s.teach_job is not None
        find_pending = s.find_job is not None
        log_tail = w.log[-12:]
    out: dict[str, Any] = {
        "page_version": camera._page_version(),
        "camera_live": showservo.live_camera() is not None,
        "arm_connected": jog.current_robot_id() is not None,
        "worker": {
            "running": w.running,
            "recovery_running": w.recovery_proc is not None and w.recovery_proc.poll() is None,
            "recovery_ready": any("worker ready" in line for line in w.recovery_log),
            "recovery_log": "\n".join(w.recovery_log[-12:]),
            "ready": any("worker ready" in line for line in w.log),
            "log": "\n".join(log_tail),
        },
        "teach_pending": teach_pending,
        "find_pending": find_pending,
        "flat": s.flat,
        "trust_share": s.trust_share,
        "depth_check": s.depth_check,
        "depth_tol_mm": s.depth_tol_m * 1000.0,
        "track": {
            "on": s.track.on,
            "algo": s.track.algo,
            "follow": s.track.follow,
            "hover_mm": s.track.hover_mm,
            "fps": s.track.fps,
            "last": dict(s.track.last),
        },
        "act": {
            "on": s.act.on,
            "step": s.act.step,
            "ok": s.act.ok,
            "reason": s.act.reason,
            "progress": s.act.progress,
            "speed": s.act.speed,
            "queue_horizon_s": s.act.queue_horizon_s,
            "plan": s.act.plan,
            "place": s.act.place,
        },
        "located": {}
        if s.demo is None
        else {
            name: acquisition._located_info(f)
            for name in list(s.located)
            if (f := acquisition._located(s.demo, name)) is not None
        },
        "demo": None if s.demo is None else demos._demo_info(s.demo),
        "recording": None
        if s.recording is None
        else {
            "since": s.recording["t0"],
            "frames": len(s.recording["frames"]),
            "samples": jog.record_count(),
        },
        "teach": None,
        "test": None,
    }
    if teach is not None:
        out["teach"] = {
            "at": teach.at,
            "box": list(teach.box),
            **acquisition._teach_info(teach.keypoints),
            "tip_mm": None if teach.tip_pose is None else (teach.tip_pose[:3, 3] * 1000.0).tolist(),
            "gripper": teach.gripper,
        }
    if test is not None:
        r = test.result
        info = {k: v for k, v in r.items() if k not in runtime._HEAVY}
        if r.get("ok"):
            info["motion"] = core.motion_summary(r["delta_cam"])
        if test.transported is not None:
            info["transported_tip_mm"] = (test.transported[:3, 3] * 1000.0).tolist()
        out["test"] = {"at": test.at, **info}
    return out
