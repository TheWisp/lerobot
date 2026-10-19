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

"""Starting and stopping an act (the act engine: engine_act), and the manual approach control."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from . import holds, planning, runtime

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/go")
async def go(body: runtime.GoBody) -> dict:
    """Walk the fingertip to the transported pre-grasp, ``hover_mm`` above it (0 to go exactly there)."""
    from .. import jog

    with runtime._state.lock:
        test, teach = runtime._state.test, runtime._state.teach
    if test is None or test.transported is None:
        raise HTTPException(409, "no transported pose; capture the test frame first")
    check = test.result.get("camera_check") or {}
    if check.get("moved"):
        raise HTTPException(
            409,
            f"the camera or the tray moved since the calibration (markers shifted {check['max_px']:.1f} px); "
            "redo the camera calibration before going anywhere",
        )
    pose = test.transported.copy()
    pose[2, 3] += body.hover_mm / 1000.0
    try:
        jog.set_target_pose(pose)
        # The gripper opening is part of the taught pose.
        if teach is not None and teach.gripper is not None:
            jog.set_gripper(teach.gripper)
    except RuntimeError as e:
        raise HTTPException(409, str(e)) from e
    return {"target_mm": (pose[:3, 3] * 1000.0).tolist(), "gripper": None if teach is None else teach.gripper}


class ActBody(BaseModel):
    speed: float = 1.0  # scales the demo's clock
    confirm_recovered_pose: str | None = None  # token from the warning for this find and demo
    perturb: list[dict[str, Any]] | None = None  # the engine's faults injected on purpose (engine.perturb)
    model_config = ConfigDict(extra="forbid")


async def _act_task(speed: float, server: str | None = None) -> None:
    """Validate demo references, then run the act engine."""
    with runtime._state.lock:
        demo, act = runtime._state.demo, runtime._state.act
    if demo is not None:
        try:
            problem = await asyncio.get_running_loop().run_in_executor(
                runtime._RENDER_EXECUTOR, holds._prepare_demo_references, demo
            )
        except Exception as exc:
            logger.exception("Demo surface validation failed before motion")
            problem = f"demo surface validation error: {type(exc).__name__}: {exc}"
        with runtime._state.lock:
            stale = runtime._state.demo is not demo or runtime._state.act is not act or act.stop_requested
        if stale or problem:
            act.on, act.ok, act.step = False, False, "aborted"
            act.reason = "demo or act changed during surface validation" if stale else problem
            return
    from . import engine_act

    await engine_act._act_engine_task(speed, server, act.perturb)


@router.post("/act")
async def act_start(body: ActBody, request: Request) -> dict:
    """Replay the demo on the object where it is now: the whole recorded path, carried by the object's motion."""
    from .. import jog
    from . import engine_act

    refusal = engine_act.arm_refusal()  # outside the act state's lock: it takes the jog's
    if refusal:
        raise HTTPException(409, refusal)
    with runtime._state.lock:
        demo, test, teach, act = (
            runtime._state.demo,
            runtime._state.test,
            runtime._state.teach,
            runtime._state.act,
        )
        if act.on:
            raise HTTPException(409, "an act is in progress")
        if demo is None:
            raise HTTPException(409, "record or load a demo first")
        if not planning._has_key_frames(demo):
            raise HTTPException(409, "mark a key frame first (Edit demo)")
        # No trusted live view is needed: the act finds the object afresh without one. A track kept through an act
        # has added the points it saw while carried, more than a view of the object at rest can show.
        if teach is None or runtime._state.teach_job is not None:
            raise HTTPException(409, "find the object first")
        reference = teach.keypoints.get("ref") or {}
        problem = planning._recovered_pose_problem(planning._session_object(demo) or "object", reference)
        token = planning._recovered_pose_token(demo, teach) if problem else None
        if problem and body.confirm_recovered_pose != token:
            raise HTTPException(
                409,
                {
                    "code": "recovered_pose_requires_confirmation",
                    "message": f"{problem}. The displayed path is a projection, not a collision check; "
                    "the selected end assignment could drive the gripper into the scene.",
                    "confirmation": token,
                    "preview": "/api/pregrasp/act/recovered-preview.jpg",
                },
            )
        if body.confirm_recovered_pose is not None and not problem:
            raise HTTPException(409, "the proposed pose changed; press Act to review it again")
        if test is not None and (test.result.get("camera_check") or {}).get("moved"):
            raise HTTPException(409, "the camera or the tray moved since the calibration; recalibrate first")
        runtime._state.track.follow = False  # the act owns the target now
        act.perturb = body.perturb
        act.force_recovered_pose = bool(problem)
        act.recovered_pose_token = token if problem else None
        act.recovered_pose_object = planning._session_object(demo) if problem else None
        # Finds whose ends are settled: measured, or (the object just checked) confirmed by the operator.
        settled = {
            name: found
            for name, found in runtime._state.located.items()
            if (found.get("surface_constraint") or {}).get("state") != "recovered"
        }
        settled[planning._session_object(demo) or ""] = reference
        act.start_poses = {
            name: np.asarray(found["delta"], dtype=float)
            for name, found in settled.items()
            if name and found.get("ok") and found.get("delta") is not None
        }
        act.finds = []
        act.on, act.ok, act.reason, act.step, act.stop_requested, act.progress, act.speed = (
            True,
            None,
            "",
            "starting",
            False,
            0.0,
            body.speed,
        )
    if jog.current_robot_id() is None:
        with runtime._state.lock:
            act.on = False
        raise HTTPException(409, "connect the arm first")
    act.task = asyncio.create_task(_act_task(body.speed, str(request.base_url).rstrip("/")))
    return {"status": "acting", "n": int(len(demo.t))}


@router.post("/act/stop")
async def act_stop() -> dict:
    """Stop the replay and hold the arm where it is."""
    from .. import jog

    with runtime._state.lock:
        runtime._state.act.stop_requested = True
    cur = jog.current_tip_and_anchor()
    if cur is not None:
        with contextlib.suppress(RuntimeError):
            jog.set_target_pose(cur[0])
    return {"status": "stopping"}
