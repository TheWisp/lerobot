"""Read-only, stationary RGB-D and object-pose measurements for planned resets.

The GUI owns the camera and arm. These endpoints never command motion and never
connect to serial. A path token can measure only at an arrived phase boundary.
"""

import asyncio
import copy
import io
import json
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from . import _pregrasp_core as core, jog, pregrasp
from ._guarded_path import ARRIVAL_ANGLE_DEG, ARRIVAL_DEG, ARRIVAL_M

router = APIRouter(prefix="/api/reset", tags=["reset observation"])


def _at_cartesian_target(tip, target, observed, commanded):
    """Use the path's arrival units; matrix entries mix metres and rotation coefficients."""
    if target is None:
        return False
    position, angle = core.pose_residual(tip, target)
    lag = max(abs(observed[name] - commanded[name]) for name in observed if name != "gripper")
    return bool(
        np.isfinite([position, angle, lag]).all()
        and position <= ARRIVAL_M
        and angle <= ARRIVAL_ANGLE_DEG
        and lag <= ARRIVAL_DEG
    )


def _stationary(path_id):
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    j = jog._jog
    with j.lock:
        state = {
            "q_obs": dict(j.q_obs),
            "q_cmd": dict(j.q_cmd),
            "read_age_s": None if not j.last_read_at else time.monotonic() - j.last_read_at,
        }
        if not j.connected or j.halted or j.path_busy:
            raise HTTPException(409, "reset observation needs a connected, unhalted, stationary arm")
        if path_id is not None:
            if (
                j.mode != "path"
                or not j.guarded_path
                or j.guarded_path.identity != path_id
                or j.guarded_path.status != "done"
            ):
                raise HTTPException(409, "reset observation needs this client's arrived path")
        elif j.mode != "cartesian":
            raise HTTPException(409, "a path token is required while another controller owns the arm")
        if state["read_age_s"] is None or state["read_age_s"] > 0.25:
            raise HTTPException(409, "reset observation has stale arm telemetry")
        names = tuple(MOTOR_NAMES)
        q = np.array([j.q_obs[n] for n in names])
        if not np.isfinite(q).all():
            raise HTTPException(409, "reset observation has invalid encoders")
        kin = copy.deepcopy(j.kin)
        target = None if j.target is None else j.target.copy()
    tip = kin.forward_kinematics(q)
    if path_id is None and not _at_cartesian_target(tip, target, state["q_obs"], state["q_cmd"]):
        raise HTTPException(409, "arm has a pending Cartesian target")
    if pregrasp._state.act.on or pregrasp._state.track.follow:
        raise HTTPException(409, "stop the act/follow before observing for a reset")
    return {"owner": j, "names": names, "state": state, "q": q, "tip": tip}


def _same_pose(before, after):
    if before["owner"] is not after["owner"] or before["names"] != after["names"]:
        raise HTTPException(409, "arm changed during reset observation")
    if (
        np.max(np.abs(before["q"] - after["q"])) > 0.25
        or np.linalg.norm(before["tip"][:3, 3] - after["tip"][:3, 3]) > 0.001
    ):
        raise HTTPException(409, "arm moved during reset observation")


async def _view(path_id):
    before = _stationary(path_id)
    rgb, depth, intr = await pregrasp._frame()
    after = _stationary(path_id)
    _same_pose(before, after)
    check = await pregrasp._camera_check_async(rgb)
    if check and check.get("moved"):
        raise HTTPException(409, "camera/tray moved since calibration")
    return before, rgb, depth, intr


def _pack(view, t_bc):
    before, rgb, depth, intr = view
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        rgb=rgb,
        depth=depth,
        intr=np.array(json.dumps(intr)),
        motor_names=np.array(before["names"]),
        q_obs=before["q"],
        q_cmd=np.array([before["state"]["q_cmd"][n] for n in before["names"]]),
        tip=before["tip"],
        t_base_cam=t_bc,
    )
    return buf.getvalue()


@router.get("/frame.npz")
async def frame(path_id: str | None = None):
    view = await _view(path_id)
    t_bc = pregrasp._t_base_cam()
    body = await asyncio.get_running_loop().run_in_executor(pregrasp._RENDER_EXECUTOR, _pack, view, t_bc)
    return Response(content=body, media_type="application/octet-stream")


class PoseBody(BaseModel):
    object: str
    click: list[int]
    path_id: str | None = None


@router.post("/object_pose")
async def object_pose(body: PoseBody):
    """Fresh strong finds; return measured motion from the demo reference in base coordinates.

    ``hold_from_reference`` maps demo-reference base points into the current
    tool frame. It is not the object's box frame: callers must compose their
    independently registered reference-to-box transform, not treat it as one.
    """
    if len(body.click) != 2:
        raise HTTPException(422, "click needs two pixel coordinates")
    with pregrasp._state.lock:
        demo = pregrasp._state.demo
        if demo is None or demo.objects.get(body.object, {}).get("status") != "done":
            raise HTTPException(409, "object must be tracked in the loaded demo")
        if not pregrasp._state.worker.running:
            raise HTTPException(409, "start the perception worker")
    t_bc = pregrasp._t_base_cam()
    points = await asyncio.get_running_loop().run_in_executor(
        pregrasp._RENDER_EXECUTOR, pregrasp._object_points, demo, body.object, t_bc
    )
    first = _stationary(body.path_id)
    holds, evidence = [], []
    for _ in range(core.HOLD_VIEWS + 2):
        before, rgb, depth, intr = await _view(body.path_id)
        _same_pose(first, before)
        found = await pregrasp._locate(
            body.object,
            rgb,
            depth,
            intr,
            body.click,
            lambda: False,
            track=False,
            log={"why": "stationary planned-reset measurement"},
        )
        after = _stationary(body.path_id)
        _same_pose(first, after)
        if pregrasp._state.demo is not demo:
            raise HTTPException(409, "demo changed during reset measurement")
        evidence.append(pregrasp._find_info(found))
        if found.get("ok") and core.find_strength(found.get("inliers"), found.get("card_points"))[0] is True:
            delta = np.asarray(found["delta"])
            if delta.shape == (4, 4) and np.isfinite(delta).all():
                motion = t_bc @ delta @ np.linalg.inv(t_bc)
                holds.append(np.linalg.inv(before["tip"]) @ motion)
        if len(holds) >= core.HOLD_VIEWS:
            break
    measured = core.average_hold(holds, points.mean(axis=0))
    if measured is None:
        raise HTTPException(409, "not enough consistent strong object-pose measurements")
    hold = measured["hold"]
    return {
        "object": body.object,
        "reference": [demo.name, int(demo.objects[body.object]["frame"])],
        "motion_base": (after["tip"] @ hold).tolist(),
        "hold_from_reference": hold.tolist(),
        "tip": after["tip"].tolist(),
        "q_obs": after["state"]["q_obs"],
        "q_cmd": after["state"]["q_cmd"],
        "t_base_cam": t_bc.tolist(),
        "views": measured["views"],
        "accepted": measured["n"],
        "spread_mm": measured["spread_mm"],
        "spread_deg": measured["spread_deg"],
        "finds": evidence,
    }
