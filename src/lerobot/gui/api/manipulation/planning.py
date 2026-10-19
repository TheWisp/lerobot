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

"""Object-relative key-frame composition, reach checks and placement planning."""

from __future__ import annotations

import functools
import hashlib
import pathlib
import threading
from collections.abc import Callable
from typing import Any

import numpy as np
from fastapi import HTTPException

from lerobot.showservo import motion as core

from . import acquisition, holds, recording, runtime

# An act finds its objects afresh before the arm moves. A new find whose ends are in doubt keeps the ends of the find
# the act started from (measured, or confirmed by the operator) while it puts the object in the same place. Finds of a
# still dowel on 2026-10-16 moved its centre and ends by at most 4.5 mm between them; its other end assignment moved
# them 104 mm.
RECONFIRM_M = 0.015

# The live view draws the act's path on every frame it renders, about ten a second, but the path changes only when a
# key frame's target moves. Re-planning it cost 44-51 ms a frame through the acts of 2026-10-16, holding the GIL beside
# the motor loop. It is planned again once a key frame's target has moved further than the line can show at the tray
# (2 mm or 1 deg, about 3 pixels); in those acts' recorded object poses that was a quarter of the rendered frames.
PREVIEW_REPLAN_M = 0.002
PREVIEW_REPLAN_DEG = 1.0
_preview_lock = threading.Lock()
_preview_last: dict[str, Any] = {}  # the live view's last planned path and what it was planned for


def _recovered_pose_token(demo: runtime._Demo, teach: runtime._Teach) -> str:
    """Bind a force choice to exactly the find and key frames shown in its preview."""
    ref = teach.keypoints.get("ref") or {}
    digest = hashlib.sha256()
    digest.update(f"{demo.name}:{demo.root}:{teach.stamp}:{demo.landing}".encode())
    digest.update(repr(demo.keypoints).encode())
    digest.update(repr(ref.get("surface_constraint")).encode())
    digest.update(np.asarray(ref.get("delta"), dtype=np.float64).tobytes())
    return digest.hexdigest()[:24]


def _recovered_pose_problem(obj: str, found: dict[str, Any], *, preview: bool = False) -> str:
    """A recovered axis has alternatives but no collision-checked grasp choice yet."""
    if (found.get("surface_constraint") or {}).get("state") != "recovered":
        return ""
    # One act at a time owns this state. A preview may draw the proposed path,
    # while only an explicit operator confirmation lets the runner use it.
    if preview:
        return ""
    state = runtime._state
    act = state.act
    teach, demo = state.teach, state.demo
    if act.on and _same_place(demo, obj, act.start_poses.get(obj), found.get("delta")):
        return ""  # where the act's starting find, whose ends were measured or confirmed, put it
    if act.on and act.force_recovered_pose and obj == act.recovered_pose_object:
        if (
            demo is not None
            and teach is not None
            and found is teach.keypoints.get("ref")
            and act.recovered_pose_token == _recovered_pose_token(demo, teach)
        ):
            return ""
        return (
            f"{obj}'s proposed grasp changed after confirmation; review its new preview and confirm it again"
        )
    return (
        f"{obj}'s measured pose has equivalent end assignments; collision-checked grasp selection is required"
    )


def _same_place(demo: runtime._Demo | None, obj: str, a: Any, b: Any) -> bool:
    """Whether two finds' motions (camera frame) put ``obj`` in one place: none of the points a find carries
    (tracking._view_points) lands more than :data:`RECONFIRM_M` apart."""
    if a is None or b is None or demo is None or demo.objects.get(obj, {}).get("status") != "done":
        return False
    from . import tracking

    points = tracking._view_points(demo, obj)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    apart = points @ (a[:3, :3] - b[:3, :3]).T + (a[:3, 3] - b[:3, 3])
    return bool(len(points)) and float(np.max(np.linalg.norm(apart, axis=1))) <= RECONFIRM_M


def _object_motion(
    demo: runtime._Demo, obj: str, t_bc: np.ndarray, carry: Callable[[float], np.ndarray] | None = None
) -> tuple[np.ndarray | None, str]:
    """``(motion, problem)``: how ``obj``, one of :func:`_other_objects`, moved from the demo to now, base frame. Its
    last locate against the demo's view of it, times the inverse of where the demo's track had it on its pose frame
    (:func:`_pose_choice`). With ``carry``, the locate is moved on by ``carry(stamp)``, how the object moved since the
    locate's frame was read (camera frame). Pre: a key frame is held in ``obj``'s frame."""
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    found = acquisition._located(demo, obj)
    if found is None:
        return None, f"click {obj} in the camera view to find it"
    if not found["ok"]:
        return None, found["reason"]
    if problem := _recovered_pose_problem(obj, found):
        return None, problem
    if acquisition._weak(found):
        return None, acquisition._weak(found)
    f = _pose_frame(demo, obj)
    if f is None:
        return None, f"{obj}'s demo pose is not visible before this interaction"
    with runtime._state.lock:  # a placing view replaces both at once
        pin, stamp = np.asarray(found["delta"], dtype=float), found.get("stamp")
    if carry is not None and stamp is not None:
        pin = carry(float(stamp)) @ pin
    reference, problem = holds._demo_control_pose(demo, obj, f)
    if reference is None:
        return None, problem
    cam = pin @ np.linalg.inv(reference)
    return t_bc @ cam @ np.linalg.inv(t_bc), ""


def _object_points(demo: runtime._Demo, obj: str, t_bc: np.ndarray) -> np.ndarray:
    """The designated object's surface on the frame it was clicked on, base frame (N, 3): its depth under its mask
    there, every fourth pixel. The finds' motions are from that view. Pre: the object is done on a stream demo."""
    o = demo.objects[obj]
    _rgb, depth = recording._stream_frame(demo.recording, int(o["frame"]))
    ys, xs = np.nonzero(np.asarray(o["mask"], dtype=bool) & (depth > 0))
    ys, xs = ys[::4], xs[::4]
    k = np.loadtxt(pathlib.Path(demo.recording) / "cam_K.txt")
    z = depth[ys, xs]
    cam = np.stack([(xs - k[0, 2]) * z / k[0, 0], (ys - k[1, 2]) * z / k[1, 1], z], axis=1)
    return cam @ t_bc[:3, :3].T + t_bc[:3, 3]


def _normal_marks(marks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Marks as the server keeps them, in time order: a time and a kind, with an object, a frame (always, on a key
    frame mark), a move, a gripper and a lift's height when given. Raises HTTPException for a mark without a time or a kind."""
    try:
        return sorted(
            (
                {
                    "t": float(k["t"]),
                    "kind": str(k["kind"]),
                    **({"object": str(k["object"])} if k.get("object") else {}),
                    **(
                        {"frame": str(k.get("frame") or "")}
                        if k["kind"] == core.KEY_KIND or k.get("frame")
                        else {}
                    ),
                    **{key: str(k[key]) for key in ("move", "grip", "constraint") if k.get(key)},
                    **({"up_mm": float(k["up_mm"])} if k.get("up_mm") is not None else {}),
                }
                for k in marks
            ),
            key=lambda k: k["t"],
        )
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(422, f"a mark needs a time and a kind: {e}") from e


def _marks_problem(demo: runtime._Demo, kps: list[dict[str, Any]]) -> str:
    """Why ``kps`` cannot be the demo's marks, or "" when they can: their own rules (:func:`core.keypoints_problem`);
    every object they name tracked in the demo; a pose read on a frame its object is seen on; an unnamed frame only on
    a demo recorded after a teach; and no key frame the demo made holding an object held in that object's own frame,
    which the gripper carries."""
    problem = core.keypoints_problem(kps, float(demo.t[0]), float(demo.t[-1]))
    if problem:
        return problem
    frames = core.key_frames(kps)
    framed = {k["frame"] for k in frames} - {"", core.WORLD_FRAME}
    for named in sorted(({k.get("object", "") for k in kps} | framed) - {""}):
        if demo.objects.get(named, {}).get("status") != "done":
            return f"{named!r} is not a tracked object of this demo"
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    for k in kps:
        if k["kind"] == "pose" and len(times):
            f = int(np.argmin(np.abs(times - (demo.t0 + k["t"]))))
            if not demo.objects[k["object"]]["seen"][f]:
                return (
                    f"{k['object']} is hidden at {k['t']:.2f} s: read its pose on a frame where it is in view"
                )
    if any(k["frame"] == "" for k in frames) and not demo.taught:
        return "nothing was taught before this demo: the key frames follow an object clicked on its recording"
    trial = runtime._Demo(**{**demo.__dict__, "keypoints": kps})
    holdings = _holdings(trial)
    try:
        core.interaction_phases(holdings, trial.t, frames)
    except ValueError as exc:
        return str(exc)
    for n, k in enumerate(frames):
        holding = core.holding_at(holdings, trial.t, frames, n)
        if (
            holding is not None
            and k["frame"] == holding["frame"]
            and core.held_at(holding, trial.t, frames, n)
        ):
            who = holding["frame"] or "the object"
            return (
                f"{k['label']} at {k['t']:.2f} s is held in {who}'s frame while the gripper holds {who}: "
                "hold it in another object's frame or the world's"
            )
    return ""


def _has_key_frames(demo: runtime._Demo) -> bool:
    return bool(core.key_frames(demo.keypoints))


def _act_objects(demo: runtime._Demo) -> list[str]:
    """The designated objects the demo's key frames are held in, in the order the first key frame in each comes."""
    frames = core.key_frames(demo.keypoints)
    return list(dict.fromkeys(k["frame"] for k in frames if k["frame"] not in ("", core.WORLD_FRAME)))


def _session_object(demo: runtime._Demo) -> str | None:
    """The object whose find starts the live Point2Pose session, the first the key frames are held in (the others
    join it: :func:`_session_names`); None for the object taught before the demo, which the teach follows."""
    return next(iter(_act_objects(demo)), None)


def _other_objects(demo: runtime._Demo) -> list[str]:
    """The act's objects that join the session started from :func:`_session_object`'s find, followed by
    :func:`_apply_others` and moved since the demo by :func:`_object_motion`."""
    first = _session_object(demo)
    return [n for n in _act_objects(demo) if n != first]


def _landing_object(demo: runtime._Demo) -> str | None:
    """The object the demo's landing rule turns its key frames about: the first a key frame made holding is held
    in; None when no key frame is made holding in a designated object's frame."""
    frames = core.key_frames(demo.keypoints)
    holding = _holding(demo)
    return next(
        (
            k["frame"]
            for n, k in enumerate(frames)
            if core.held_at(holding, demo.t, frames, n) and k["frame"] not in ("", core.WORLD_FRAME)
        ),
        None,
    )


def _reference_motion(
    demo: runtime._Demo, teach: runtime._Teach | None, *, preview_pose: np.ndarray | None = None
) -> tuple[np.ndarray | None, str]:
    """``(R, problem)``: the live track's motion times ``R`` is the object's motion from the demo to now.

    The live track follows the object whose find started the session (:func:`_session_object`). For the object
    taught before the demo, ``R`` undoes where the object was when the demo began. For a designated object it is the
    live view's registration against the demo's view of it, times the inverse of where the demo's track had the
    object on its pose frame (:func:`_pose_choice`). ``R`` is None, with the reason, when that object has not been
    found live yet.
    """
    obj = _session_object(demo)
    if obj is None:
        if not demo.taught:
            return None, "the marks name no object: open Edit demo and save them for the object clicked there"
        return np.linalg.inv(demo.delta0), ""
    o = demo.objects.get(obj)
    if o is None or o.get("status") != "done" or demo.recording is None:
        return None, f"{obj!r} is not tracked in this demo"
    ref = (teach.keypoints.get("ref") if teach is not None else None) or {}
    if ref.get("object") != obj:
        return None, f"click {obj} in the camera view to find it"
    if not ref.get("ok") or ref.get("delta") is None:
        return None, ref.get("reason") or f"{obj} was not found in the live view"
    if problem := _recovered_pose_problem(obj, ref, preview=preview_pose is not None):
        return None, problem
    # A certified live track only follows motion since its find. It cannot
    # strengthen that find's registration to the demo. Apply the same gate as
    # _find_afresh before either runner reuses an already-running session.
    if core.find_strength(ref.get("inliers"), ref.get("card_points"))[0] is False:
        return None, (
            f"a weak find: {obj} matched {ref['inliers']} of the demo view's {ref['card_points']} points; "
            "turn it closer to how it lay in the demo, then press Act"
        )
    f = _pose_frame(demo, obj)
    if f is None:
        return None, f"{obj}'s demo pose is not visible before this interaction"
    reference, problem = holds._demo_control_pose(demo, obj, f)
    if reference is None:
        return None, problem
    return np.asarray(ref["delta"] if preview_pose is None else preview_pose, dtype=float) @ np.linalg.inv(
        reference
    ), ""


def _pose_frame(demo: runtime._Demo, obj: str | None) -> int | None:
    """The stream frame the act reads a designated object's demo pose from, as :func:`_pose_choice` picks it."""
    choice = None if obj is None else _pose_choice(demo, obj)
    return None if choice is None else choice[0]


def _approach_times(demo: runtime._Demo, obj: str) -> list[float]:
    """The moments of the lines held in ``obj``'s frame before the arm first follows the demo in it, while nothing
    the act does has touched it."""
    frames = core.key_frames(demo.keypoints)
    out = []
    for k in frames:
        if k["frame"] != obj:
            continue
        if k["move"] == "follow":
            break
        out.append(k["t"])
    return out


def _pose_choice(demo: runtime._Demo, obj: str) -> tuple[int, str] | None:
    """``(stream frame, where it comes from)``: the frame the act reads ``obj``'s demo pose from.

    The frame the operator set for it with a pose mark ("set"). Else the frame it was clicked on ("clicked"): its
    view there is the one every find matches against, so its pose there is exact, and the operator clicked it where
    it could be seen; taken when it comes no later than the arm first follows the demo in its frame. Else the last
    frame it was seen at or before its first key frame. None when ``obj`` is not tracked in the demo or no key frame
    is held in its frame yet (:func:`_approach_times`).
    """
    o = demo.objects.get(obj)
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    pre = _approach_times(demo, obj)
    if o is None or o.get("status") != "done" or not len(times) or not pre:
        return None
    chosen = next(
        (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == obj), None
    )
    if chosen is not None:
        return int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set"
    click = int(o["frame"])
    if times[click] - demo.t0 <= max(pre):
        return click, "clicked"
    f0 = int(np.argmin(np.abs(times - (demo.t0 + min(pre)))))
    seen = np.flatnonzero(np.asarray(o["seen"])[: f0 + 1])
    return (int(seen[-1]) if len(seen) else click), "last seen by its first key frame"


def _object_pose_info(demo: runtime._Demo, name: str) -> dict[str, Any]:
    """Where the editor says ``name``'s demo pose is read: ``pose_t`` (seconds into the demo) and ``pose_from``. An
    object no mark follows yet reports the frame set for it, else the frame it was clicked on."""
    choice = _pose_choice(demo, name)
    times = recording._stream_times(demo.recording) if demo.recording is not None else np.zeros(0)
    if choice is None:
        o = demo.objects[name]
        chosen = next(
            (float(k["t"]) for k in demo.keypoints if k["kind"] == "pose" and k.get("object") == name), None
        )
        choice = (
            (int(o["frame"]), "clicked")
            if chosen is None
            else (int(np.argmin(np.abs(times - (demo.t0 + chosen)))), "set")
        )
    frame, where = choice
    return {"pose_t": float(times[frame] - demo.t0) if len(times) > frame else None, "pose_from": where}


def _delta_base(
    demo: runtime._Demo, delta_cam: np.ndarray, t_bc: np.ndarray, *, preview_pose: np.ndarray | None = None
) -> np.ndarray:
    """The object's motion from the demo to now, in the base frame: the tracker's fit as it is.

    Pre: :func:`_reference_motion` has no problem for the current teach.
    """
    with runtime._state.lock:
        teach = runtime._state.teach
    ref, problem = _reference_motion(demo, teach, preview_pose=preview_pose)
    assert ref is not None, problem
    rel = np.asarray(delta_cam, dtype=float) @ ref
    return t_bc @ rel @ np.linalg.inv(t_bc)


def _certified_since(since: float | None) -> list[tuple[float, np.ndarray]]:
    """The tracker's certified motions with their frames' times, oldest first, of frames read after ``since`` (wall
    clock), or all of them."""
    with runtime._state.lock:
        hist = list(runtime._state.track.history)
    return [
        (w, np.asarray(d, dtype=float))
        for (w, ok, d) in hist
        if ok and d is not None and (since is None or w > since)
    ]


def _frame_motion(
    demo: runtime._Demo,
    delta_cam: np.ndarray | None,
    t_bc: np.ndarray,
    motions: dict[str, np.ndarray] | None,
    aim: np.ndarray | None = None,
    preview_pose: np.ndarray | None = None,
) -> Callable[[str], np.ndarray | None]:
    """Each key frame's frame's motion since the demo, base frame, read when asked: the world's none; the object the
    live track follows (:func:`_delta_base`, ``aim`` before it); another object's from ``motions`` (by name); None
    for one not found yet."""
    picked = _session_object(demo) or ""

    def motion(frame: str) -> np.ndarray | None:
        if frame == core.WORLD_FRAME:
            return np.eye(4)
        if frame == picked:
            base = _delta_base(demo, delta_cam, t_bc, preview_pose=preview_pose)
            return base if aim is None else aim @ base
        return (motions or {}).get(frame)

    return motion


def _found_motions(
    demo: runtime._Demo, t_bc: np.ndarray, carry: Callable[[str, float], np.ndarray] | None = None
) -> dict[str, np.ndarray]:
    """Each of :func:`_other_objects` found so far, by name: its motion since the demo (:func:`_object_motion`),
    moved on by ``carry(name, stamp)`` when given."""
    out = {}
    for name in _other_objects(demo):
        motion = _object_motion(demo, name, t_bc, None if carry is None else functools.partial(carry, name))[
            0
        ]
        if motion is not None:
            out[name] = motion
    return out


def _act_preview(
    demo: runtime._Demo, delta_cam: np.ndarray, t_bc: np.ndarray, *, preview_pose: np.ndarray | None = None
) -> np.ndarray | None:
    """The act's fingertip path from its first key frame on, for the camera view, through every key frame whose
    object has been found (as if held as in the demo); None until a key frame is marked."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    from .. import jog

    if not _has_key_frames(demo):
        return None
    with runtime._state.lock:
        teach = runtime._state.teach
    if _reference_motion(demo, teach, preview_pose=preview_pose)[0] is None:
        return None
    gi = MOTOR_NAMES.index("gripper")
    steps = _act_steps(
        demo,
        "all",
        0,
        _frame_motion(demo, delta_cam, t_bc, _found_motions(demo, t_bc), preview_pose=preview_pose),
        None,
    )
    if not steps:
        return None
    key = (
        id(demo),
        repr(demo.keypoints),
        np.asarray(t_bc).tobytes(),
        [(k["t"], k["frame"], k["move"]) for k in steps],
    )
    targets = [k["motion"] @ demo.tips[int(np.argmin(np.abs(demo.t - k["t"])))] @ k["fix"] for k in steps]
    if preview_pose is None:  # a review's candidate poses are planned as asked
        with _preview_lock:
            last = dict(_preview_last)
        if last.get("key") == key and all(
            distance <= PREVIEW_REPLAN_M and angle <= PREVIEW_REPLAN_DEG
            for distance, angle in (
                core.pose_residual(a, b) for a, b in zip(last["targets"], targets, strict=True)
            )
        ):
            return last["poses"]
    i = int(np.argmin(np.abs(demo.t - steps[0]["t"])))
    plan = core.plan_key_frames(
        steps,
        demo.t,
        demo.tips,
        demo.q_cmd[:, gi],
        demo.q_obs,
        steps[0]["motion"] @ demo.tips[i] @ steps[0]["fix"],
        float(demo.q_cmd[i, gi]),
        jog.MAX_LINEAR_M_S,
        jog.MAX_ANGULAR_RAD_S,
        jog.GRIP_UNITS_S,
        1.0,
        jog.HZ,
    )
    if preview_pose is None:
        with _preview_lock:
            _preview_last.update(key=key, targets=targets, poses=plan["poses"])
    return plan["poses"]


def _holding(demo: runtime._Demo) -> dict[str, Any] | None:
    """The demo's first holding, or None."""
    holdings = _holdings(demo)
    return holdings[0] if holdings else None


def _holdings(demo: runtime._Demo) -> list[dict[str, Any]]:
    """All recorded attachments; object identity comes from the key frames."""
    from lerobot.robots.so107_description.joint_alignment import MOTOR_NAMES

    gi = MOTOR_NAMES.index("gripper")
    return core.demo_holdings(demo.t, demo.q_cmd[:, gi], demo.q_obs[:, gi], core.key_frames(demo.keypoints))


def _grasp_split(frames: list[dict[str, Any]], holding: dict[str, Any] | None) -> int:
    """How many of the key frames the act does before it checks the grasp: those up to the follow the demo's hold
    began in, else up to the first follow, else all of them."""
    if holding is not None:
        return holding["begins"] + 1
    return next((n + 1 for n, k in enumerate(frames) if k["move"] == "follow"), len(frames))


def _act_steps(
    demo: runtime._Demo,
    part: str | tuple[int, int],
    skip: int,
    motion: Callable[[str], np.ndarray | None],
    hold_fix: np.ndarray | None,
    grasp_adjust: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """The demo's key frames an act plans, resolved for where things are now (:func:`core.plan_key_frames`).

    ``part`` "grasp" takes those up to the grasp check (:func:`_grasp_split`), "place" the rest, "all" both, and
    ``(start, stop)`` those numbered so in time order; ``skip`` leaves out that many of the part's leading lines,
    already reached. ``motion(frame)`` is that frame's motion since the demo, base frame, or None while unknown (an
    object not found yet): the steps stop before the first such. Those the demo made holding something are corrected
    by ``hold_fix`` (the identity when None)."""
    frames = core.key_frames(demo.keypoints)
    holding = _holding(demo)
    split = _grasp_split(frames, holding)
    holdings = _holdings(demo)
    if isinstance(part, tuple):
        lo, hi = part
    else:
        lo, hi = {"grasp": (0, split), "place": (split, len(frames)), "all": (0, len(frames))}[part]
    assert skip == 0 or all(k["move"] == "line" for k in frames[lo : lo + skip + 1]), (
        "a line is left to reach"
    )
    fix = np.eye(4) if hold_fix is None else np.asarray(hold_fix, dtype=float)
    steps: list[dict[str, Any]] = []
    for n in range(lo + skip, hi):
        m = motion(frames[n]["frame"])
        if m is None:
            break
        attached = core.holding_at(holdings, demo.t, frames, n)
        held = attached is not None
        steps.append(
            {
                **frames[n],
                "since": frames[n - 1]["t"] if n else None,
                "motion": m,
                "fix": fix if attached is not None and attached == holding else np.eye(4),
                "adjust": (
                    np.asarray(grasp_adjust, float)
                    if grasp_adjust is not None
                    and holding is not None
                    and n <= holding["begins"]
                    and frames[n]["frame"] == holding["frame"]
                    else np.eye(4)
                ),
                "held": held,
                "held_object": None if attached is None else attached["frame"],
            }
        )
    return steps
