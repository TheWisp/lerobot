"""Compare a carried object directly with the demonstration's carried views.

The reference is an observed surface, not a pose registered to the designation
view. Each reference has its own recorded fingertip frame. Never average their
camera-frame transforms together or install them in the designation's tracker.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

import numpy as np

from lerobot.showservo import motion as core, surface_constraint

from . import acquisition, holds, recording, runtime


def references(demo: runtime._Demo, obj: str) -> list[dict]:
    """Still, segmented observations in this attachment's carry window.

    The track need not have accepted a six-dimensional pose. Its segmentation
    proposes a click only; SAM resegments the reference before registration.
    Spread a bounded set across the existing sample window, including its ends.
    """
    if demo.recording is None or "masks" not in demo.objects.get(obj, {}):
        return []
    frames = holds._still_held_frames(demo, obj, need_seen=False)
    frames = [(f, i) for f, i in frames if np.any(demo.objects[obj]["masks"][f])]
    if len(frames) > core.HOLD_MIN_VIEWS:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, core.HOLD_MIN_VIEWS, dtype=int)]
    return [
        {"frame": f, "sample": i, "mask": demo.objects[obj]["masks"][f], "refresh_mask": True}
        for f, i in frames
    ]


def consensus(observations: list[dict], predicted_fix: np.ndarray | None) -> dict | None:
    """Compare attachments in fingertip coordinates, retaining ambiguous rotation.

    Registration supplies a material patch centre and a proposed rotation. The
    rotation must agree across reference views within the planner's angular
    tolerance AND across observations within the hold tolerance. Otherwise keep
    the predicted rotation and solve translation at the matched patch centre,
    not at the arbitrary origin of a camera transform.
    """
    from scipy.spatial.transform import Rotation

    if len(observations) < core.HOLD_MIN_VIEWS:
        return None
    rotations = Rotation.from_matrix(np.stack([o["rotation"] for o in observations]))
    mean = rotations.mean()
    refs = sorted({o["frame"] for o in observations})
    by_ref = [rotations[[i for i, o in enumerate(observations) if o["frame"] == f]].mean() for f in refs]
    between = max((float(np.degrees((a * b.inv()).magnitude())) for a in by_ref for b in by_ref), default=0.0)
    rotation_verified = (
        len(refs) >= 2
        and between <= core.ACT_REACH_TOL_DEG
        and np.max(np.degrees((rotations * mean.inv()).magnitude())) <= core.HOLD_AGREE_DEG
    )
    if not rotation_verified and predicted_fix is None:
        return None
    rotation = mean.as_matrix() if rotation_verified else np.asarray(predicted_fix)[:3, :3]
    shifts = np.stack([o["desired"] - rotation @ o["observed"] for o in observations])
    keep = np.linalg.norm(shifts - np.median(shifts, axis=0), axis=1) <= core.HOLD_AGREE_M
    if keep.sum() < core.HOLD_MIN_VIEWS or 2 * keep.sum() <= len(observations):
        return None
    fix = np.eye(4)
    fix[:3, :3], fix[:3, 3] = rotation, shifts[keep].mean(axis=0)
    return {
        "fix": fix,
        "n": int(keep.sum()),
        "views": len(observations),
        "spread_mm": float(np.median(np.linalg.norm(shifts[keep] - fix[:3, 3], axis=1)) * 1000),
        "spread_deg": float(np.median(np.degrees((rotations * mean.inv()).magnitude()))),
        "reference_frames": refs,
        "reference_rotation_disagreement_deg": between,
        "orientation": "measured" if rotation_verified else "predicted",
        "reason": "" if rotation_verified else "reference views do not independently agree on rotation",
    }


async def measure(
    demo: runtime._Demo,
    obj: str,
    t_bc: np.ndarray,
    predicted_fix: np.ndarray | None,
    stopped: Callable[[], bool],
) -> tuple[dict | None, str, list[dict]]:
    """Fresh held-view registration with the arm stopped before contact.

    Prediction proposes the search region; fresh segmentation, matching, depth
    constraints and multi-frame agreement determine the measured correction.
    Failures remain failures and share one bounded wait budget.
    """
    refs = await asyncio.get_running_loop().run_in_executor(runtime._RENDER_EXECUTOR, references, demo, obj)
    if not refs:
        return None, "no segmented still held views in this demo attachment", []
    intr_demo = holds._recording_intr(demo)
    until = time.monotonic() + runtime.HOLD_TRACK_WAIT_S
    observations, attempts = [], []

    def cancelled() -> bool:
        return stopped() or time.monotonic() >= until

    for attempt in range(core.HOLD_VIEWS + 2):
        if cancelled():
            break
        ref = refs[attempt % len(refs)]
        view = await holds._held_view(obj)
        if view is None:
            return None, "the arm went away", attempts
        rgb, depth, intr, tip, mask = view
        if "depth" not in ref:
            _, ref["depth"] = await asyncio.get_running_loop().run_in_executor(
                runtime._RENDER_EXECUTOR, recording._stream_frame, demo.recording, ref["frame"]
            )
        import cv2

        ref_mask = np.asarray(ref["mask"], np.uint8)
        if ref_mask.shape != ref["depth"].shape:
            ref_mask = cv2.resize(
                ref_mask, (ref["depth"].shape[1], ref["depth"].shape[0]), interpolation=cv2.INTER_NEAREST
            )
        points = surface_constraint.depth_points(ref["depth"], ref_mask, intr_demo)
        click = acquisition._deepest_pixel(mask, rgb.shape)
        if click is None and predicted_fix is not None and len(points):
            base_points = points @ t_bc[:3, :3].T + t_bc[:3, 3]
            motion = tip @ np.linalg.inv(predicted_fix) @ np.linalg.inv(demo.tips[ref["sample"]])
            with runtime._state.lock:
                tolerance = runtime._state.depth_tol_m
            click = holds._held_click(base_points, motion, t_bc, intr, depth, tolerance)
        if click is None:
            attempts.append({"reference_frame": ref["frame"], "reason": "no supported held-object click"})
            continue
        found = await acquisition._locate(
            obj,
            rgb,
            depth,
            intr,
            click,
            cancelled,
            reference=ref,
            log={
                "why": "act: direct held-view comparison",
                "came_from": {"held_reference_frame": ref["frame"]},
            },
        )
        check = found.get("surface_constraint") or {}
        attempts.append(
            {
                "reference_frame": ref["frame"],
                "find_id": found.get("find_id"),
                "reason": found.get("reason") or acquisition._weak(found),
                "constraint": check,
            }
        )
        if found.get("reference_mask") is not None:
            ref["mask"], ref["refresh_mask"] = found["reference_mask"], False
        if cancelled():
            break
        if not found["ok"] or acquisition._weak(found) or check.get("pose") is None:
            continue
        # Both points describe the SAME reference material patch, expressed in
        # the demo and current fingertip frames. Neither uses its original pose.
        points = surface_constraint.depth_points(ref["depth"], ref["mask"], intr_demo)
        if not len(points):
            continue
        desired = np.linalg.inv(demo.tips[ref["sample"]]) @ t_bc
        observed = np.linalg.inv(tip) @ t_bc @ np.asarray(found["delta"])
        centre = np.r_[points.mean(axis=0), 1.0]
        observations.append(
            {
                "frame": ref["frame"],
                "desired": (desired @ centre)[:3],
                "observed": (observed @ centre)[:3],
                "rotation": desired[:3, :3] @ observed[:3, :3].T,
            }
        )
        result = consensus(observations, predicted_fix)
        # Sample different references and at least the normal hold view count;
        # an early three-view cluster must not win before the alternatives run.
        if result is not None and len(observations) >= core.HOLD_VIEWS:
            return result, "", attempts
    result = consensus(observations, predicted_fix)
    problem = (
        ""
        if result is not None
        else f"held-view comparison: {len(observations)} accepted views did not establish a current attachment"
    )
    return result, "stopped" if stopped() else problem, attempts
