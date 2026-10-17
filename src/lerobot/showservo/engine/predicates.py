# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""What a check asks of the camera: which of two things an object moved with, and whether it is where a pose puts it
(act_execution.md, "What a check answers, and how close counts").

Every answer is success (YES), failure (NO) or can't tell (UNKNOWN), with the evidence. No threshold here is a number
fitted to a rig: each comes from the noise measured on the object in the same act, an allowance the operator sets, or
the object's own shape. Two counts are choices, named where they are used: three points for a median, and a majority
of the points tested.
"""

from __future__ import annotations

import dataclasses
import enum
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import Camera, Depth, Tracks


class Answer(enum.Enum):
    YES = "success"
    NO = "failure"
    UNKNOWN = "can't tell"


@dataclasses.dataclass(frozen=True)
class Verdict:
    answer: Answer
    why: str
    evidence: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def yes(self) -> bool:
        return self.answer is Answer.YES

    @property
    def no(self) -> bool:
        return self.answer is Answer.NO


def _unknown(why: str, **evidence: Any) -> Verdict:
    return Verdict(Answer.UNKNOWN, why, evidence)


MEDIAN_MIN = 3  # a choice: the fewest values whose median ignores one wrong one


def spread(values: np.ndarray) -> float | None:
    """How far a set of measurements of one still thing wanders: the 95th percentile of their distances from their
    median, None with fewer than MEDIAN_MIN of them."""
    v = np.asarray(values, dtype=float)
    v = v.reshape(len(v), -1)
    if len(v) < MEDIAN_MIN:
        return None
    return float(np.percentile(np.linalg.norm(v - np.median(v, axis=0), axis=1), 95))


def spread_deg(rotations: np.ndarray) -> float | None:
    """The same for orientations (N, 3, 3): the 95th percentile of their angles from their mean, in degrees."""
    r = np.asarray(rotations, dtype=float)
    if len(r) < MEDIAN_MIN:
        return None
    rot = Rotation.from_matrix(r)
    return float(np.percentile(np.degrees((rot * rot.mean().inv()).magnitude()), 95))


def decide(obs: np.ndarray, new: np.ndarray, old: np.ndarray, noise: float, unit: str) -> Verdict:
    """Which of two predictions an observation is: ``new`` (success) or ``old`` (failure), in the same units as the
    ``noise`` measured on what is observed. Undecided until the predictions are more than twice the noise apart;
    then success within half their distance less the noise of ``new``, failure within that of ``old`` or beyond half
    their distance plus the noise from both (something else moved it)."""
    obs, new, old = (np.asarray(x, dtype=float) for x in (obs, new, old))
    apart = float(np.linalg.norm(new - old))
    to_new, to_old = float(np.linalg.norm(obs - new)), float(np.linalg.norm(obs - old))
    ev = {"apart": apart, "to_new": to_new, "to_old": to_old, "noise": noise, "unit": unit}
    if apart <= 2.0 * noise:
        return _unknown(
            f"the two outcomes are {apart:.1f} {unit} apart, within the noise ({noise:.1f})", **ev
        )
    near = apart / 2.0 - noise
    if to_new <= near:
        return Verdict(
            Answer.YES, f"as expected: {to_new:.1f} {unit} from it, {apart:.1f} from the other", ev
        )
    if to_old <= near:
        return Verdict(
            Answer.NO, f"the other outcome: {to_old:.1f} {unit} from it, {apart:.1f} from the expected", ev
        )
    if to_new > apart / 2.0 + noise and to_old > apart / 2.0 + noise:
        return Verdict(
            Answer.NO, f"neither: {to_new:.1f} {unit} from the expected, {to_old:.1f} from the other", ev
        )
    return _unknown(
        f"between the two: {to_new:.1f} {unit} from the expected, {to_old:.1f} from the other", **ev
    )


def moves_with(
    ref: Tracks, now: Tracks, shift_new: np.ndarray, shift_old: np.ndarray, noise_px: float
) -> Verdict:
    """Did the object's tracked points move, from frame ``ref`` to ``now``, as the expected thing moved (``shift_new``,
    its predicted image displacement) or as the other did (``shift_old``)? Their median displacement, over the points
    seen in both frames, decided by :func:`decide` against ``noise_px``, how far that median wandered while the object
    stood untouched."""
    before = {int(i): xy for i, xy, seen in zip(ref.ids, ref.uv, ref.visible, strict=True) if seen}
    pairs = [
        (before[int(i)], xy)
        for i, xy, seen in zip(now.ids, now.uv, now.visible, strict=True)
        if seen and int(i) in before
    ]
    if len(pairs) < MEDIAN_MIN:
        return _unknown(f"{len(pairs)} tracked points seen in both frames", points=len(pairs))
    shift = np.median(np.array([b - a for a, b in pairs]), axis=0)
    v = decide(shift, shift_new, shift_old, noise_px, "px")
    return dataclasses.replace(v, evidence={**v.evidence, "points": len(pairs), "shift_px": shift.tolist()})


def thickness(surface: np.ndarray) -> float:
    """How thick the object is in its thinnest direction: the 5th to 95th percentile spread of its surface points
    along their smallest principal axis."""
    pts = np.asarray(surface, dtype=float)
    centred = pts - pts.mean(axis=0)
    along = centred @ np.linalg.svd(centred, full_matrices=False)[2][-1]
    return float(np.percentile(along, 95) - np.percentile(along, 5))


def _disk(radius_px: float, step: int = 2) -> np.ndarray:
    """Pixel offsets within ``radius_px``, every ``step`` pixels (the object spans many pixels; a coarser grid misses
    no surface), the centre first."""
    r = int(np.ceil(radius_px))
    du, dv = np.meshgrid(np.arange(-r, r + 1, step), np.arange(-r, r + 1, step))
    off = np.stack([du.ravel(), dv.ravel()], axis=1)
    off = off[np.hypot(off[:, 0], off[:, 1]) <= radius_px + 1e-9]
    return off[np.argsort(np.hypot(off[:, 0], off[:, 1]), kind="stable")]


def sits_at(
    surface: np.ndarray,
    motion: np.ndarray,
    depth: Depth,
    camera: Camera,
    depth_noise_m: float,
    lateral_m: float = 0.0,
    thickness_m: float | None = None,
) -> Verdict:
    """Is the object where ``motion`` puts it? Its surface points (base frame, at its demo pose) moved by ``motion``,
    the nearest per pixel, against the depth image. A point is present when the depth within ``lateral_m`` of it
    (the allowance for the pose's own error) reads its depth to within the depth noise plus that allowance; absent
    when its pixel reads farther (the camera sees past it); covered when nearer. The depth tolerance never reaches
    half the object's thickness, so what it rests on cannot pass for it. Decided by a majority of the points (a
    choice), at least MEDIAN_MIN of them."""
    thick = thickness(surface) if thickness_m is None else thickness_m
    if depth_noise_m >= thick / 2.0:
        return _unknown(
            f"the depth wanders {depth_noise_m * 1000:.1f} mm, half the object's thickness or more"
        )
    tol = min(depth_noise_m + lateral_m, thick / 2.0)
    m = np.asarray(motion, dtype=float)
    pts = np.asarray(surface, dtype=float) @ m[:3, :3].T + m[:3, 3]
    uv, z = camera.project(pts)
    h, w = depth.metres.shape
    ui, vi = np.round(uv[:, 0]), np.round(uv[:, 1])
    inside = (z > 0) & np.isfinite(ui) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[inside].astype(int), vi[inside].astype(int), z[inside]
    pixel = vi * w + ui
    order = np.lexsort((z, pixel))  # by pixel, nearest first: the point the camera would see there
    first = np.ones(len(order), dtype=bool)
    first[1:] = pixel[order][1:] != pixel[order][:-1]
    keep = order[first]
    ui, vi, z = ui[keep], vi[keep], z[keep]
    offsets = (
        _disk(lateral_m * camera.K[0, 0] / max(float(z.min()), 1e-6))
        if len(z) and lateral_m > 0
        else np.zeros((1, 2), int)
    )
    near = depth.metres[
        np.clip(vi[:, None] + offsets[None, :, 1], 0, h - 1),
        np.clip(ui[:, None] + offsets[None, :, 0], 0, w - 1),
    ]
    present_mask = np.any((near > 0) & (np.abs(near - z[:, None]) <= tol), axis=1)
    centre = depth.metres[vi, ui]
    absent_mask = ~present_mask & (centre > z + tol)
    covered_mask = ~present_mask & (centre > 0) & (centre < z - tol)
    present, absent, covered = int(present_mask.sum()), int(absent_mask.sum()), int(covered_mask.sum())
    tested = present + absent + covered
    ev = {"present": present, "absent": absent, "covered": covered, "tol_mm": tol * 1000.0}
    if tested < MEDIAN_MIN:
        return _unknown(f"{tested} of its points in view with a depth reading", **ev)
    if present * 2 > tested:
        return Verdict(Answer.YES, f"there: {present} of {tested} points", ev)
    if absent * 2 > tested:
        return Verdict(Answer.NO, f"not there: the camera sees past {absent} of {tested} points", ev)
    return _unknown(f"{present} present, {absent} absent, {covered} covered", **ev)


def _turn_deg(rot: Rotation, free_axis: np.ndarray | None, turns: tuple[np.ndarray, int] | None) -> float:
    """The angle of ``rot`` less what a declared symmetry makes the same: any turn about ``free_axis`` (a continuous
    axis, the swing of a swing-twist split), or the nearest of ``turns`` = (axis, order) repeating turns."""
    if turns is not None:
        axis, order = turns
        a = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        return min(
            float(np.degrees((Rotation.from_rotvec(a * 2.0 * np.pi * k / order).inv() * rot).magnitude()))
            for k in range(order)
        )
    if free_axis is None:
        return float(np.degrees(rot.magnitude()))
    a = np.asarray(free_axis, dtype=float) / np.linalg.norm(free_axis)
    x, y, z, w = rot.as_quat()
    twist = np.array([*((np.array([x, y, z]) @ a) * a), w])
    if np.linalg.norm(twist) < 1e-12:
        return float(np.degrees(rot.magnitude()))
    return float(np.degrees((rot * Rotation.from_quat(twist / np.linalg.norm(twist)).inv()).magnitude()))


def related(
    obj_motion: np.ndarray,
    ref_motion: np.ndarray,
    rel: np.ndarray,
    centre: np.ndarray,
    noise_m: float,
    noise_deg: float,
    allow_m: float = 0.0,
    allow_deg: float = 0.0,
    free_axis: np.ndarray | None = None,
    turns: tuple[np.ndarray, int] | None = None,
) -> Verdict:
    """Is the object, by a view of it, where its reference's motion times ``rel`` puts it? Compared at its centre
    (base frame, at its demo pose), and in rotation less any turn its declared symmetry makes the same. Success within
    the operator's allowance plus the views' measured noise; failure beyond that by more than the noise again."""
    expected = np.asarray(ref_motion, dtype=float) @ np.asarray(rel, dtype=float)
    observed = np.asarray(obj_motion, dtype=float)
    c = np.append(np.asarray(centre, dtype=float)[:3], 1.0)
    off_m = float(np.linalg.norm((expected @ c)[:3] - (observed @ c)[:3]))
    off_deg = _turn_deg(Rotation.from_matrix(expected[:3, :3].T @ observed[:3, :3]), free_axis, turns)
    tol_m, tol_deg = allow_m + noise_m, allow_deg + noise_deg
    ev = {"off_mm": off_m * 1000.0, "off_deg": off_deg, "tol_mm": tol_m * 1000.0, "tol_deg": tol_deg}
    if off_m <= tol_m and off_deg <= tol_deg:
        return Verdict(Answer.YES, f"where expected, {off_m * 1000:.0f} mm and {off_deg:.0f} deg off", ev)
    if off_m > tol_m + noise_m or off_deg > tol_deg + noise_deg:
        return Verdict(Answer.NO, f"not where expected: {off_m * 1000:.0f} mm and {off_deg:.0f} deg off", ev)
    return _unknown(
        f"{off_m * 1000:.0f} mm and {off_deg:.0f} deg off, within the views' noise of the limit", **ev
    )


def combine(verdicts: list[Verdict]) -> Verdict:
    """One answer from several pieces of evidence: a success or a failure stands when nothing contradicts it; both at
    once can't tell, and a later frame decides."""
    yes = [v for v in verdicts if v.yes]
    no = [v for v in verdicts if v.no]
    sources = [{"answer": v.answer.value, "why": v.why} for v in verdicts]
    if yes and no:
        return _unknown(f"the evidence disagrees: {yes[0].why}; {no[0].why}", sources=sources)
    if yes:
        return Verdict(Answer.YES, yes[0].why, {"sources": sources})
    if no:
        return Verdict(Answer.NO, no[0].why, {"sources": sources})
    return _unknown("; ".join(v.why for v in verdicts) or "no evidence", sources=sources)
