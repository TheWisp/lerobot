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

"""The GUI's tracking worker. It hosts PatchFit, the tracker built here (SAM3 by concept, DINO patch
features matched frame by frame, a 3D rigid fit, a growing card of the object's points), and the
bridge to Point2Pose, the tracker the live loop uses by default.

Runs as its own process because the models need the GPU and must never load
inside the GUI. It long-polls the GUI for jobs, fetches the job's frame, and
posts a result. Two kinds of job:

- ``teach``: designate the object by concept, compile a card (descriptors +
  camera-frame 3D on the mask) and keep it here.
- ``find``: designate again, match against the card, and fit the object's rigid
  motion (teach -> now) in the camera frame with a certificate.

The same code the first-contact bench measured with; only the transport of
frames and results differs.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m0 import DinoTier, Sam3Concept  # noqa: E402
from showservo_real import MIN_INLIERS, Card  # noqa: E402

from lerobot.gui.api import _pregrasp_core as core  # noqa: E402
from lerobot.showservo.pose import CameraIntrinsics, ransac_fit_rigid, sample_depth  # noqa: E402
from lerobot.showservo.tracker import KLTTracker  # noqa: E402


class _Frame:
    """What :class:`Card` and :func:`bind_rigid3d` need of a scene. ``t`` is the frame's time in
    seconds (a recording's own clock, or the wall clock when it arrives live): the tracker's
    motion bounds and its growth cadence are rates, so they run on the frames' clock, not the
    machine's."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, name: str, t: float | None = None):
        self.rgb = np.ascontiguousarray(rgb)
        self.depth = depth
        self.name = name
        self.t = time.time() if t is None else float(t)


class Models:
    def __init__(self, device: str, dino_model: str, resolution: int):
        self.device, self.dino_model, self.resolution = device, dino_model, resolution
        self.sam: Sam3Concept | None = None
        self.tier: DinoTier | None = None
        # One Point2Pose process per mode for the worker's life; a teach anchors the selected one.
        self.p2p: dict[str, P2PBridge] = {}
        self.p2p_error: dict[str, str] = {}

    def p2p_bridge(self, mode: str = "p2p", key: str | None = None):
        """The Point2Pose process for ``mode`` ("p2p" or "p2p_dense"), started on first use; None when
        it cannot start (its environment is absent), with the reason kept for the readout. ``key``
        names a separate process with the same configuration: the recorded-stream tracking runs in
        its own, so a live track is never re-anchored by it."""
        mode = mode if mode in P2P_CONFIGS else "p2p"
        key = key or mode
        if key not in self.p2p and key not in self.p2p_error:
            try:
                self.p2p[key] = P2PBridge(P2P_CONFIGS[mode])
            except Exception as e:  # the worker goes on without it
                self.p2p_error[key] = f"{type(e).__name__}: {e}"
                print(f"Point2Pose ({key}) unavailable: {self.p2p_error[key]}", flush=True)
        return self.p2p.get(key)

    def drop_bridge(self, key: str) -> None:
        """Close one Point2Pose process and forget it, freeing its GPU memory; the next use starts a fresh one."""
        bridge = self.p2p.pop(key, None)
        self.p2p_error.pop(key, None)
        if bridge is not None:
            bridge.close()

    def ensure(self, concept: str) -> tuple[Sam3Concept, DinoTier]:
        if self.tier is None:
            print(f"loading {self.dino_model} on {self.device}", flush=True)
            self.tier = DinoTier(self.dino_model, device=self.device)
        if self.sam is None:
            print(f"loading SAM3 on {self.device} for {concept!r}", flush=True)
            self.sam = Sam3Concept(concept, device=self.device, resolution=self.resolution)
        elif self.sam.concept != concept:
            self.sam.adapter.set_control({"prompt": concept})
            self.sam.concept = concept
            self.sam.label = f"SAM3 {concept!r}"
        return self.sam, self.tier


FACE_INLIER_M = 0.0025  # a depth point this close to the plane lies on it (the sensor's own noise band)
FACE_TRIALS = 200
FACE_MIN_INLIERS = 30


def _consensus_plane(pts: np.ndarray, rng: np.random.Generator, trials: int) -> np.ndarray | None:
    """Inlier mask of the plane through the most points, from random triplets; None when none holds enough."""
    n_pts = len(pts)
    if n_pts < FACE_MIN_INLIERS:
        return None
    idx = rng.choice(n_pts, size=(trials, 3), replace=True)
    p0, p1, p2 = pts[idx[:, 0]], pts[idx[:, 1]], pts[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    lengths = np.linalg.norm(normals, axis=1)
    good = lengths > 1e-9
    if not good.any():
        return None
    normals = normals[good] / lengths[good, None]
    p0 = p0[good]
    dist = np.abs(np.einsum("tj,tnj->tn", normals, pts[None, :, :] - p0[:, None, :]))
    counts = (dist < FACE_INLIER_M).sum(axis=1)
    best = int(np.argmax(counts))
    if counts[best] < FACE_MIN_INLIERS:
        return None
    inl = dist[best] < FACE_INLIER_M
    # Two least-squares refits on the consensus set tighten the normal and recount.
    for _ in range(2):
        c = pts[inl].mean(axis=0)
        _u, _s, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
        inl = np.abs((pts - c) @ vt[2]) < FACE_INLIER_M
        if inl.sum() < FACE_MIN_INLIERS:
            return None
    return inl


def face_plane(depth: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics) -> dict | None:
    """:func:`_face_fit` without the points, for a result's metadata."""
    return _face_fit(depth, mask, intr)[0]


def _face_fit(
    depth: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics
) -> tuple[dict | None, np.ndarray | None]:
    """The face the camera sees: the plane holding the most of the designated depth cloud, and its points.

    A mask usually covers more than one face (a cube from above shows its top and
    a side), so the plane is found by consensus, not by fitting the whole cloud.
    Post: unit normal facing the camera, centroid, ``planarity`` = the fraction of
    the cloud on that plane, ``n_plane`` its point count, and ``dominance`` = its
    count over the next-largest plane's, which says whether one face dominates or
    two compete; None when the mask has too little depth. Hundreds of points pin
    this normal to about a degree, where a handful of matched features cannot.
    """
    vs, us = np.nonzero(mask & (depth > 0))
    if len(us) < 2 * FACE_MIN_INLIERS:
        return None, None
    pts = intr.deproject(np.stack([us, vs], axis=1), depth[vs, us])
    rng = np.random.default_rng(0)  # deterministic: the same cloud gives the same face
    inl = _consensus_plane(pts, rng, FACE_TRIALS)
    if inl is None:
        return None, None
    c = pts[inl].mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    second = _consensus_plane(pts[~inl], rng, FACE_TRIALS // 2)
    n_second = 0 if second is None else int(second.sum())
    info = {
        "normal": n.tolist(),
        "centroid": c.tolist(),
        "planarity": float(inl.mean()),
        "n": int(len(us)),
        "n_plane": int(inl.sum()),
        "dominance": float(inl.sum() / max(n_second, 1)),
    }
    return info, pts[inl]


MATCH_MIN_SIM = 0.6  # a live patch below this cosine similarity to its nearest card point is not a match


def _match(card_desc: np.ndarray, desc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each live patch to its nearest card point, one live patch per card point (the most similar wins).

    Mutual nearest neighbours lose most matches on a card that holds several
    looks of the same spot, since the reverse lookup lands on a sibling (2026-10-02:
    41 matches against 119 on the first frame at a new spot). Geometry, the rigid
    fit, is what rejects a wrong pair. Post: index arrays (M,), (M,) into the card
    and the live patches.
    """
    if len(desc) == 0 or len(card_desc) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    sim = desc @ card_desc.T
    nearest = sim.argmax(axis=1)
    best = sim[np.arange(len(desc)), nearest]
    ia, ib = [], []
    taken = np.zeros(len(card_desc), dtype=bool)
    for j in np.argsort(-best):
        k = nearest[j]
        if best[j] < MATCH_MIN_SIM:
            break
        if not taken[k]:
            taken[k] = True
            ia.append(k)
            ib.append(j)
    return np.asarray(ia, dtype=int), np.asarray(ib, dtype=int)


SCALE_NOISE_M = 0.002  # the depth sensor's noise, which the fitted scale cannot tell from a size change
SCALE_TOL_MIN = (
    0.1  # the scale certificate's tolerance for a large object; a small one gets the noise over its radius
)


# How far the object may have moved since a certified frame: a floor for the fit's own jitter plus
# a rate for the time elapsed, so a long occlusion widens the bound until it means nothing (at 2.5 s
# the rotation bound reaches 180 degrees and the reference is dropped). Every certified frame of the
# last seconds is a reference: candidates beyond any of their bounds are never selected. Measured on
# YCBInEOAT mustard0 (a bottle lifted, turned and set upright by a gripper): without a bound the fit
# flipped to the mirrored pose, 160 to 178 degrees off, on a hundred matches; a 45-degree bound
# against the last frame alone was walked round in four certified steps; against every recent frame
# at 30 degrees plus 60 degrees a second no frame of 737 flipped, with 91 held as occluded instead.
STEP_ROT_MAX_DEG = 30.0
STEP_ROT_RATE_DEG_S = 60.0
STEP_TRANS_MAX_M = 0.15
STEP_TRANS_RATE_M_S = 0.5


def _turn_deg(rot: np.ndarray, ref: np.ndarray | None) -> float:
    """The angle between two rotations; infinite when there is no reference yet."""
    if ref is None:
        return float("inf")
    return float(np.degrees(np.arccos(np.clip((np.trace(rot @ ref.T) - 1.0) / 2.0, -1.0, 1.0))))


def motion_bound(elapsed_s: float) -> tuple[float, float]:
    """(rotation deg, translation m) a rigid body is allowed since a certified frame ``elapsed_s`` ago."""
    return (
        min(180.0, STEP_ROT_MAX_DEG + STEP_ROT_RATE_DEG_S * max(elapsed_s, 0.0)),
        STEP_TRANS_MAX_M + STEP_TRANS_RATE_M_S * max(elapsed_s, 0.0),
    )


def _bind(
    card: Card,
    frame: _Frame,
    region: np.ndarray,
    tier: DinoTier,
    intr: CameraIntrinsics,
    priors=None,
):
    """Match the card inside ``region`` and fit the rigid motion teach -> now.

    Post: ``(fit, live_uv, card_idx, patches)`` with ``fit`` None when nothing
    certified; ``live_uv`` are the matched pixels with depth and ``card_idx``
    their card points, both over the same rows, so ``fit.inliers`` indexes
    either; ``patches`` is every extracted ``(uv, desc)`` in the region, for
    growing the card. ``priors`` are the recent certified motions with how far from
    each a candidate may lie (see :func:`motion_bound`).
    """
    uv, desc = tier.teach(frame.rgb, region)
    desc = np.asarray(desc, dtype=np.float32)
    patches = (uv, desc)
    ia, ib = _match(card.desc, desc)
    if len(ia) < MIN_INLIERS:
        return None, np.zeros((0, 2)), None, patches
    z, ok = sample_depth(frame.depth, uv[ib])
    live, idx = uv[ib][ok], ia[ok]
    inlier_m = float(np.clip(0.15 * card.radius, 0.003, 0.010))
    fit = ransac_fit_rigid(
        card.xyz[idx],
        intr.deproject(live, z[ok]),
        inlier_m=inlier_m,
        hypo_weights=card.hypo_w[idx],
        priors=priors,
    )
    scale_tol = max(SCALE_TOL_MIN, SCALE_NOISE_M / max(card.radius, 1e-3))
    if not (fit.ok and fit.n_inliers >= MIN_INLIERS and fit.scale_is_plausible(scale_tol)):
        return None, live, None, patches
    return fit, live, idx, patches


# The card grows: a certified frame adds the patches it shows that the card does not hold yet, in the
# object's own frame, so the object is known from every side it has been seen from.
CARD_GROW_MIN_INLIERS = 24  # fewer and the frame's pose is too uncertain to anchor new points to
CARD_GROW_MIN_RATIO = (
    0.5  # inliers over matches: a frame whose matches mostly disagreed is not trusted to grow
)
CARD_VOXEL_M = 0.003  # a new point closer than this to a card point is the same surface, already held
CARD_GROW_RADIUS = 1.2  # new points lie within this times the card's radius of its centre: on the object
CARD_GROW_HEIGHT_M = 0.003  # and this far above the resting surface: not the table around it
CARD_MAX_POINTS = 1500  # the oldest additions go first; the teach view's own points stay
CARD_SAME_LOOK = 0.85  # a fresh descriptor this similar to the one held at that spot is the same look
CARD_LOOKS_PER_SPOT = 3  # how many looks of one spot the card keeps (viewpoints, lighting)
# Growth rebuilds the hypothesis ballot, so not on every frame; but an object being turned shows a
# new side quickly: on a mustard bottle lifted and turned by a gripper (YCBInEOAT mustard0), growth
# once a second of video left the card at 743 points and the fit on 9 to 13 inliers, while growth
# every few frames carried it to 1500 points and held. Both run on the frames' clock, so a recording
# replays the same way at any speed. The object having turned this far since the last growth is one
# trigger, the time since the last growth the other.
GROW_EVERY_S = 0.3
CARD_GROW_TURN_DEG = 10.0


def _rebuild_ballot(card: Card) -> None:
    """Recompute the card's hypothesis ballot (which points may propose a pose) after its points changed."""
    sim = card.desc @ card.desc.T
    gap = np.linalg.norm(card.xyz[:, None, :] - card.xyz[None, :, :], axis=2)
    sim[gap < 0.4 * card.radius] = -1.0
    card.distinct = np.clip(1.0 - sim.max(axis=1), 0.0, 1.0)
    k = max(int(round(0.1 * len(card.distinct))), 12)
    if len(card.distinct) >= 2 * k:
        card.hypo_w = np.zeros(len(card.distinct))
        card.hypo_w[np.argsort(card.distinct)[-k:]] = 1.0
    else:
        card.hypo_w = np.ones(len(card.distinct))


def _grow_card(card: Card, frame: _Frame, fit, patches, intr: CameraIntrinsics, surface) -> int:
    """Add what a certified frame shows and the card lacks. Post: the number of points added."""
    from scipy.spatial import cKDTree

    uv, desc = patches
    z, ok = sample_depth(frame.depth, uv)
    if not ok.any():
        return 0
    pts, desc = intr.deproject(uv[ok], z[ok]), desc[ok]
    rot, trans = fit.transform.rot, fit.transform.trans
    in_card = (pts - trans) @ rot  # the frame's points where the card's frame would see them
    centre = card.xyz[: card.n_teach].mean(axis=0)
    keep = np.linalg.norm(in_card - centre, axis=1) <= CARD_GROW_RADIUS * card.radius
    if surface is not None:
        n, c_s = surface
        keep &= (pts - c_s) @ n > CARD_GROW_HEIGHT_M
    if not keep.any():
        return 0
    cand, cand_desc = in_card[keep], desc[keep]
    far, near = cKDTree(card.xyz).query(cand)
    new = far > CARD_VOXEL_M
    # A known spot whose look has changed keeps the new look too, up to a few per spot: the same
    # surface from another angle or under other light matches again next time.
    same_spot = ~new
    changed = same_spot & (np.einsum("ij,ij->i", cand_desc, card.desc[near]) < CARD_SAME_LOOK)
    if changed.any():
        looks = getattr(card, "looks", None)
        if looks is None:
            looks = card.looks = {}
        vox_keys = [tuple(v) for v in np.floor(cand / CARD_VOXEL_M).astype(int)]
        for i in np.flatnonzero(changed):
            if looks.get(vox_keys[i], 1) >= CARD_LOOKS_PER_SPOT:
                changed[i] = False
            else:
                looks[vox_keys[i]] = looks.get(vox_keys[i], 1) + 1
    take = new | changed
    if not take.any():
        return 0
    cand, cand_desc = cand[take], cand_desc[take]
    if new[take].any():
        fresh = new[take]
        _vox, first = np.unique(np.floor(cand[fresh] / CARD_VOXEL_M).astype(int), axis=0, return_index=True)
        keep_rows = np.concatenate([np.flatnonzero(fresh)[np.sort(first)], np.flatnonzero(~fresh)])
        cand, cand_desc = cand[np.sort(keep_rows)], cand_desc[np.sort(keep_rows)]
    card.xyz = np.vstack([card.xyz, cand])
    card.desc = np.vstack([card.desc, cand_desc.astype(np.float32)])
    card.uv = np.vstack([card.uv, intr.project(cand)])
    if len(card.xyz) > CARD_MAX_POINTS:
        keep_n = CARD_MAX_POINTS - card.n_teach
        card.xyz = np.vstack([card.xyz[: card.n_teach], card.xyz[-keep_n:]])
        card.desc = np.vstack([card.desc[: card.n_teach], card.desc[-keep_n:]])
        card.uv = np.vstack([card.uv[: card.n_teach], card.uv[-keep_n:]])
    _rebuild_ballot(card)
    return int(len(cand))


FOOTPRINT_PAD_PX = (
    12  # the blob search reaches this far past the designation, whose edge pixels the mask may miss
)


def _intr_dict(intr: CameraIntrinsics) -> dict[str, float]:
    return {"fx": intr.fx, "fy": intr.fy, "cx": intr.cx, "cy": intr.cy}


def _table_fit(depth: np.ndarray, region: np.ndarray, intr: CameraIntrinsics):
    """The surface the object rests on, fitted to the depth in a ring around ``region``: ``(unit normal, point)``
    in the camera frame, or None when the ring has too little depth."""
    vs, us = np.nonzero(region)
    if len(us) == 0:
        return None
    box = (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1)
    try:
        return core.table_plane(depth, _intr_dict(intr), box)
    except ValueError:
        return None


def _footprint(depth: np.ndarray, region: np.ndarray, plane, intr: CameraIntrinsics) -> dict | None:
    """The above-surface blob that ``region`` overlaps most, with its footprint in the surface; None when nothing stands there."""
    import cv2

    k = 2 * FOOTPRINT_PAD_PX + 1
    search = cv2.dilate(region.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
    standing = core.above_table(depth, _intr_dict(intr), plane, search)
    n_lab, labels, _s, _c = cv2.connectedComponentsWithStats(standing.astype(np.uint8), connectivity=8)
    best, best_overlap = None, 0
    for lab in range(1, n_lab):
        overlap = int(np.count_nonzero(region & (labels == lab)))
        if overlap > best_overlap:
            best, best_overlap = lab, overlap
    if best is None:
        return None
    try:
        return core.shape_stats(depth, _intr_dict(intr), plane, labels == best)
    except ValueError:
        return None


def _face_outline(face_pts: np.ndarray | None, surface) -> np.ndarray | None:
    """The face's points projected into the resting surface, about their own centre: a clean outline.

    The whole above-surface blob also holds side faces seen at a grazing angle,
    whose depth is noisy and which smear the outline (2026-10-02: a resting
    cube's turn wandered 30 degrees between frames); the face's own points do not.
    """
    if face_pts is None or len(face_pts) < FACE_MIN_INLIERS:
        return None
    xy = core.footprint(face_pts, surface, face_pts.mean(axis=0))
    return xy - xy.mean(axis=0)


def _geometry(
    card: Card | None, depth: np.ndarray, region: np.ndarray, intr: CameraIntrinsics, prefer_deg=None
) -> dict:
    """What the depth says about the object under ``region``, the same way at teach, find and track.

    The resting surface is fitted around the region; the blob standing on it
    that the region overlaps most is the object; the face is fitted on that
    blob, not on the designation, so the two never disagree on what the object
    is; the turn comes from the face's outline when the card has one. Post: a
    dict with ``surface`` and ``blob`` (worker-side) and the result keys
    ``table_find``, ``face_find`` and the ``footprint_*`` turn, plus ``face_pts``.
    """
    surface = _table_fit(depth, region, intr)
    out: dict = {"surface": surface, "blob": None, "face_find": None, "face_pts": None}
    out["table_find"] = None if surface is None else [float(v) for v in surface[0]]
    if surface is None:
        return out
    blob = _footprint(depth, region, surface, intr)
    out["blob"] = blob
    face_region = blob["mask"] if blob is not None else region
    face_info, face_pts = _face_fit(depth, face_region, intr)
    out["face_find"], out["face_pts"] = face_info, face_pts
    if card is not None:
        out.update(_turn_from_footprint(card, depth, region, intr, surface, face_pts, prefer_deg, blob))
    return out


def _turn_from_footprint(
    card: Card,
    depth: np.ndarray,
    region: np.ndarray,
    intr: CameraIntrinsics,
    surface=None,
    face_pts=None,
    prefer_deg: float | None = None,
    blob=None,
) -> dict:
    """The surface under ``region`` and the object's turn teach -> now from its geometry, as result keys.

    The outline of the face toward the camera carries the turn when both frames
    have one; the whole above-surface blob's footprint otherwise. Geometry
    carries the turn of a plain object whose texture cannot, ambiguous only by
    the object's own symmetry.
    """
    fit_t = surface if surface is not None else _table_fit(depth, region, intr)
    out: dict = {"table_find": None if fit_t is None else [float(v) for v in fit_t[0]]}
    if fit_t is None:
        return out
    taught_face = getattr(card, "face_xy", None)
    found_face = _face_outline(face_pts, fit_t)
    if taught_face is not None and found_face is not None:
        fy = core.footprint_yaw(taught_face, found_face, prefer_deg)
        out.update(
            footprint_yaw_deg=float(fy["yaw_deg"]),
            footprint_iou=float(fy["iou"]),
            footprint_symmetric=bool(fy["symmetric"]),
            footprint_points=int(len(found_face)),
            footprint_from="face outline",
        )
        return out
    shape = getattr(card, "shape", None)
    if shape is None:
        return out
    found = blob if blob is not None else _footprint(depth, region, fit_t, intr)
    if found is None:
        return out
    fy = core.footprint_yaw(shape["footprint"], found["footprint"], prefer_deg)
    out.update(
        footprint_yaw_deg=float(fy["yaw_deg"]),
        footprint_iou=float(fy["iou"]),
        footprint_symmetric=bool(fy["symmetric"]),
        footprint_points=int(found["n_points"]),
        footprint_from="blob",
    )
    return out


def _rect_mask(uv: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The axis-aligned bounding box of pixels ``uv`` as a boolean image, clipped to the frame."""
    pts = np.asarray(uv)
    x0, y0 = np.clip(np.floor(pts.min(axis=0)).astype(int), 0, [shape[1] - 1, shape[0] - 1])
    x1, y1 = np.clip(np.ceil(pts.max(axis=0)).astype(int) + 1, 1, [shape[1], shape[0]])
    m = np.zeros(shape, dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _acquire_region(card: Card, mask: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """The region to extract descriptors from when the object was designated: the teach view's
    extent, scaled by the designation's depth, centred on it.

    The extractor crops to the region's bounding box and resizes the crop to a fixed size, so the
    patch scale follows the box. A designation that shows only part of the object (a hand or a
    gripper over the rest) shrinks the box and shifts the scale, and the card no longer matches:
    measured on a soup can in a robot hand, a half-covered designation matched nothing at all
    against a card taught from the whole can. The object's size is known from the teach, so the
    box is sized from it and only placed by the designation.
    """
    ys, xs = np.nonzero(mask)
    z = depth[ys, xs]
    z = z[z > 0]
    n_teach = getattr(card, "n_teach", len(card.uv))
    teach_uv = card.uv[:n_teach]
    z_teach = float(np.median(card.xyz[:n_teach][:, 2]))
    if len(z) == 0 or z_teach <= 0:
        return mask
    scale = z_teach / float(np.median(z))
    half = 0.5 * (teach_uv.max(axis=0) - teach_uv.min(axis=0)) * scale
    centre = np.array([xs.mean(), ys.mean()])
    return _rect_mask(np.array([centre - half, centre + half]), mask.shape)


def _hull_mask(uv: np.ndarray, shape: tuple[int, int], pad_px: int) -> np.ndarray:
    """The convex hull of pixels ``uv``, grown by ``pad_px``, as a boolean image."""
    import cv2

    m = np.zeros(shape, np.uint8)
    pts = np.round(np.asarray(uv)).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, shape[0] - 1)
    if len(pts) >= 3:
        cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    else:
        m[pts[:, 1], pts[:, 0]] = 1
    if pad_px > 0:
        m = cv2.dilate(m, np.ones((2 * pad_px + 1, 2 * pad_px + 1), np.uint8))
    return m.astype(bool)


def _delta(fit) -> np.ndarray:
    d = np.eye(4)
    d[:3, :3], d[:3, 3] = fit.transform.rot, fit.transform.trans
    return d


# Whether a track's designation is carried forward from the segmenter's memory between frames, or
# the object is detected by name in every frame. Measured on YCBInEOAT: a soup can turned inside a soft
# hand is detected by name in one frame of ten and from memory in most (certified frames 172 to 816
# of 1308, ADD-S AUC 51.9 to 57.4); on the mustard bottle set upright by a gripper the memory's masks
# let the fit slide in bounded steps where fresh detection held (89.2 to 84.6). Occlusion is the case
# that matters at the bench, so memory it is; one constant flips it.
DESIGNATE_FROM_MEMORY = True
LOST_AFTER = 8  # frames without a certified fit before the object counts as lost rather than occluded
FACE_PAD_PX = 2
KLT_MIN_POINTS = 12  # fewer live tracked points than this and the KLT path re-acquires


P2P_PYTHON = os.environ.get(
    "LEROBOT_P2P_PYTHON", str(pathlib.Path.home() / ".cache/point2pose/venv/bin/python")
)
P2P_REPO = os.environ.get("LEROBOT_P2P_REPO", str(pathlib.Path.home() / ".cache/point2pose/point-to-pose"))
P2P_BRIDGE = pathlib.Path(__file__).resolve().parent / "p2p_bridge.py"
# The Point2Pose modes the menu offers, each a configuration of the same pipeline.
P2P_CONFIGS = {
    "p2p": pathlib.Path(__file__).resolve().parent / "p2p_rig.yaml",
    "p2p_dense": pathlib.Path(__file__).resolve().parent / "p2p_rig_dense.yaml",
}
P2P_READY_S = 180.0  # the first start loads SAM2 and BootsTAPIR onto the GPU


P2P_FIT_ARRAYS = ("fit_uv", "fit_inlier")
P2P_FIT_FIELDS = (
    *P2P_FIT_ARRAYS,
    "n_tracks",
    "n_model_points",
    "fit_points",
    "fit_inliers",
    "jump_guard_rejected",
    "lost_streak",
)


class P2PBridge:
    """Point2Pose in its own environment (benchmarks/p2p_bridge.py), one request in flight at a time."""

    def __init__(self, config: pathlib.Path | None = None):
        self.config = pathlib.Path(config) if config is not None else P2P_CONFIGS["p2p"]
        self.served = False
        self._start()

    def _start(self) -> None:
        log_fd, self.log = tempfile.mkstemp(prefix="p2p_bridge_", suffix=".log")
        self.proc = subprocess.Popen(
            [
                P2P_PYTHON,
                str(P2P_BRIDGE),
                "--repo",
                P2P_REPO,
                "--config",
                str(self.config),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_fd,
        )
        os.close(log_fd)  # the child holds its own copy
        print(f"Point2Pose bridge started (pid {self.proc.pid}, log {self.log})", flush=True)
        ready = self._read()
        if not (ready and json.loads(str(ready["meta"])).get("ready")):
            raise RuntimeError(f"Point2Pose bridge did not come up; see {self.log}")

    def _read_exact(self, n: int) -> bytes | None:
        chunks = []
        while n > 0:
            chunk = self.proc.stdout.read(n)
            if not chunk:
                return None
            chunks.append(chunk)
            n -= len(chunk)
        return b"".join(chunks)

    def _read(self) -> dict | None:
        head = self._read_exact(4)
        if head is None:
            return None
        body = self._read_exact(struct.unpack(">I", head)[0])
        if body is None:
            return None
        z = np.load(io.BytesIO(body), allow_pickle=False)
        return {k: z[k] for k in z.files}

    def _call(self, **arrays) -> dict:
        if self.proc.poll() is not None:
            raise RuntimeError(f"Point2Pose bridge exited with {self.proc.returncode}; see {self.log}")
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        data = buf.getvalue()
        self.proc.stdin.write(struct.pack(">I", len(data)) + data)
        self.proc.stdin.flush()
        reply = self._read()
        if reply is None:
            raise RuntimeError(f"Point2Pose bridge closed the pipe; see {self.log}")
        meta = json.loads(str(reply.pop("meta")))
        return {**meta, **reply}

    def init(self, rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, intr: CameraIntrinsics) -> dict:
        """Start a pipeline on this frame. Every init after the first runs in a fresh bridge process: a pipeline
        replaced inside one process left part of its models and buffers on the GPU, every act's find added more,
        and the memory came back only when the process exited."""
        if self.served:
            self.close()
            self._start()
        self.served = True
        k = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]])
        return self._call(
            kind="init",
            rgb=rgb,
            depth=np.asarray(depth_m, dtype=np.float32),
            K=k,
            mask=np.asarray(mask, dtype=bool),
        )

    def step(self, rgb: np.ndarray, depth_m: np.ndarray) -> dict:
        return self._call(kind="step", rgb=rgb, depth=np.asarray(depth_m, dtype=np.float32))

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def anchor_p2p(bridge, card: Card, intr: CameraIntrinsics) -> bool:
    """Start Point2Pose's session on the teach frame itself, so its first pose IS the teach pose.

    Linking it later through one SAM3 + DINO acquisition anchored its whole track to that one
    fit, and on a thin self-similar object the fit's turn is a guess: a USB stick's carried cloud
    sat 30 degrees off, and a mode switch re-anchored it at the current pose as if nothing had
    turned. From the teach frame the link is the identity and nothing is guessed. Post: True when
    the session is up; False when Point2Pose declined the frame (no reply, or nothing to seed).
    """
    try:
        r = bridge.init(card.scene.rgb, card.scene.depth, card.mask, intr)
    except Exception as e:
        print(f"Point2Pose could not anchor on the teach: {type(e).__name__}: {e}", flush=True)
        return False
    return bool(r.get("ok"))


class Tracker:
    """Follows one taught card frame after frame with the chosen algorithm.

    SAM3 designates only to acquire and to re-acquire; every certified frame is
    registered against the card (teach -> now), never against the previous
    frame, so nothing accumulates. States: acquiring, tracking, occluded (a few
    frames without a certified fit, the last pose is held), lost (longer; the
    designator runs every frame until the object is back).
    """

    def __init__(self, card: Card, sam: Sam3Concept, tier: DinoTier, intr: CameraIntrinsics, p2p=None):
        self.card, self.sam, self.tier, self.intr = card, sam, tier, intr
        self.algo: str | None = None
        self.state = "acquiring"
        self.misses = 0
        self.last_fit = None
        self.last_centre = None  # the card's projected centre at the last certified frame, pixels
        self.velocity = np.zeros(2)  # its pixel motion over the last certified step
        self.klt: KLTTracker | None = None
        self.klt_idx: np.ndarray | None = None
        self.depth_teach: dict | None = None
        self.patches = None  # the last frame's extracted patches, for growing the card
        self.last_grow = -float("inf")  # frame time of the last growth
        self.last_grow_rot: np.ndarray | None = None  # the fit's rotation at the last growth
        self.last_t = -float("inf")  # frame time of the last certified fit
        self.certified: list[tuple[float, Any]] = []  # (frame time, motion) of recent certified fits
        self.last_yaw: float | None = None  # the previous footprint turn, preferred among equal peaks
        # Point2Pose, anchored on the teach frame by anchor_p2p and fed every frame from then on, so
        # its session is current whichever algorithm is selected and a switch never re-anchors it.
        self.p2p = p2p
        self.p2p_last: dict | None = None  # its answer for the current frame

    def _reset(self, algo: str) -> None:
        self.algo, self.state, self.misses, self.last_fit = algo, "acquiring", 0, None
        self.last_centre, self.velocity = None, np.zeros(2)
        self.klt, self.klt_idx = None, None
        self.certified = []

    def close(self) -> None:
        self.p2p = None  # the process belongs to the worker; the next teach re-anchors it

    def _priors(self, frame: _Frame) -> list:
        """The recent certified motions, each with how far from it the object may be by now; those
        whose bound has widened to any pose are dropped, so a long occlusion leaves none."""
        kept = []
        for t, motion in self.certified:
            rot_deg, trans_m = motion_bound(frame.t - t)
            if rot_deg < 180.0:
                kept.append((t, motion, rot_deg, trans_m))
        self.certified = [(t, m) for t, m, _, _ in kept]
        return [(m, r, d) for _, m, r, d in kept]

    def _acquire(self, frame: _Frame):
        # Between frames of one track the designation is carried forward from the segmenter's
        # memory; a track that is lost, or just starting, detects the object by name afresh.
        continuous = DESIGNATE_FROM_MEMORY and self.state in ("tracking", "occluded")
        mask = self.sam.mask(frame.rgb, continuous=continuous)
        if mask is None:
            return None, None, np.zeros((0, 2)), None
        region = _acquire_region(self.card, mask, frame.depth)
        fit, live, idx, patches = _bind(
            self.card, frame, region, self.tier, self.intr, priors=self._priors(frame)
        )
        self.patches = patches
        return mask, fit, live, idx

    def _seed_klt(self, frame: _Frame, fit, live: np.ndarray, idx: np.ndarray) -> None:
        inl = fit.inliers
        self.klt = KLTTracker()
        self.klt.init(frame.rgb, live[inl])
        self.klt_idx = idx[inl]

    def step(self, frame: _Frame, algo: str) -> dict:
        if algo != self.algo:
            self._reset(algo)
        t0 = time.perf_counter()
        mask, fit, live, idx = None, None, np.zeros((0, 2)), None
        n_matches = 0
        depth_extra: dict = {}
        # Between two certified frames the window suffices; after any miss SAM3 designates again, which
        # costs one slow frame and finds an object that slid out of the window or came back from behind a hand.
        fresh = self.last_fit is None or self.misses > 0 or self.state in ("acquiring", "lost")
        if self.p2p is not None:
            self.p2p_last = self.p2p.step(frame.rgb, frame.depth)
        if algo in P2P_CONFIGS:
            r = self.p2p_last
            if r is None:
                raise RuntimeError("Point2Pose is not running; teach again with its environment installed")
            if not r.get("ok"):
                raise RuntimeError(r.get("reason", "Point2Pose failed"))
            mask = r.get("mask")
            live = np.asarray(r["live_uv"], dtype=np.float64).reshape(-1, 2)
            n_matches = int(r["n_visible"])
            if not r["lost"]:
                fit = _PoseFit(np.asarray(r["delta"]), n_matches, float(r["mean_residual_m"]))
            depth_extra = {
                "model_xyz": r.get("model"),
                # What a run recording keeps: the points this pose was fitted on, and the tracker's own counts.
                **{k: r[k] for k in P2P_FIT_FIELDS if k in r},
            }
        elif algo == "refind" or (algo in ("dino", "klt") and fresh):
            mask, fit, live, idx = self._acquire(frame)
            n_matches = len(live)
            if algo == "klt" and fit is not None:
                self._seed_klt(frame, fit, live, idx)
        elif algo == "dino":
            # The window is the card where it was last seen, carried by its last pixel velocity and
            # grown by a fraction of its own size: the descriptor extractor crops to the window, so a
            # window much larger than the object changes the patch scale and loses the matches.
            # The descriptor extractor crops to the region and resizes the crop to a fixed size, so the
            # region must have the teach view's extent for the patches to match the card's: the card's
            # projected bounding box, carried by its velocity, with no padding of our own.
            proj = (
                self.intr.project(self.last_fit.transform.apply(self.card.xyz[: self.card.n_teach]))
                + self.velocity
            )
            window = _rect_mask(proj, frame.depth.shape)
            fit, live, idx, self.patches = _bind(
                self.card, frame, window, self.tier, self.intr, priors=self._priors(frame)
            )
            n_matches = len(live)
        elif algo == "klt":
            st = self.klt.step(frame.rgb)
            valid = np.flatnonzero(st.valid)
            n_matches = len(valid)
            if len(valid) >= KLT_MIN_POINTS:
                z, ok = sample_depth(frame.depth, st.uv[valid])
                rows = valid[ok]
                live, idx = st.uv[rows], self.klt_idx[rows]
                inlier_m = float(np.clip(0.15 * self.card.radius, 0.003, 0.010))
                cand = ransac_fit_rigid(
                    self.card.xyz[idx],
                    self.intr.deproject(live, z[ok]),
                    inlier_m=inlier_m,
                    priors=self._priors(frame),
                )
                if cand.ok and cand.n_inliers >= MIN_INLIERS and cand.scale_is_plausible():
                    fit = cand
                    self.klt.drop(rows[~cand.inliers])  # a point that left the consensus is off the object
        elif algo == "depth":
            if self.depth_teach is None:
                intr_d = {"fx": self.intr.fx, "fy": self.intr.fy, "cx": self.intr.cx, "cy": self.intr.cy}
                self.depth_teach = core.shape_teach_mask(
                    self.card.scene.depth, intr_d, self.card.mask, self.card.scene.rgb
                )
            intr_d = {"fx": self.intr.fx, "fy": self.intr.fy, "cx": self.intr.cx, "cy": self.intr.cy}
            r = core.shape_register(self.depth_teach, frame.depth, intr_d, frame.rgb)
            if r.get("ok"):
                mask = r["live_mask"]
                n_matches = int(r["n_points"])
                fit = _DepthFit(np.asarray(r["delta_cam"]), int(r["n_points"]))
                depth_extra = {"yaw_deg": float(r["yaw_deg"]), "symmetric": bool(r["symmetric"])}
        else:
            raise ValueError(f"unknown tracking algorithm {algo!r}")

        certified = fit is not None
        if certified:
            centre = self.intr.project(fit.transform.apply(self.card.xyz.mean(axis=0).reshape(1, 3)))[0]
            # A velocity is only the step between two consecutive certified frames; after a miss the
            # last centre is stale and carrying its jump forward threw the window past the object.
            self.velocity = (
                np.zeros(2) if (self.last_centre is None or self.misses > 0) else centre - self.last_centre
            )
            self.last_centre = centre
            self.state, self.misses, self.last_fit, self.last_t = "tracking", 0, fit, frame.t
            self.certified.append((frame.t, fit.transform))
        else:
            self.misses += 1
            self.velocity = np.zeros(2)
            self.state = "occluded" if self.misses < LOST_AFTER else "lost"
        out = {
            **depth_extra,
            "ok": certified,
            "state": self.state,
            "algo": algo,
            "n_matches": int(n_matches),
            "ms": (time.perf_counter() - t0) * 1000.0,
            "mask": mask,
            "live_uv": live[fit.inliers] if certified and idx is not None else live,
        }
        if certified:
            out.update(
                delta=_delta(fit), n_inliers=int(fit.n_inliers), rms_m=float(fit.rms), scale=float(fit.scale)
            )
            face_region = (
                mask
                if (mask is not None and algo != "depth")
                else _hull_mask(
                    self.intr.project(fit.transform.apply(self.card.xyz)), frame.depth.shape, FACE_PAD_PX
                )
            )
            geo = _geometry(self.card, frame.depth, face_region, self.intr, self.last_yaw)
            surface = geo.pop("surface")
            geo.pop("blob")
            geo.pop("face_pts")
            if algo == "depth":
                geo["face_find"] = None
            out.update(geo)
            if out.get("footprint_yaw_deg") is not None:
                self.last_yaw = float(out["footprint_yaw_deg"])
            if (
                algo in ("refind", "dino")
                and self.patches is not None
                and fit.n_inliers >= CARD_GROW_MIN_INLIERS
                and fit.n_inliers >= CARD_GROW_MIN_RATIO * max(n_matches, 1)
                and (
                    frame.t - self.last_grow >= GROW_EVERY_S
                    or _turn_deg(fit.transform.rot, self.last_grow_rot) >= CARD_GROW_TURN_DEG
                )
            ):
                out["card_grew"] = _grow_card(self.card, frame, fit, self.patches, self.intr, surface)
                self.last_grow, self.last_grow_rot = frame.t, np.array(fit.transform.rot)
            out["card_points"] = int(len(self.card.xyz))
        return out


class _PoseFit:
    """A motion computed elsewhere in the fit's clothes: its supporting points count as inliers."""

    def __init__(self, delta: np.ndarray, n_points: int, rms_m: float = 0.0):
        from lerobot.showservo.pose import Rigid3

        self.transform = Rigid3(delta[:3, :3], delta[:3, 3])
        self.n_inliers = n_points
        self.inliers = np.ones(0, dtype=bool)
        self.rms = rms_m
        self.scale = 1.0


_DepthFit = _PoseFit


def _npz(compress: bool = True, **arrays) -> bytes:
    buf = io.BytesIO()
    (np.savez_compressed if compress else np.savez)(buf, **arrays)
    return buf.getvalue()


def run(server: str, models: Models) -> None:
    import requests

    http = requests.Session()
    cards: dict[str, Card] = {}
    trackers: dict[str, Tracker] = {}
    print("worker ready", flush=True)
    idle_errors = 0
    while True:
        try:
            r = http.get(server + "api/pregrasp/worker/job", params={"wait": 20}, timeout=30)
        except requests.RequestException as e:
            idle_errors += 1
            print(f"server unreachable ({e}); retry {idle_errors}", flush=True)
            if idle_errors > 10:
                return
            time.sleep(2)
            continue
        idle_errors = 0
        if r.status_code == 410:
            print("worker stopped by the server", flush=True)
            return
        if r.status_code != 200:
            continue
        job = r.json()
        job_id, kind, concept = job["id"], job["kind"], job["concept"]
        t0 = time.perf_counter()
        try:
            fr = http.get(server + "api/pregrasp/worker/frame.npz", params={"id": job_id}, timeout=30)
            fr.raise_for_status()
            data = np.load(io.BytesIO(fr.content))
            intr_d = json.loads(str(data["intr"]))
            intr = CameraIntrinsics(fx=intr_d["fx"], fy=intr_d["fy"], cx=intr_d["cx"], cy=intr_d["cy"])
            frame = _Frame(data["rgb"], data["depth"], f"job_{job_id}")
            sam, tier = models.ensure(concept)
            if kind == "track":
                result = _track(job, frame, cards, trackers, sam, tier, intr, models=models)
            elif kind == "stream_object":

                def progress(done: int, total: int, job_id: str = job_id) -> None:
                    # Progress is a courtesy; the result still arrives without it.
                    with contextlib.suppress(requests.RequestException):
                        http.post(
                            server + "api/pregrasp/worker/progress",
                            params={"id": job_id, "done": done, "total": total},
                            timeout=5,
                        )

                result = _track_stream(job, frame, sam, models, intr, progress)
            elif kind == "locate":
                result = _locate(frame, sam, tier, intr, job.get("click"), _job_ref(job, data))
            else:
                mode = job.get("algo") if job.get("algo") in P2P_CONFIGS else "p2p"
                p2p = models.p2p_bridge(mode) if kind == "teach" else None
                ref = _job_ref(job, data)
                result = _teach_or_find(
                    kind,
                    concept,
                    frame,
                    cards,
                    trackers,
                    sam,
                    tier,
                    intr,
                    p2p=p2p,
                    click=job.get("click"),
                    mode=mode,
                    ref=ref,
                )
        except Exception as e:  # the job fails, the worker lives
            import traceback

            traceback.print_exc()
            result = _npz(meta=json.dumps({"ok": False, "reason": f"worker error: {e}"}))
        dt = time.perf_counter() - t0
        try:
            http.post(server + "api/pregrasp/worker/result", params={"id": job_id}, data=result, timeout=30)
        except requests.RequestException as e:
            print(f"could not post result for {job_id}: {e}", flush=True)
        if kind != "track":
            print(f"{kind} {job_id} done in {dt:.1f} s", flush=True)


STREAM_MASK_SCALE = (
    4  # a recorded object's masks are kept at this fraction of the frame: enough to draw, small to store
)


def _track_stream(job, frame, sam, models, intr, progress) -> bytes:
    """Track one object through a recorded demo stream, forward and backward from the frame it was clicked on.

    The object is whatever SAM3 segments under the click on frame ``k`` (the job's own
    frame). Point2Pose follows it from there to the end, then, started afresh on the
    same frame, back to the start, in a process of its own. Post: NPZ with ``deltas``
    (K, 4, 4), the object's motion from frame ``k`` in camera coordinates (identity at
    ``k``); ``seen`` (K,), whether Point2Pose still had it; ``masks`` (K, h/s, w/s),
    its mask where Point2Pose gave one; ``mask``, the full-size mask on frame ``k``.
    """
    import cv2

    rec = pathlib.Path(job["recording"])
    k = int(job["frame"])
    x, y = job["click"]
    n = len(np.atleast_1d(np.loadtxt(rec / "times.txt")))
    mask0 = sam.mask_at(frame.rgb, float(x), float(y))
    if mask0 is None or not mask0.any():
        return _npz(meta=json.dumps({"ok": False, "reason": "nothing segments at that pixel"}))
    bridge = models.p2p_bridge("p2p", key="stream")
    if bridge is None:
        reason = models.p2p_error.get("stream", "Point2Pose is unavailable")
        return _npz(meta=json.dumps({"ok": False, "reason": reason}))
    h, w = mask0.shape
    hs, ws = h // STREAM_MASK_SCALE, w // STREAM_MASK_SCALE

    def small(m: np.ndarray) -> np.ndarray:
        return cv2.resize(np.asarray(m, dtype=np.uint8), (ws, hs), interpolation=cv2.INTER_AREA) > 0

    def read(j: int) -> tuple[np.ndarray, np.ndarray]:
        bgr = cv2.imread(str(rec / "rgb" / f"{j:06d}.jpg"), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(rec / "depth" / f"{j:06d}.png"), cv2.IMREAD_UNCHANGED)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float32) / 1000.0

    deltas = np.tile(np.eye(4), (n, 1, 1))
    seen = np.zeros(n, dtype=bool)
    masks = np.zeros((n, hs, ws), dtype=bool)
    seen[k], masks[k] = True, small(mask0)
    done = 0
    try:
        for order in (range(k + 1, n), range(k - 1, -1, -1)):
            bridge.init(frame.rgb, frame.depth, mask0, intr)
            for j in order:
                r = bridge.step(*read(j))
                if not r.get("ok"):
                    reason = f"Point2Pose failed at frame {j}: {r.get('reason', 'no reason given')}"
                    return _npz(meta=json.dumps({"ok": False, "reason": reason}))
                deltas[j] = np.asarray(r["delta"], dtype=float)
                seen[j] = not bool(r.get("lost"))
                if r.get("mask") is not None:
                    masks[j] = small(r["mask"])
                done += 1
                if done % 10 == 0:
                    progress(done, n - 1)
    finally:
        # A pipeline started afresh does not hand back all its GPU memory, so each job gets a new process.
        models.drop_bridge("stream")
    progress(n - 1, n - 1)
    return _npz(
        meta=json.dumps({"ok": True, "frames": n, "seen_fraction": float(seen.mean())}),
        deltas=deltas,
        seen=seen,
        masks=masks,
        mask=mask0,
    )


def _job_ref(job: dict, data) -> dict | None:
    """The demo's view of the object a job names, when it carries one: its recording, frame and mask there."""
    if not job.get("ref_recording") or "ref_mask" not in data.files:
        return None
    return {"recording": job["ref_recording"], "frame": job["ref_frame"], "mask": data["ref_mask"]}


def _locate(frame: _Frame, sam, tier: DinoTier, intr: CameraIntrinsics, click, ref: dict | None) -> bytes:
    """Find the object under ``click`` against the demo's view of it, and nothing else: no card is taught and no
    track or Point2Pose session restarts. For the object a place goes onto, found before the act, and for the held
    object in the gripper, which the live track is busy following.

    Post: NPZ with ``meta`` (``ok`` once SAM3 has a mask, then ``ref_ok`` with ``ref_inliers``, ``ref_turn_deg`` and
    ``ref_card_points``, or ``ref_reason``), ``mask``, and ``ref_delta`` (4x4, camera frame) when found.
    """
    if ref is None:
        return _npz(meta=json.dumps({"ok": False, "reason": "a locate needs the demo's view of the object"}))
    if click is None:
        return _npz(meta=json.dumps({"ok": False, "reason": "a locate needs a click on the object"}))
    mask = sam.mask_at(frame.rgb, click[0], click[1])
    if mask is None:
        return _npz(meta=json.dumps({"ok": False, "reason": "SAM3 found no object under the click"}))
    found = _find_reference(ref, frame, mask, tier, intr)
    arrays = {"ref_delta": found.pop("ref_delta")} if "ref_delta" in found else {}
    return _npz(meta=json.dumps({"ok": True, **found}), mask=mask, **arrays)


def _find_reference(
    ref: dict, frame: _Frame, mask: np.ndarray, tier: DinoTier, intr: CameraIntrinsics
) -> dict:
    """Register the live view of an object against the demo's view of it: the motion from that view to this one.

    ``ref`` names the demo's recorded stream, the frame the object was designated on
    and its mask there. The demo's view becomes a card and is matched once against the
    live mask, with no tracking between them. Post: ``ref_ok`` with ``ref_delta`` (4x4,
    camera coordinates), ``ref_inliers``, ``ref_turn_deg`` and ``ref_card_points``; or ``ref_ok``
    False with ``ref_reason``.
    """
    import cv2

    rec = pathlib.Path(ref["recording"])
    k = int(ref["frame"])
    bgr = cv2.imread(str(rec / "rgb" / f"{k:06d}.jpg"), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(rec / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED)
    if bgr is None or depth is None:
        return {"ref_ok": False, "ref_reason": f"the demo's frame {k} is missing from {rec}"}
    view = _Frame(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth.astype(np.float32) / 1000.0, "reference")
    card = Card(view, np.asarray(ref["mask"], dtype=bool), tier, intr)
    fit, _uv, _idx, _patches = _bind(card, frame, mask, tier, intr)
    if fit is None:
        return {"ref_ok": False, "ref_reason": "the live view does not match the demo's view of the object"}
    delta = np.eye(4)
    delta[:3, :3], delta[:3, 3] = fit.transform.rot, fit.transform.trans
    turn = float(np.degrees(np.arccos(np.clip((np.trace(delta[:3, :3]) - 1.0) / 2.0, -1.0, 1.0))))
    return {
        "ref_ok": True,
        "ref_delta": delta,
        "ref_inliers": int(fit.n_inliers),
        "ref_turn_deg": turn,
        "ref_card_points": len(card.xyz),  # what the inliers are a share of
    }


def _teach_or_find(
    kind, concept, frame, cards, trackers, sam, tier, intr, p2p=None, click=None, mode="p2p", ref=None
) -> bytes:
    if click is not None:
        mask = sam.mask_at(frame.rgb, click[0], click[1])
        if mask is None:
            return _npz(meta=json.dumps({"ok": False, "reason": "SAM3 found no object under the click"}))
    else:
        mask = sam.mask(frame.rgb)
        if mask is None:
            return _npz(meta=json.dumps({"ok": False, "reason": f"SAM3 found no {concept!r} in the frame"}))
    if kind == "teach":
        card = Card(frame, mask, tier, intr)
        card.n_teach = int(len(card.xyz))
        card.mask = mask
        geo = _geometry(None, frame.depth, mask, intr)
        fit_t = geo["surface"]
        card.face = geo["face_find"]
        card.table_normal = geo["table_find"]
        card.shape = geo["blob"]
        card.face_xy = None if fit_t is None else _face_outline(geo["face_pts"], fit_t)
        cards[concept] = card
        old = trackers.pop(concept, None)  # a new card starts a new track
        if old is not None:
            old.close()
        card.p2p_anchored = {mode: True} if (p2p is not None and anchor_p2p(p2p, card, intr)) else {}
        meta = {
            "ok": True,
            "n_points": int(len(card.uv)),
            "radius_mm": card.radius * 1000.0,
            "shape_class": card.shape_class,
            "yaw_observable": bool(card.yaw_observable),
            "face": card.face,
        }
        arrays = {}
        if ref is not None:
            found = _find_reference(ref, frame, mask, tier, intr)
            if "ref_delta" in found:
                arrays["ref_delta"] = found.pop("ref_delta")
            meta.update(found)
        return _npz(meta=json.dumps(meta), mask=mask, uv=card.uv, xyz=card.xyz, **arrays)
    card = cards.get(concept)
    if card is None:
        return _npz(
            meta=json.dumps({"ok": False, "reason": f"no card taught for {concept!r} in this worker"})
        )
    fit, live_uv, _idx, patches = _bind(card, frame, mask, tier, intr)
    if fit is None:
        meta = {
            "ok": False,
            "reason": "no certified fit (too few matches, or no rigid consensus)",
            "n_matches": int(len(live_uv)),
        }
        return _npz(meta=json.dumps(meta), mask=mask, live_uv=live_uv)
    meta = {
        "ok": True,
        "n_matches": int(len(live_uv)),
        "n_inliers": int(fit.n_inliers),
        "rms_m": float(fit.rms),
        "scale": float(fit.scale),
        "shape_class": card.shape_class,
        "yaw_observable": bool(card.yaw_observable),
        "face_teach": getattr(card, "face", None),
        "table_teach": getattr(card, "table_normal", None),
    }
    geo = _geometry(card, frame.depth, mask, intr)
    surface = geo.pop("surface")
    geo.pop("blob")
    geo.pop("face_pts")
    meta.update(geo)
    if fit.n_inliers >= CARD_GROW_MIN_INLIERS and fit.n_inliers >= CARD_GROW_MIN_RATIO * max(len(live_uv), 1):
        meta["card_grew"] = _grow_card(card, frame, fit, patches, intr, surface)
    meta["card_points"] = int(len(card.xyz))
    return _npz(meta=json.dumps(meta), mask=mask, live_uv=live_uv[fit.inliers], delta=_delta(fit))


def _track(job, frame, cards, trackers, sam, tier, intr, models=None) -> bytes:
    concept, algo = job["concept"], job.get("algo") or "dino"
    card = cards.get(concept)
    if card is None:
        return _npz(
            compress=False,
            meta=json.dumps(
                {"ok": False, "state": "no card", "reason": f"no card taught for {concept!r} in this worker"}
            ),
        )
    tracker = trackers.get(concept)
    if tracker is None:
        tracker = trackers[concept] = Tracker(card, sam, tier, intr)
    if algo in P2P_CONFIGS and models is not None:
        bridge = models.p2p_bridge(algo)
        anchored = getattr(card, "p2p_anchored", {})
        if bridge is not None and not anchored.get(algo):
            # This mode was not the one taught into: anchor it on the teach frame now. If the object
            # has moved since the teach, that anchor is stale and the operator should teach again.
            anchored[algo] = anchor_p2p(bridge, card, intr)
            card.p2p_anchored = anchored
        tracker.p2p = bridge if anchored.get(algo) else None
    out = tracker.step(frame, algo)
    meta = {
        k: v for k, v in out.items() if k not in ("mask", "live_uv", "delta", "model_xyz", *P2P_FIT_ARRAYS)
    }
    meta.update(
        shape_class=card.shape_class,
        yaw_observable=bool(card.yaw_observable),
        face_teach=getattr(card, "face", None),
        table_teach=getattr(card, "table_normal", None),
    )
    arrays = {"live_uv": np.asarray(out["live_uv"], dtype=np.float32)}
    if out.get("mask") is not None:
        arrays["mask"] = out["mask"]
    for key in P2P_FIT_ARRAYS:
        if out.get(key) is not None:
            arrays[key] = np.asarray(out[key])
    if out.get("ok"):
        arrays["delta"] = out["delta"]
        # The object as known so far, in the teach frame: the card (teach view plus every side grown
        # since) or Point2Pose's adopted key points. Drawn carried by the motion, so growth is visible.
        model = out.get("model_xyz")
        arrays["model_xyz"] = np.asarray(card.xyz if model is None else model, dtype=np.float32)
    return _npz(compress=False, meta=json.dumps(meta), **arrays)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--server", required=True, help="GUI base URL, e.g. http://127.0.0.1:9100/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--resolution", type=int, default=1008, help="SAM3 inference resolution")
    args = ap.parse_args()
    server = args.server if args.server.endswith("/") else args.server + "/"
    run(server, Models(args.device, args.dino_model, args.resolution))


if __name__ == "__main__":
    main()
