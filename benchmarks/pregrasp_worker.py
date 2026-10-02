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

"""The GUI's designation-and-matching worker: SAM3 by concept, DINO patch features, a 3D rigid fit.

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
import io
import json
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m0 import DinoTier, Sam3Concept  # noqa: E402
from showservo_real import MIN_INLIERS, Card  # noqa: E402

from lerobot.fewshot.registration import mutual_matches  # noqa: E402
from lerobot.gui.api import _pregrasp_core as core  # noqa: E402
from lerobot.showservo.pose import CameraIntrinsics, ransac_fit_rigid, sample_depth  # noqa: E402
from lerobot.showservo.tracker import KLTTracker  # noqa: E402


class _Frame:
    """What :class:`Card` and :func:`bind_rigid3d` need of a scene."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray, name: str):
        self.rgb = np.ascontiguousarray(rgb)
        self.depth = depth
        self.name = name


class Models:
    def __init__(self, device: str, dino_model: str, resolution: int):
        self.device, self.dino_model, self.resolution = device, dino_model, resolution
        self.sam: Sam3Concept | None = None
        self.tier: DinoTier | None = None

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
    """The face the camera sees: the plane holding the most of the designated depth cloud.

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
        return None
    pts = intr.deproject(np.stack([us, vs], axis=1), depth[vs, us])
    rng = np.random.default_rng(0)  # deterministic: the same cloud gives the same face
    inl = _consensus_plane(pts, rng, FACE_TRIALS)
    if inl is None:
        return None
    c = pts[inl].mean(axis=0)
    _u, _s, vt = np.linalg.svd(pts[inl] - c, full_matrices=False)
    n = vt[2]
    if n[2] > 0:
        n = -n
    second = _consensus_plane(pts[~inl], rng, FACE_TRIALS // 2)
    n_second = 0 if second is None else int(second.sum())
    return {
        "normal": n.tolist(),
        "centroid": c.tolist(),
        "planarity": float(inl.mean()),
        "n": int(len(us)),
        "n_plane": int(inl.sum()),
        "dominance": float(inl.sum() / max(n_second, 1)),
    }


def _bind(card: Card, frame: _Frame, region: np.ndarray, tier: DinoTier, intr: CameraIntrinsics):
    """Match the card inside ``region`` and fit the rigid motion teach -> now.

    Post: ``(fit, live_uv, card_idx)`` with ``fit`` None when nothing certified;
    ``live_uv`` are the matched pixels with depth and ``card_idx`` their card
    points, both over the same rows, so ``fit.inliers`` indexes either.
    """
    uv, desc = tier.teach(frame.rgb, region)
    ia, ib = mutual_matches(card.desc, np.asarray(desc, dtype=np.float32))
    if len(ia) < MIN_INLIERS:
        return None, np.zeros((0, 2)), None
    z, ok = sample_depth(frame.depth, uv[ib])
    live, idx = uv[ib][ok], ia[ok]
    inlier_m = float(np.clip(0.15 * card.radius, 0.003, 0.010))
    fit = ransac_fit_rigid(
        card.xyz[idx], intr.deproject(live, z[ok]), inlier_m=inlier_m, hypo_weights=card.hypo_w[idx]
    )
    if not (fit.ok and fit.n_inliers >= MIN_INLIERS and fit.scale_is_plausible()):
        return None, live, None
    return fit, live, idx


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


def _turn_from_footprint(card: Card, depth: np.ndarray, region: np.ndarray, intr: CameraIntrinsics) -> dict:
    """The surface under ``region`` and the footprint's turn teach -> now, as result keys.

    The footprint is geometry: it carries the turn of a plain object whose texture
    cannot, ambiguous only by the object's own symmetry.
    """
    fit_t = _table_fit(depth, region, intr)
    out: dict = {"table_find": None if fit_t is None else [float(v) for v in fit_t[0]]}
    shape = getattr(card, "shape", None)
    if fit_t is None or shape is None:
        return out
    found = _footprint(depth, region, fit_t, intr)
    if found is None:
        return out
    fy = core.footprint_yaw(shape["footprint"], found["footprint"])
    out.update(
        footprint_yaw_deg=float(fy["yaw_deg"]),
        footprint_iou=float(fy["iou"]),
        footprint_symmetric=bool(fy["symmetric"]),
        footprint_points=int(found["n_points"]),
    )
    return out


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


LOST_AFTER = 8  # frames without a certified fit before the object counts as lost rather than occluded
WINDOW_PAD_FRAC = 0.25  # the matching window grows by this fraction of the card's projected size
WINDOW_PAD_MIN_PX = 12
FACE_PAD_PX = 2
KLT_MIN_POINTS = 12  # fewer live tracked points than this and the KLT path re-acquires


class Tracker:
    """Follows one taught card frame after frame with the chosen algorithm.

    SAM3 designates only to acquire and to re-acquire; every certified frame is
    registered against the card (teach -> now), never against the previous
    frame, so nothing accumulates. States: acquiring, tracking, occluded (a few
    frames without a certified fit, the last pose is held), lost (longer; the
    designator runs every frame until the object is back).
    """

    def __init__(self, card: Card, sam: Sam3Concept, tier: DinoTier, intr: CameraIntrinsics):
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

    def _reset(self, algo: str) -> None:
        self.algo, self.state, self.misses, self.last_fit = algo, "acquiring", 0, None
        self.last_centre, self.velocity = None, np.zeros(2)
        self.klt, self.klt_idx = None, None

    def _acquire(self, frame: _Frame):
        mask = self.sam.mask(frame.rgb)
        if mask is None:
            return None, None, np.zeros((0, 2)), None
        fit, live, idx = _bind(self.card, frame, mask, self.tier, self.intr)
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
        fresh = self.last_fit is None or self.state in ("acquiring", "lost")
        if algo == "refind" or (algo in ("dino", "klt") and fresh):
            mask, fit, live, idx = self._acquire(frame)
            n_matches = len(live)
            if algo == "klt" and fit is not None:
                self._seed_klt(frame, fit, live, idx)
        elif algo == "dino":
            # The window is the card where it was last seen, carried by its last pixel velocity and
            # grown by a fraction of its own size: the descriptor extractor crops to the window, so a
            # window much larger than the object changes the patch scale and loses the matches.
            proj = self.intr.project(self.last_fit.transform.apply(self.card.xyz)) + self.velocity
            span = float(np.ptp(proj, axis=0).max())
            window = _hull_mask(proj, frame.depth.shape, max(WINDOW_PAD_MIN_PX, int(WINDOW_PAD_FRAC * span)))
            fit, live, idx = _bind(self.card, frame, window, self.tier, self.intr)
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
                    self.card.xyz[idx], self.intr.deproject(live, z[ok]), inlier_m=inlier_m
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
            self.state, self.misses, self.last_fit = "tracking", 0, fit
            centre = self.intr.project(fit.transform.apply(self.card.xyz.mean(axis=0).reshape(1, 3)))[0]
            self.velocity = np.zeros(2) if self.last_centre is None else centre - self.last_centre
            self.last_centre = centre
        else:
            self.misses += 1
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
            out["face_find"] = None if algo == "depth" else face_plane(frame.depth, face_region, self.intr)
            out.update(_turn_from_footprint(self.card, frame.depth, face_region, self.intr))
        return out


class _DepthFit:
    """The depth path's answer in the fit's clothes: a motion, with the blob's points as its inliers."""

    def __init__(self, delta: np.ndarray, n_points: int):
        from lerobot.showservo.pose import Rigid3

        self.transform = Rigid3(delta[:3, :3], delta[:3, 3])
        self.n_inliers = n_points
        self.inliers = np.ones(0, dtype=bool)
        self.rms = 0.0
        self.scale = 1.0


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
                result = _track(job, frame, cards, trackers, sam, tier, intr)
            else:
                result = _teach_or_find(kind, concept, frame, cards, trackers, sam, tier, intr)
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


def _teach_or_find(kind, concept, frame, cards, trackers, sam, tier, intr) -> bytes:
    mask = sam.mask(frame.rgb)
    if mask is None:
        return _npz(meta=json.dumps({"ok": False, "reason": f"SAM3 found no {concept!r} in the frame"}))
    if kind == "teach":
        card = Card(frame, mask, tier, intr)
        card.face = face_plane(frame.depth, mask, intr)
        card.mask = mask
        fit_t = _table_fit(frame.depth, mask, intr)
        card.table_normal = None if fit_t is None else [float(v) for v in fit_t[0]]
        card.shape = None if fit_t is None else _footprint(frame.depth, mask, fit_t, intr)
        cards[concept] = card
        trackers.pop(concept, None)  # a new card starts a new track
        meta = {
            "ok": True,
            "n_points": int(len(card.uv)),
            "radius_mm": card.radius * 1000.0,
            "shape_class": card.shape_class,
            "yaw_observable": bool(card.yaw_observable),
            "face": card.face,
        }
        return _npz(meta=json.dumps(meta), mask=mask, uv=card.uv, xyz=card.xyz)
    card = cards.get(concept)
    if card is None:
        return _npz(
            meta=json.dumps({"ok": False, "reason": f"no card taught for {concept!r} in this worker"})
        )
    fit, live_uv, _idx = _bind(card, frame, mask, tier, intr)
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
        "face_find": face_plane(frame.depth, mask, intr),
        "table_teach": getattr(card, "table_normal", None),
        **_turn_from_footprint(card, frame.depth, mask, intr),
    }
    return _npz(meta=json.dumps(meta), mask=mask, live_uv=live_uv[fit.inliers], delta=_delta(fit))


def _track(job, frame, cards, trackers, sam, tier, intr) -> bytes:
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
    out = tracker.step(frame, algo)
    meta = {k: v for k, v in out.items() if k not in ("mask", "live_uv", "delta")}
    meta.update(
        shape_class=card.shape_class,
        yaw_observable=bool(card.yaw_observable),
        face_teach=getattr(card, "face", None),
        table_teach=getattr(card, "table_normal", None),
    )
    arrays = {"live_uv": np.asarray(out["live_uv"], dtype=np.float32)}
    if out.get("mask") is not None:
        arrays["mask"] = out["mask"]
    if out.get("ok"):
        arrays["delta"] = out["delta"]
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
