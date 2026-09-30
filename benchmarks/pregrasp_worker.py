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
from showservo_real import Card, bind_rigid3d  # noqa: E402

from lerobot.showservo.pose import CameraIntrinsics  # noqa: E402


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


def _npz(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def run(server: str, models: Models) -> None:
    import requests

    http = requests.Session()
    cards: dict[str, Card] = {}
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
            mask = sam.mask(frame.rgb)
            if mask is None:
                result = _npz(
                    meta=json.dumps({"ok": False, "reason": f"SAM3 found no {concept!r} in the frame"})
                )
            elif kind == "teach":
                card = Card(frame, mask, tier, intr)
                card.face = face_plane(frame.depth, mask, intr)
                cards[concept] = card
                meta = {
                    "ok": True,
                    "n_points": int(len(card.uv)),
                    "radius_mm": card.radius * 1000.0,
                    "shape_class": card.shape_class,
                    "yaw_observable": bool(card.yaw_observable),
                    "face": card.face,
                }
                result = _npz(meta=json.dumps(meta), mask=mask, uv=card.uv, xyz=card.xyz)
            else:
                card = cards.get(concept)
                if card is None:
                    result = _npz(
                        meta=json.dumps(
                            {"ok": False, "reason": f"no card taught for {concept!r} in this worker"}
                        )
                    )
                else:
                    fit, live_uv = bind_rigid3d(card, frame, mask, tier, intr)
                    if fit is None:
                        n = 0 if live_uv is None else int(len(live_uv))
                        meta = {
                            "ok": False,
                            "reason": "no certified fit (too few matches, or no rigid consensus)",
                            "n_matches": n,
                        }
                        result = _npz(
                            meta=json.dumps(meta),
                            mask=mask,
                            live_uv=np.zeros((0, 2)) if live_uv is None else live_uv,
                        )
                    else:
                        delta = np.eye(4)
                        delta[:3, :3], delta[:3, 3] = fit.transform.rot, fit.transform.trans
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
                        }
                        result = _npz(
                            meta=json.dumps(meta), mask=mask, live_uv=live_uv[fit.inliers], delta=delta
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
        print(f"{kind} {job_id} done in {dt:.1f} s", flush=True)


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
