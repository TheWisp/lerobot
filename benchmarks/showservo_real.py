# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Teach and bind on REAL captured scenes — the pipeline's first contact with reality.

Consumes a directory written by ``showservo_capture.py``: teach a card from one scene,
then bind and 3D-fit every other scene, exactly as the runtime would at a stage start.
No robot, no motion — the real-world twin of the sim sweeps, with SAM3 doing the
designation ground truth did in sim, and the rigid3d certificate (the gate the
viewpoint sweep showed is not viewpoint-limited).

There is no simulator to grade against out here, so the report gives what a ruler can
check and what an eye can check:

* per scene: certified?, inliers, the fitted motion of the taught object —
  ``|t|`` in mm, its camera-frame components (x right, y down, z away from camera),
  and the rotation angle;
* an overlay per scene: designation edge, matched inliers, and the TAUGHT object's
  point cloud re-projected through the fit — if the ghost lands on the object, the
  fit is right.

Place the object at tape-measured offsets between captures and compare ``|t|`` against
the ruler; that is the protocol.

Usage:
    PYTHONPATH=src <python-with-transformers> benchmarks/showservo_real.py \\
        --captures captures/ring_session --concept "green ring" --teach 0

    # several taught scenes act as multiple demos (best certificate wins):
    ... --teach 0 1 2

    # no SAM3 (or to override it): put a mask.png (nonzero = object) in each scene dir
    ... --mask files
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from showservo_m0 import DinoTier, Sam3Concept  # noqa: E402

from lerobot.fewshot.registration import mutual_matches  # noqa: E402
from lerobot.showservo.pose import CameraIntrinsics, ransac_fit_rigid, sample_depth  # noqa: E402

MIN_INLIERS = 6
MAX_CARD = 400


class Scene:
    """One captured scene. Post: rgb HxWx3 uint8, depth HxW float32 metres (0 = hole)."""

    def __init__(self, path: pathlib.Path):
        import cv2

        self.path = path
        self.name = path.name
        bgr = cv2.imread(str(path / "rgb.png"))
        assert bgr is not None, f"missing {path / 'rgb.png'}"
        self.rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        self.depth = np.load(path / "depth.npy")
        assert self.depth.shape == self.rgb.shape[:2], "depth/color size mismatch"

    def file_mask(self) -> np.ndarray | None:
        import cv2

        p = self.path / "mask.png"
        if not p.exists():
            return None
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        assert m is not None and m.shape == self.rgb.shape[:2], f"bad mask at {p}"
        return m > 0


def load_captures(root: pathlib.Path) -> tuple[CameraIntrinsics, list[Scene]]:
    meta = json.loads((root / "intrinsics.json").read_text())
    intr = CameraIntrinsics(fx=meta["fx"], fy=meta["fy"], cx=meta["cx"], cy=meta["cy"])
    scenes = [Scene(p) for p in sorted(root.glob("scene_*"))]
    assert scenes, f"no scene_* directories under {root}"
    return intr, scenes


class Designator:
    """SAM3 by concept, or mask.png files — same erosion either way."""

    def __init__(self, mode: str, concept: str | None, device: str):
        self.mode = mode
        self.sam3 = Sam3Concept(concept, device=device) if mode == "sam3" else None

    def mask(self, scene: Scene) -> np.ndarray | None:
        import cv2

        if self.mode == "files":
            m = scene.file_mask()
            if m is None:
                return None
            k = 2 * Sam3Concept.ERODE_PX + 1
            eroded = cv2.erode(m.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
            return eroded if eroded.any() else m
        assert self.sam3 is not None
        return self.sam3.mask(scene.rgb)


class Card:
    """One taught scene: descriptors + 3D, in that scene's camera frame."""

    def __init__(self, scene: Scene, mask: np.ndarray, tier: DinoTier, intr: CameraIntrinsics):
        uv, desc = tier.teach(scene.rgb, mask)
        if len(uv) > MAX_CARD:
            keep = np.linspace(0, len(uv) - 1, MAX_CARD).astype(int)
            uv, desc = uv[keep], desc[keep]
        z, ok = sample_depth(scene.depth, uv)
        assert int(ok.sum()) >= MIN_INLIERS, (
            f"{scene.name}: only {int(ok.sum())} taught points carry depth — "
            "check the depth preview for holes on the object"
        )
        self.scene = scene
        self.uv = uv[ok]
        self.desc = np.asarray(desc[ok], dtype=np.float32)
        self.xyz = intr.deproject(uv[ok], z[ok])


def bind_rigid3d(card: Card, scene: Scene, mask: np.ndarray, tier: DinoTier, intr: CameraIntrinsics):
    """The rigid3d-gated bind. Post: (fit, live_uv_of_matches) — fit.ok is the verdict."""
    uv, desc = tier.teach(scene.rgb, mask)
    ia, ib = mutual_matches(card.desc, np.asarray(desc, dtype=np.float32))
    if len(ia) < MIN_INLIERS:
        return None, None
    z, ok = sample_depth(scene.depth, uv[ib])
    fit = ransac_fit_rigid(card.xyz[ia[ok]], intr.deproject(uv[ib][ok], z[ok]), inlier_m=0.010)
    if not (fit.ok and fit.n_inliers >= MIN_INLIERS and fit.scale_is_plausible()):
        return None, uv[ib][ok]
    return fit, uv[ib][ok]


def overlay(scene: Scene, mask: np.ndarray | None, fit, live_uv, card: Card, intr, out: pathlib.Path):
    import cv2

    vis = scene.rgb.copy()
    if mask is not None:
        edge = mask ^ (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        vis[edge] = (255, 0, 255)
    if live_uv is not None:
        for p in live_uv:
            cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 220, 255), -1)
    if fit is not None:
        # The taught cloud carried through the fit: the ghost should land on the object.
        ghost = intr.project(fit.transform.apply(card.xyz))
        h, w = vis.shape[:2]
        for p in ghost:
            u, v = int(p[0]), int(p[1])
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(vis, (u, v), 2, (60, 230, 60), -1)
        cv2.putText(vis, "green ghost = taught cloud through the fit", (10, 24), 0, 0.6, (60, 230, 60), 2)
    else:
        cv2.putText(vis, "NO CERTIFIED FIT", (10, 24), 0, 0.7, (255, 70, 70), 2)
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True, type=pathlib.Path)
    ap.add_argument("--concept", default=None, help="SAM3 text prompt, e.g. 'green ring'")
    ap.add_argument("--mask", default="sam3", choices=("sam3", "files"))
    ap.add_argument("--teach", type=int, nargs="+", default=[0], help="scene indices to teach from")
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    assert args.mask == "files" or args.concept, "--concept is required with --mask sam3"

    intr, scenes = load_captures(args.captures)
    designator = Designator(args.mask, args.concept, args.device)
    tier = DinoTier(args.dino_model, device=args.device)

    cards = []
    for i in args.teach:
        mask = designator.mask(scenes[i])
        assert mask is not None, f"designation found nothing in teach scene {scenes[i].name}"
        cards.append(Card(scenes[i], mask, tier, intr))
        print(f"taught from {scenes[i].name}: {len(cards[-1].uv)} points with depth")

    print(
        f"\n{'scene':>10} {'demo':>5} {'inliers':>8} {'|move| mm':>10} "
        f"{'mx':>7} {'my':>7} {'mz':>7} {'rot deg':>8}  overlay"
    )
    taught_idx = set(args.teach)
    for k, scene in enumerate(scenes):
        if k in taught_idx:
            continue
        mask = designator.mask(scene)
        if mask is None:
            print(f"{scene.name:>10}  designation found nothing")
            continue
        # Multiple taught scenes act as demos: best certificate wins.
        best, best_demo, best_uv = None, None, None
        for d, card in enumerate(cards):
            fit, live_uv = bind_rigid3d(card, scene, mask, tier, intr)
            if fit is not None and (best is None or fit.n_inliers > best.n_inliers):
                best, best_demo, best_uv = fit, d, live_uv
        out = scene.path / "overlay.jpg"
        card = cards[best_demo if best_demo is not None else 0]
        overlay(scene, mask, best, best_uv, card, intr, out)
        if best is None:
            print(f"{scene.name:>10}   REFUSED (no demo certified)  -> {out}")
            continue
        # The ruler-comparable number is how far the OBJECT moved: the taught cloud's
        # centroid carried through the fit. The transform's raw translation is
        # origin-dependent — measured from the camera's origin, a rotation about the
        # object's own axis shows up as a large phantom translation (half a metre of
        # lever arm), which the synthesized-capture dry run caught.
        centroid = card.xyz.mean(axis=0)
        move = (best.transform.apply(centroid.reshape(1, 3))[0] - centroid) * 1000.0
        from lerobot.showservo.pose import rotation_vector

        rot_deg = float(np.rad2deg(np.linalg.norm(rotation_vector(best.transform.rot))))
        print(
            f"{scene.name:>10} {best_demo:>5d} {best.n_inliers:>8d} {np.linalg.norm(move):>10.1f} "
            f"{move[0]:>7.1f} {move[1]:>7.1f} {move[2]:>7.1f} {rot_deg:>8.1f}  {out}"
        )
    print("\naxes: x right, y down, z away from camera. Compare |move| against the ruler.")


if __name__ == "__main__":
    main()
