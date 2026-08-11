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
        # 1008 rather than the adapter's 672 default: measured on a real edge-on,
        # hand-held scene, 672 returned nothing and 1008 designated it — a real
        # miss on exactly the marginal views the servo must survive. Designation
        # here is offline/low-rate, so the ~1.8x cost buys the recall.
        self.sam3 = Sam3Concept(concept, device=device, resolution=1008) if mode == "sam3" else None

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

        # Shape of the taught cloud, for two decisions downstream. ``radius`` scales
        # the residual gate to the OBJECT (a fixed 10 mm was 20% of a 48 mm ring, and
        # let a 90-degree roll certify as 4 degrees). ``yaw_observable``: a thin,
        # in-plane-ISOTROPIC cloud (a disc) maps onto itself under any rotation about
        # its normal, so that rotation rests entirely on texture — which on a
        # self-similar surface lies coherently. Measured: ring 0.79 in-plane ratio
        # (yaw certified 4 deg for a true ~90 deg roll), tissue pack 0.52 (yaw pinned
        # by the oblong geometry itself). Threshold 0.7 splits them with margin.
        spread = self.xyz - self.xyz.mean(axis=0)
        self.radius = float(np.percentile(np.linalg.norm(spread, axis=1), 80))
        lam = np.sort(np.linalg.eigvalsh(spread.T @ spread / len(spread)))[::-1]
        self.inplane_ratio = float(lam[1] / max(lam[0], 1e-12))
        self.yaw_observable = self.inplane_ratio < 0.7


def bind_rigid3d(card: Card, scene: Scene, mask: np.ndarray, tier: DinoTier, intr: CameraIntrinsics):
    """The rigid3d-gated bind. Post: (fit, live_uv_of_matches) — fit.ok is the verdict.

    The residual gate scales with the object: 15% of the taught cloud's radius,
    floored at the sensor's ~3 mm depth noise, capped at the old fixed 10 mm.
    """
    uv, desc = tier.teach(scene.rgb, mask)
    ia, ib = mutual_matches(card.desc, np.asarray(desc, dtype=np.float32))
    if len(ia) < MIN_INLIERS:
        return None, None
    z, ok = sample_depth(scene.depth, uv[ib])
    inlier_m = float(np.clip(0.15 * card.radius, 0.003, 0.010))
    fit = ransac_fit_rigid(card.xyz[ia[ok]], intr.deproject(uv[ib][ok], z[ok]), inlier_m=inlier_m)
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
        draw_gizmo(vis, fit, card, intr)
        cv2.putText(vis, "green ghost = taught cloud through the fit", (10, 24), 0, 0.6, (60, 230, 60), 2)
    else:
        cv2.putText(vis, "NO CERTIFIED FIT", (10, 24), 0, 0.7, (255, 70, 70), 2)
    cv2.imwrite(str(out), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


class _LiveFrame:
    """Duck-typed stand-in for :class:`Scene` in :func:`bind_rigid3d`."""

    def __init__(self, rgb: np.ndarray, depth: np.ndarray):
        self.rgb = rgb
        self.depth = depth


def draw_gizmo(vis: np.ndarray, fit, card: Card, intr: CameraIntrinsics) -> None:
    """Project a pose gizmo through the fit: XYZ triad + the taught cloud's box.

    The triad is the taught frame's axes carried by the fitted rotation — the part a
    dot cloud cannot show. Axis colours follow the universal convention X=red,
    Y=green, Z=blue (drawn in RGB; conversion to BGR happens at save). The box is the
    taught cloud's bounding volume, so it is thin along axes the teach view never saw
    — honest, not a bug.
    """
    import cv2

    centroid = card.xyz.mean(axis=0)
    spread = card.xyz - centroid
    axis_len = float(np.percentile(np.linalg.norm(spread, axis=1), 80))

    def px(points3):
        return intr.project(fit.transform.apply(np.asarray(points3, dtype=np.float64)))

    origin = px([centroid])[0]
    o = (int(origin[0]), int(origin[1]))
    for axis, color, label in (
        ((axis_len, 0, 0), (255, 60, 60), "x"),
        ((0, axis_len, 0), (60, 255, 60), "y"),
        ((0, 0, axis_len), (80, 120, 255), "z"),
    ):
        tip = px([centroid + np.array(axis)])[0]
        t = (int(tip[0]), int(tip[1]))
        cv2.arrowedLine(vis, o, t, color, 3, cv2.LINE_AA, tipLength=0.22)
        cv2.putText(vis, label, (t[0] + 4, t[1] - 4), 0, 0.6, color, 2, cv2.LINE_AA)

    lo, hi = card.xyz.min(axis=0), card.xyz.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    c2 = px(corners).astype(int)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (4, 6), (5, 7), (6, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for a, b in edges:
        cv2.line(vis, tuple(c2[a]), tuple(c2[b]), (250, 240, 80), 1, cv2.LINE_AA)


def live_loop(server: str, cards: list[Card], designator, tier, intr) -> None:
    """Fit every frame the GUI serves, post the annotated view back. Exits when the
    server stops answering with frames (live mode turned off, session closed).

    This IS the bind-only runtime loop — designate, bind, fit, fresh each frame, best
    demo card wins — just rendered to pixels instead of feeding a controller.
    """
    import io

    import cv2
    import requests

    http = requests.Session()
    while True:
        r = http.get(server + "api/showservo/live/frame.npz", timeout=10)
        if r.status_code != 200:
            return
        data = np.load(io.BytesIO(r.content))
        frame = _LiveFrame(data["rgb"], data["depth"])

        mask = designator.mask(frame)
        best, best_uv, best_demo = None, None, 0
        if mask is not None:
            for d, card in enumerate(cards):
                fit, uv = bind_rigid3d(card, frame, mask, tier, intr)
                if fit is not None and (best is None or fit.n_inliers > best.n_inliers):
                    best, best_uv, best_demo = fit, uv, d

        vis = frame.rgb.copy()
        if mask is not None:
            edge = mask ^ (cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            vis[edge] = (255, 0, 255)
        if best is not None:
            card = cards[best_demo]
            for p in best_uv:
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 220, 255), -1)
            ghost = intr.project(best.transform.apply(card.xyz))
            h, w = vis.shape[:2]
            for p in ghost:
                u, v = int(p[0]), int(p[1])
                if 0 <= u < w and 0 <= v < h:
                    cv2.circle(vis, (u, v), 2, (60, 230, 60), -1)
            draw_gizmo(vis, best, card, intr)
            centroid = card.xyz.mean(axis=0)
            move = (best.transform.apply(centroid.reshape(1, 3))[0] - centroid) * 1000.0
            from lerobot.showservo.pose import rotation_vector

            rot_deg = float(np.rad2deg(np.linalg.norm(rotation_vector(best.transform.rot))))
            rot_txt = f"{rot_deg:5.1f} deg" if card.yaw_observable else "n/obs (disc)"
            text = (
                f"|move| {np.linalg.norm(move):6.1f} mm   rot {rot_txt}   "
                f"inliers {best.n_inliers}   demo {best_demo}"
            )
            color = (60, 230, 60)
        elif mask is None:
            text, color = "designation found nothing", (255, 190, 90)
        else:
            text, color = "REFUSED (no certified fit)", (255, 70, 70)
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(vis, text, (10, 24), 0, 0.62, color, 2, cv2.LINE_AA)

        _ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
        p = http.post(
            server + "api/showservo/live/result",
            data=jpg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=10,
        )
        if p.status_code != 200:
            return


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", required=True, type=pathlib.Path)
    ap.add_argument("--concept", default=None, help="SAM3 text prompt, e.g. 'green ring'")
    ap.add_argument("--mask", default="sam3", choices=("sam3", "files"))
    ap.add_argument("--teach", type=int, nargs="+", default=[0], help="scene indices to teach from")
    ap.add_argument("--dino-model", default="facebook/dinov3-vits16-pretrain-lvd1689m")
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--live",
        default=None,
        help="GUI base URL (e.g. http://127.0.0.1:9100/): fit the live camera stream "
        "against the taught cards instead of the captured scenes",
    )
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
        print(f"taught from {scenes[i].name}: {len(cards[-1].uv)} points with depth", flush=True)

    if args.live:
        server = args.live if args.live.endswith("/") else args.live + "/"
        print(f"live fit against {server} — stop from the GUI", flush=True)
        live_loop(server, cards, designator, tier, intr)
        return

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
        rot_str = f"{rot_deg:7.1f}{'*' if not card.yaw_observable else ' '}"
        print(
            f"{scene.name:>10} {best_demo:>5d} {best.n_inliers:>8d} {np.linalg.norm(move):>10.1f} "
            f"{move[0]:>7.1f} {move[1]:>7.1f} {move[2]:>7.1f} {rot_str:>8}  {out}"
        )
    print("\naxes: x right, y down, z away from camera. Compare |move| against the ruler.")
    if any(not c.yaw_observable for c in cards):
        print(
            "* the taught cloud is a near-isotropic disc: rotation about its normal is "
            "NOT observable — trust position, ignore yaw."
        )


if __name__ == "__main__":
    main()
