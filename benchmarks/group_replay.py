"""Replay a camera recording through Point2Pose and the point groups (showservo.groups), and show what the groups
make of it: every track coloured by its group, each object's outline where its group puts it while its own points
are hidden, and the error of that estimate against the object's own points where they reappear.

Bodies in the Point2Pose session: the objects clicked on the start frame (each the raised body under its click, at
the click's height), and the tray around them in tiles, so the tracker has points on what the objects rest on.
Point2Pose's own per-body fit is kept for comparison ("p2p"), and so is a pose held where the object was last
seen ("held": what the trust gate does today).

Usage: python benchmarks/group_replay.py RECORDING OUT_DIR --object gamepad=u,v --object cube=u,v
         [--start K] [--end K] [--every N] [--tiles 3x2]
Writes OUT_DIR/groups.mp4, OUT_DIR/timeline.json and prints the hidden-pose errors per object."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pregrasp_worker import P2P_CONFIGS, P2PBridge  # noqa: E402

from lerobot.showservo.groups import GroupTracker  # noqa: E402
from lerobot.showservo.pose import CameraIntrinsics, ransac_fit_rigid  # noqa: E402

PALETTE = [
    (60, 180, 75),
    (230, 25, 75),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
    (170, 110, 40),
    (255, 250, 200),
]


def load_frame(rec: pathlib.Path, k: int) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(rec / "rgb" / f"{k:06d}.jpg"))
    depth = cv2.imread(str(rec / "depth" / f"{k:06d}.png"), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth


def points_3d(depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    z[z <= 0] = np.nan
    return np.stack([(u - k[0, 2]) * z / k[0, 0], (v - k[1, 2]) * z / k[1, 1], z], axis=-1)


def tray_height(pts: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Height of every pixel above the plane fitted (RANSAC) to the region's points: the tray, where most of the
    region is tray."""
    p = pts[region & np.isfinite(pts).all(axis=2)]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(200):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n = n / np.linalg.norm(n)
        d = np.abs((p - a) @ n)
        count = int((d < 0.004).sum())
        if count > best[0]:
            best = (count, (a, n))
    a, n = best[1]
    inl = p[np.abs((p - a) @ n) < 0.004]
    centre = inl.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov((inl - centre).T))
    n = vecs[:, 0]  # the least-variance direction: the plane's normal
    if n[2] > 0:  # the camera looks down: up is toward the camera, -z
        n = -n
    return (pts - centre) @ n


def object_mask(height: np.ndarray, click: tuple[int, int], band_m: float = 0.012) -> np.ndarray:
    """The raised body under the click, at the click's height: a stacked object parts from what it stands on."""
    u, v = click
    raised = (height > 0.004) & np.isfinite(height)
    n, lab = cv2.connectedComponents(raised.astype(np.uint8))
    body = lab == lab[v, u]
    h0 = np.nanmedian(height[max(v - 3, 0) : v + 4, max(u - 3, 0) : u + 4])
    mask = body & (np.abs(height - h0) < band_m)
    n, lab = cv2.connectedComponents(mask.astype(np.uint8))
    return lab == lab[v, u]


def tray_tiles(
    height: np.ndarray, pts: np.ndarray, objects: list[np.ndarray], tiles: tuple[int, int]
) -> list[np.ndarray]:
    """The tray's surface (within 12 mm of its plane) minus the objects, in ``tiles`` (columns x rows) of the image."""
    surface = (np.abs(height) < 0.012) & np.isfinite(pts).all(axis=2)
    for m in objects:
        surface &= ~cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    surface = cv2.erode(surface.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    h, w = surface.shape
    ys, xs = np.nonzero(surface)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    out = []
    for r in range(tiles[1]):
        for c in range(tiles[0]):
            tile = np.zeros_like(surface)
            tile[
                y0 + (y1 - y0) * r // tiles[1] : y0 + (y1 - y0) * (r + 1) // tiles[1],
                x0 + (x1 - x0) * c // tiles[0] : x0 + (x1 - x0) * (c + 1) // tiles[0],
            ] = True
            tile &= surface
            if tile.sum() >= 2000:
                out.append(tile)
    return out


def project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack(
            [k[0, 0] * xyz[:, 0] / xyz[:, 2] + k[0, 2], k[1, 1] * xyz[:, 1] / xyz[:, 2] + k[1, 2]], axis=1
        )


def contour_3d(mask: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """The object's outline as 3D points (camera frame, start frame), for drawing it where a pose puts it."""
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea).reshape(-1, 2)
    p = pts[c[:, 1], c[:, 0]]
    inner = pts[mask]
    fill = np.nanmedian(inner, axis=0)
    p = np.where(np.isfinite(p), p, fill)
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording")
    ap.add_argument("out")
    ap.add_argument("--object", action="append", default=[], help="name=u,v on the start frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--tiles", default="3x2")
    ap.add_argument("--config", default=str(pathlib.Path(__file__).resolve().parent / "p2p_groups.yaml"))
    args = ap.parse_args()
    rec, out = pathlib.Path(args.recording), pathlib.Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    k = np.loadtxt(rec / "cam_K.txt")
    times = np.atleast_1d(np.loadtxt(rec / "times.txt"))
    end = len(times) if args.end is None else min(args.end, len(times))
    tiles = tuple(int(v) for v in args.tiles.split("x"))

    rgb0, depth0 = load_frame(rec, args.start)
    pts0 = points_3d(depth0, k)
    h, w = depth0.shape
    region = np.zeros((h, w), bool)
    region[h // 6 : 5 * h // 6, w // 8 : 7 * w // 8] = True
    height0 = tray_height(pts0, region)
    names, clicks = [], []
    for spec in args.object:
        name, uv = spec.split("=")
        names.append(name)
        clicks.append(tuple(int(x) for x in uv.split(",")))
    obj_masks = [object_mask(height0, c) for c in clicks]
    env_masks = tray_tiles(height0, pts0, obj_masks, tiles)
    masks = np.stack(obj_masks + env_masks)
    print(
        f"bodies: {len(obj_masks)} objects ({', '.join(f'{n} {m.sum()} px' for n, m in zip(names, obj_masks, strict=True))}), "
        f"{len(env_masks)} tray tiles",
        flush=True,
    )
    cv2.imwrite(str(out / "bodies.png"), draw_bodies(rgb0, obj_masks, env_masks, names))

    bridge = P2PBridge(pathlib.Path(args.config) if args.config else P2P_CONFIGS["p2p"])
    intr = CameraIntrinsics(fx=float(k[0, 0]), fy=float(k[1, 1]), cx=float(k[0, 2]), cy=float(k[1, 2]))
    reply = bridge.init(rgb0, depth0, masks, intr)
    tracker = GroupTracker()
    n_bodies = len(masks)
    outlines = [contour_3d(m, pts0) for m in obj_masks]
    centres0 = [np.nanmedian(pts0[m], axis=0) for m in obj_masks]
    track_anchor: dict[int, np.ndarray] = {}  # a track's first 3D position, for the objects' own fits
    timeline = []
    t_wall = time.time()
    frame_no = 0
    for kf in range(args.start, end, args.every):
        if kf != args.start:
            rgb, depth = load_frame(rec, kf)
            reply = bridge.step(rgb, depth)
        else:
            rgb, depth = rgb0, depth0
        pts = points_3d(depth, k)
        n_tracks = max(
            int(reply[f"track_idx_{i}"].max()) + 1 if len(reply[f"track_idx_{i}"]) else 0
            for i in range(n_bodies)
        )
        xyz = np.full((n_tracks, 3), np.nan)
        seen = np.zeros(n_tracks, bool)
        owner = np.full(n_tracks, -1)
        for i in range(n_bodies):
            idx, uv = reply[f"track_idx_{i}"].astype(int), reply[f"track_uv_{i}"]
            vis = reply[f"track_vis_{i}"].astype(bool)
            if len(idx) == 0:
                continue
            u = np.clip(np.rint(uv[:, 0]).astype(int), 0, w - 1)
            v = np.clip(np.rint(uv[:, 1]).astype(int), 0, h - 1)
            p = pts[v, u]
            inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
            ok = vis & inside & np.isfinite(p).all(axis=1)
            xyz[idx[ok]] = p[ok]
            seen[idx[ok]] = True
            owner[idx] = i
        for t_id in np.flatnonzero(seen):
            track_anchor.setdefault(int(t_id), xyz[t_id].copy())
        if kf == args.start:
            for i, name in enumerate(names):
                pose = np.eye(4)
                pose[:3, 3] = centres0[i]
                tracker.add_object(name, reply[f"track_idx_{i}"].astype(int), pose)
        else:  # tracks the sampler added since: the object's, by their owner
            for i, name in enumerate(names):
                tracker.objects[name].tracks = reply[f"track_idx_{i}"].astype(int)
        tracker.update(xyz, seen)
        row = {
            "frame": kf,
            "t": float(times[kf]),
            "objects": {},
            "groups": len(tracker.groups),
            "free": int((tracker.group_of == -1).sum()),
        }
        for i, name in enumerate(names):
            obj = tracker.objects[name]
            own = own_fit(obj.tracks, track_anchor, xyz, seen, centres0[i])
            delta = reply[f"delta_{i}"]
            p2p = delta[:3, :3] @ centres0[i] + delta[:3, 3]
            row["objects"][name] = {
                "n_seen": obj.n_seen,
                "group": obj.group,
                "grouped": obj.n_grouped,
                "composed": obj.pose[:3, 3].tolist(),
                "p2p": p2p.tolist(),
                "own": None if own is None else own.tolist(),
                "p2p_lost": bool(reply["objects"][i]["lost"]) if "objects" in reply else None,
            }
        timeline.append(row)
        cv2.imwrite(
            str(out / "frames" / f"{frame_no:06d}.jpg"),
            draw(rgb, k, tracker, xyz, seen, owner, names, outlines, centres0, reply, row),
        )
        frame_no += 1
        if frame_no % 50 == 0:
            print(
                f"frame {kf}: {len(tracker.groups)} groups, {row['free']} free, "
                + ", ".join(f"{n} seen {row['objects'][n]['n_seen']}" for n in names)
                + f", {frame_no / (time.time() - t_wall):.1f} fps",
                flush=True,
            )
    bridge.close()
    (out / "timeline.json").write_text(json.dumps(timeline))
    fps = max(
        1.0, min(30.0, frame_no / max(1e-3, float(times[min(end - 1, len(times) - 1)] - times[args.start])))
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            f"{fps:.2f}",
            "-i",
            str(out / "frames" / "%06d.jpg"),
            "-c:v",
            "libx264",
            "-crf",
            "22",
            "-pix_fmt",
            "yuv420p",
            str(out / "groups.mp4"),
        ],
        check=True,
    )
    report(timeline, names)


def own_fit(tracks, track_anchor, xyz, seen, centre0, min_points: int = 8):
    """Where the object's own visible points put its start-frame centre: a rigid fit from their first positions."""
    ids = [int(t) for t in tracks if int(t) in track_anchor and t < len(seen) and seen[t]]
    if len(ids) < min_points:
        return None
    src = np.asarray([track_anchor[t] for t in ids])
    fit = ransac_fit_rigid(src, xyz[ids], inlier_m=0.008)
    if not fit.ok or fit.n_inliers < min_points:
        return None
    return fit.transform.apply(np.asarray(centre0)[None])[0]


def draw_bodies(rgb, obj_masks, env_masks, names):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for i, m in enumerate(env_masks):
        img[m] = (0.6 * img[m] + 0.4 * np.array(PALETTE[(i + 3) % len(PALETTE)][::-1])).astype(np.uint8)
    for i, (m, name) in enumerate(zip(obj_masks, names, strict=True)):
        img[m] = (0.4 * img[m] + 0.6 * np.array(PALETTE[i][::-1])).astype(np.uint8)
        ys, xs = np.nonzero(m)
        cv2.putText(
            img, name, (int(xs.mean()), int(ys.mean())), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return img


def draw(rgb, k, tracker, xyz, seen, owner, names, outlines, centres0, reply, row):
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    # tracks: filled when seen, hollow at the group's prediction when not; grey when in no group
    for t in range(len(tracker.group_of)):
        g = tracker.group_of[t]
        colour = (160, 160, 160) if g < 0 else PALETTE[g % len(PALETTE)][::-1]
        if seen[t]:
            uv = project(xyz[t], k)[0]
            cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, -1)
        elif g >= 0 and g in tracker.groups:
            p = tracker.groups[g].motion.apply(tracker.anchor[t][None])[0]
            uv = project(p, k)[0]
            if np.isfinite(uv).all():
                cv2.circle(img, (int(uv[0]), int(uv[1])), 3, colour, 1)
    # objects: the outline where the group puts it (solid), where p2p puts it (dashed-ish thin), own fit (cross)
    for i, name in enumerate(names):
        obj = tracker.objects[name]
        o = row["objects"][name]
        colour = PALETTE[i][::-1]
        pose = obj.pose
        # the pose is camera<-object with the object's frame at its start-frame centre: carry the outline through it
        rel = outlines[i] - centres0[i]
        moved = rel @ pose[:3, :3].T + pose[:3, 3]
        uv = project(moved, k)
        if np.isfinite(uv).all():
            cv2.polylines(img, [uv.astype(np.int32).reshape(-1, 1, 2)], True, colour, 2)
        delta = reply[f"delta_{i}"]
        p2p_uv = project(rel @ delta[:3, :3].T + delta[:3, :3] @ centres0[i] + delta[:3, 3], k)
        if np.isfinite(p2p_uv).all():
            cv2.polylines(img, [p2p_uv.astype(np.int32).reshape(-1, 1, 2)], True, (255, 255, 255), 1)
        c = project(pose[:3, 3], k)[0]
        label = f"{name}: seen {o['n_seen']}, group {o['group']}"
        if o["own"] is not None:
            ouv = project(np.asarray(o["own"]), k)[0]
            cv2.drawMarker(img, (int(ouv[0]), int(ouv[1])), colour, cv2.MARKER_CROSS, 14, 2)
            label += f", own fit {np.linalg.norm(np.asarray(o['own']) - pose[:3, 3]) * 1000:.1f} mm off"
        cv2.putText(
            img, label, (int(c[0]) - 40, int(c[1]) - 12 - 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2
        )
    text = f"frame {row['frame']}  groups {row['groups']}  free {row['free']}  " + "  ".join(
        f"g{g}:{d['members']}" for g, d in list(tracker.state()["groups"].items())[:8]
    )
    cv2.putText(img, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(
        img,
        "solid: the group's estimate   thin white: Point2Pose's own fit   cross: the object's own points",
        (8, img.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )
    return img


def report(timeline, names):
    """Per object: the hidden stretches (own points under 8 seen), and at each reappearance the error of the group's
    estimate, Point2Pose's, and a pose held from the last frame the object was seen, against its own points' fit."""
    for name in names:
        rows = [r["objects"][name] for r in timeline]
        print(f"\n{name}:")
        held, hidden_since, errors = None, None, []
        for n, o in enumerate(rows):
            visible = o["own"] is not None
            if visible and hidden_since is None:
                held = np.asarray(o["composed"]), np.asarray(o["p2p"]), np.asarray(o["own"])
            if not visible and hidden_since is None and held is not None:
                hidden_since = n
            if visible and hidden_since is not None:
                own = np.asarray(o["own"])
                err = {
                    "hidden_frames": n - hidden_since,
                    "group_mm": float(np.linalg.norm(np.asarray(o["composed"]) - own) * 1000),
                    "p2p_mm": float(np.linalg.norm(np.asarray(o["p2p"]) - own) * 1000),
                    "held_mm": float(np.linalg.norm(held[2] - own) * 1000),
                }
                errors.append(err)
                print(
                    f"  hidden {err['hidden_frames']} frames (from {timeline[hidden_since]['frame']}): at reappearance "
                    f"group {err['group_mm']:.1f} mm, p2p {err['p2p_mm']:.1f} mm, held {err['held_mm']:.1f} mm"
                )
                hidden_since = None
        if not errors:
            print("  never hidden and seen again")


if __name__ == "__main__":
    main()
