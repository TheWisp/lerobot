"""Phone-size diagnostic of local 3D anchors on moving objects.

A new local frame is explicit after each unsupported interval or manual reseed.
The video does not register A/B/C identities across those cuts.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from lerobot.policies.flash.registration import transform_points

ROOT = Path("/tmp/flash-moving-check")
OUT = ROOT / "FLASH_moving_objects_pose_gizmo.mp4"
W, H, FPS = 1080, 1440, 30
BG = (22, 24, 27)
WHITE = (248, 248, 248)
CYAN = (236, 216, 27)
AXIS_COLORS = ((60, 75, 255), (63, 225, 85), (255, 150, 70))  # X/Y/Z in BGR
DEMO = Path("/home/feit/.cache/huggingface/lerobot/demos/insertion/recording")
GROWTH = Path(
    "/home/feit/.cache/huggingface/lerobot/demos/insertion/flash/v1/tracker-runs/trackeverything-growth-ui-20261021-a"
)
YELLOW = Path("/tmp/flash-multivisible/ring_yellow")
GAMEPAD = ROOT / "held_gamepad"


def txt(im, s, x, y, scale=0.9, color=WHITE, thick=3):
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 5, cv2.LINE_AA)
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def project(points, k):
    a = np.asarray(points)
    z = a[:, 2]
    return np.column_stack((k[0, 0] * a[:, 0] / z + k[0, 2], k[1, 1] * a[:, 1] / z + k[1, 2]))


def triplet_ids(points, indices):
    cloud = points[indices]
    depths = cloud[:, 2]
    median = np.median(depths)
    mad = np.median(np.abs(depths - median))
    if mad > 0:
        consistent = np.abs(depths - median) <= 3 * 1.4826 * mad
        if np.count_nonzero(consistent) >= 3:
            indices = indices[consistent]
            cloud = points[indices]
    center = np.median(cloud, axis=0)
    a = int(np.argmax(np.linalg.norm(cloud - center, axis=1)))
    b = int(np.argmax(np.linalg.norm(cloud - cloud[a], axis=1)))
    axis = cloud[b] - cloud[a]
    axis /= np.linalg.norm(axis)
    dif = cloud - cloud[a]
    distance = np.linalg.norm(dif - np.outer(dif @ axis, axis), axis=1)
    c = int(np.argmax(distance))
    return indices[[a, b, c]]


def mask_contour(im, mask, offset=(0, 0), scale=1.0):
    if mask is None:
        return
    con, _ = cv2.findContours(mask.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = offset
    for p in con:
        q = np.rint((p.astype(float) - np.array([[[ox, oy]]])) * scale).astype("int32")
        cv2.drawContours(im, [q], -1, WHITE, 2, cv2.LINE_AA)


def marks(im, uv, offset=(0, 0), scale=1.0, support=None, labels=False):
    if uv is None:
        return
    q = (np.asarray(uv) - np.array(offset)) * scale
    if not np.isfinite(q).all():
        return
    q = np.rint(q).astype(int)
    for n, (x, y) in enumerate(q):
        if not (-35 < x < im.shape[1] + 35 and -35 < y < im.shape[0] + 35):
            continue
        r = 13 if labels else 7
        cv2.circle(im, (int(x), int(y)), r, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(
            im, (int(x), int(y)), r - 3, WHITE, -1 if support is None or support[n] else 2, cv2.LINE_AA
        )
        if labels:
            txt(im, "ABC"[n], int(x) + 17, int(y) - 11, 0.85, WHITE, 3)


def gizmo_points(landmarks):
    """A is the origin; AB and the component of AC normal to AB define X/Y."""
    a, b, c = np.asarray(landmarks, dtype=float)
    x = b - a
    x_norm = np.linalg.norm(x)
    if not np.isfinite(x_norm) or x_norm <= 0:
        return None
    x /= x_norm
    y = c - a - np.dot(c - a, x) * x
    y_norm = np.linalg.norm(y)
    if not np.isfinite(y_norm) or y_norm <= 0:
        return None
    y /= y_norm
    z = np.cross(x, y)
    # Keep all three axes readable while their scale follows the local landmark spread.
    axis_length = 0.65 * max(np.linalg.norm(b - a), np.linalg.norm(c - a))
    return np.stack((a, a + axis_length * x, a + axis_length * y, a + axis_length * z))


def draw_gizmo(im, landmarks, k, offset=(0, 0), scale=1.0, labels=False):
    axes = gizmo_points(landmarks)
    if axes is None or np.any(axes[:, 2] <= 0):
        return
    q = np.rint((project(axes, k) - np.asarray(offset)) * scale).astype(int)
    origin = tuple(q[0])
    for n, end in enumerate(q[1:]):
        tip = tuple(end)
        cv2.arrowedLine(im, origin, tip, (0, 0, 0), 7 if labels else 4, cv2.LINE_AA, tipLength=0.25)
        cv2.arrowedLine(im, origin, tip, AXIS_COLORS[n], 4 if labels else 2, cv2.LINE_AA, tipLength=0.25)
        if labels:
            txt(im, "XYZ"[n], tip[0] + 7, tip[1] - 7, 0.75, AXIS_COLORS[n], 2)


def draw(rgb, mask, fit_xyz, raw_xyz, raw_support, title, k, side=235):
    fit_uv = None if fit_xyz is None else project(fit_xyz, k)
    raw_uv = None if raw_xyz is None else project(raw_xyz, k)
    full = rgb.copy()
    mask_contour(full, mask)
    if fit_xyz is not None:
        draw_gizmo(full, fit_xyz, k)
    marks(full, fit_uv)
    canvas = np.full((H, W, 3), BG, "uint8")
    txt(canvas, title, 34, 80, 0.91)
    cv2.line(canvas, (34, 106), (1046, 106), CYAN, 3)
    canvas[130:738] = cv2.resize(full, (W, 608), interpolation=cv2.INTER_AREA)
    yy, xx = np.nonzero(mask) if mask is not None else (np.array([]), np.array([]))
    if len(xx):
        cx, cy = int(np.mean(xx)), int(np.mean(yy))
    elif fit_uv is not None:
        cx, cy = np.rint(np.mean(fit_uv, axis=0)).astype(int)
    elif raw_uv is not None and np.isfinite(raw_uv).all():
        cx, cy = np.rint(np.mean(raw_uv, axis=0)).astype(int)
    else:
        cx, cy = rgb.shape[1] // 2, rgb.shape[0] // 2
    left = max(0, min(rgb.shape[1] - side, cx - side // 2))
    top = max(0, min(rgb.shape[0] - side, cy - side // 2))
    fit_label = (
        "FIT (UNVERIFIED)" if "UNVERIFIED" in title else ("RIGID FIT" if fit_uv is not None else "NO POSE")
    )
    for ri, (label, uv, xyz, support) in enumerate(
        ((fit_label, fit_uv, fit_xyz, None), ("SAME RAW IDS", raw_uv, raw_xyz, raw_support))
    ):
        x0 = 20 + ri * 530
        cv2.rectangle(canvas, (x0, 770), (x0 + 510, 1395), (39, 42, 45), -1)
        txt(canvas, label, x0 + 17, 815, 0.81)
        crop = cv2.resize(
            rgb[top : top + side, left : left + side], (480, 480), interpolation=cv2.INTER_LINEAR
        )
        mask_contour(crop, mask, (left, top), 480 / side)
        if ri == 0 and xyz is not None:
            draw_gizmo(crop, xyz, k, (left, top), 480 / side, True)
        marks(crop, uv, (left, top), 480 / side, support, True)
        canvas[850:1330, x0 + 15 : x0 + 495] = crop
    return canvas


def read_mask(path):
    if not path.exists():
        return None
    a = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return None if a is None else a > 0


def dowel_frames():
    local = np.load(ROOT / "dowel_rotation_segments.npz")
    segments = json.loads((ROOT / "dowel_rotation_segments.json").read_text())
    scene = np.load(GROWTH / "scene.npz")
    k = np.loadtxt(DEMO / "cam_K.txt").reshape(3, 3)
    for frame in range(120, 371):
        i = frame - 100
        rgb = cv2.imread(str(DEMO / "rgb" / f"{frame:06d}.jpg"))
        mask = read_mask(GROWTH / "segmentation/masks/0" / f"{frame:06d}.png")
        segment = int(local["segment_by_frame"][i])
        if segment:
            item = segments[segment - 1]
            ids = np.array(item["anchor_track_ids"])
            fit_xyz = local["anchors_by_frame"][i]
            raw_xyz = scene["query_coords_m"][i, ids]
            support = local["observed_by_frame"][i].tolist()
            title = f"DOWEL ROTATION - LOCAL FRAME {segment}"
            if segment > 1 and frame - item["frames"][0] < 12:
                title = f"DOWEL - UNLINKED RESEED {segment}"
            if not any(support):
                if segment > 1 and frame - item["frames"][0] < 12:
                    title = f"DOWEL - RESEED {segment} UNVERIFIED"
                else:
                    title = f"DOWEL - FRAME {segment} UNVERIFIED"
        else:
            previous = next((x for x in reversed(segments) if x["frames"][0] < frame), None)
            ids = np.array(previous["anchor_track_ids"]) if previous is not None else np.array([], int)
            raw_xyz = scene["query_coords_m"][i, ids] if len(ids) == 3 else None
            support = local["support"][i, ids].tolist() if len(ids) == 3 else None
            fit_xyz = None
            title = "DOWEL ROTATION - TRACK LOST"
        yield frame, draw(rgb, mask, fit_xyz, raw_xyz, support, title, k, 225)


def yellow_frames():
    scene = np.load(YELLOW / "scene.npz")
    k = np.loadtxt(YELLOW / "recording/cam_K.txt").reshape(3, 3)
    role = 1
    ids0 = np.flatnonzero((scene["query_roles"] == "block") & scene["visible_surface_support"][0])
    ids = triplet_ids(scene["query_coords_m"][0], ids0)
    (ROOT / "yellow_anchor_ids.json").write_text(json.dumps(ids.tolist()))
    base = scene["query_coords_m"][0, ids]
    origin = scene["poses"][role, 0]
    for frame in range(75):
        rgb = cv2.imread(str(YELLOW / "recording/rgb" / f"{frame:06d}.jpg"))
        mask = read_mask(YELLOW / "masks/1" / f"{frame:06d}.png")
        pose = scene["poses"][role, frame] @ np.linalg.inv(origin)
        fit_xyz = None if scene["lost"][role, frame] else transform_points(pose, base)
        raw_xyz = scene["query_coords_m"][frame, ids]
        support = scene["visible_surface_support"][frame, ids].tolist()
        yield frame, draw(rgb, mask, fit_xyz, raw_xyz, support, "YELLOW LEGO - MOVING ON TRAY", k, 225)


def gamepad_frames():
    k = np.loadtxt(GAMEPAD / "recording/cam_K.txt").reshape(3, 3)
    stages = [
        (90, 119, GAMEPAD / "reseed090.npz"),
        (120, 144, GAMEPAD / "reseed120.npz"),
        (145, 170, GAMEPAD / "reseed145.npz"),
    ]
    for stage, (lo, hi, path) in enumerate(stages, 1):
        scene = np.load(path)
        first = int(scene["frames"][0])
        ids0 = np.flatnonzero(scene["visible_surface_support"][0])
        ids = triplet_ids(scene["query_coords_m"][0], ids0)
        base = scene["query_coords_m"][0, ids]
        origin = scene["poses"][0]
        for frame in range(lo, hi + 1):
            i = frame - first
            rgb = cv2.imread(str(GAMEPAD / "recording/rgb" / f"{frame:06d}.jpg"))
            mask = read_mask(GAMEPAD / "masks/0" / f"{frame:06d}.png")
            fit_xyz = (
                None
                if scene["lost"][i]
                else transform_points(scene["poses"][i] @ np.linalg.inv(origin), base)
            )
            raw_xyz = scene["query_coords_m"][i, ids]
            support = scene["visible_surface_support"][i, ids].tolist()
            title = f"HELD GAMEPAD - LOCAL FRAME {stage}"
            if stage > 1 and frame - lo < 12:
                title = f"GAMEPAD - UNLINKED RESEED {stage}"
            yield frame, draw(rgb, mask, fit_xyz, raw_xyz, support, title, k, 240)


def main():
    proc = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{W}x{H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "22",
            "-profile:v",
            "main",
            "-level",
            "4.1",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUT),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for name, generator, shots in [
            ("dowel", dowel_frames(), (120, 172, 190, 217, 259, 267, 300, 330, 370)),
            ("yellow", yellow_frames(), (0, 37, 58, 74)),
            ("gamepad", gamepad_frames(), (90, 119, 120, 144, 145, 170)),
        ]:
            count = 0
            for frame, image in generator:
                if frame in shots:
                    cv2.imwrite(str(ROOT / f"{name}-{frame:03d}.jpg"), image)
                for _ in range(2):
                    proc.stdin.write(image.tobytes())
                count += 1
            print(name, count, flush=True)
    finally:
        proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
