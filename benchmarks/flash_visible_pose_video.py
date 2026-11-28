from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flash_rgbd_registration import cloud

from lerobot.policies.flash.registration import transform_points

ROOT = Path("/tmp/flash-multivisible")
OUT = ROOT / "FLASH_visible_pose_tracking_examples.mp4"
W, H, FPS = 1080, 1440, 30
BG = (22, 24, 27)
WHITE = (248, 248, 248)
CYAN = (236, 216, 27)
COLORS = [(60, 75, 255), (63, 225, 85), (255, 150, 70)]  # BGR: A red, B green, C blue
JOBS = [("moving_tray_ring_cube", 75), ("gamepad_cube_act", 36), ("gamepad_blue_demo", 31)]
NAMES = {"ring": "RING", "cube": "CUBE", "gamepad": "GAMEPAD", "blue_ring": "BLUE RING"}


def txt(im, s, x, y, scale=1.0, color=WHITE, thick=3):
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 5, cv2.LINE_AA)
    cv2.putText(im, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def project(xyz, k):
    xyz = np.asarray(xyz)
    z = xyz[:, 2]
    uv = np.column_stack((k[0, 0] * xyz[:, 0] / z + k[0, 2], k[1, 1] * xyz[:, 1] / z + k[1, 2]))
    return uv


def landmarks(points):
    # Define one material triangle from the initial RGB-D surface. Never reselect it.
    if len(points) > 300:
        points = points[np.linspace(0, len(points) - 1, 300).astype(int)]
    center = np.median(points, axis=0)
    a = int(np.argmax(np.linalg.norm(points - center, axis=1)))
    b = int(np.argmax(np.linalg.norm(points - points[a], axis=1)))
    direction = points[b] - points[a]
    direction /= np.linalg.norm(direction)
    perpendicular = (points - points[a]) - np.outer((points - points[a]) @ direction, direction)
    c = int(np.argmax(np.linalg.norm(perpendicular, axis=1)))
    return points[[a, b, c]]


def read(name):
    base = ROOT / name
    with np.load(base / "scene.npz", allow_pickle=False) as data:
        scene = {k: data[k] for k in data.files}
    roles = [str(r) for r in scene["roles"]]
    k = np.loadtxt(base / "recording" / "cam_K.txt").reshape(3, 3)
    depth = cv2.imread(str(base / "recording" / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    anchors = {}
    for ri, role in enumerate(roles):
        mask = cv2.imread(str(base / "masks" / str(ri) / "000000.png"), cv2.IMREAD_GRAYSCALE) > 0
        anchors[role] = landmarks(cloud(depth, mask, k, 3))
    return base, scene, roles, k, anchors


def overlay(im, mask, uv, offset=(0, 0), scale=1.0, large=False, support=None):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ox, oy = offset
    if contours:
        moved = [
            np.rint((p.astype(np.float32) - np.array([[[ox, oy]]])) * scale).astype(np.int32)
            for p in contours
        ]
        cv2.drawContours(im, moved, -1, WHITE, 2 if large else 1, cv2.LINE_AA)
    q = (uv - np.array([ox, oy])) * scale
    if not np.isfinite(q).all():
        return
    q = np.rint(q).astype(int)
    cv2.polylines(im, [q], True, CYAN, 4 if large else 2, cv2.LINE_AA)
    p0 = tuple(q[0])
    p1 = tuple(q[1])
    # A->B is a material direction, not a newly chosen image-space vector.
    cv2.arrowedLine(im, p0, p1, (80, 245, 240), 4 if large else 2, cv2.LINE_AA, tipLength=0.18)
    for j, p in enumerate(q):
        x, y = int(p[0]), int(p[1])
        r = 13 if large else 7
        if -30 < x < im.shape[1] + 30 and -30 < y < im.shape[0] + 30:
            cv2.circle(im, (x, y), r, (0, 0, 0), -1, cv2.LINE_AA)
            if support is None or support[j]:
                cv2.circle(im, (x, y), r - 3, COLORS[j], -1, cv2.LINE_AA)
            else:
                cv2.circle(im, (x, y), r - 3, COLORS[j], 2, cv2.LINE_AA)
            if large:
                txt(im, "ABC"[j], x + 17, y - 11, 0.85, COLORS[j], 3)


def frame(prepared, i):
    base, scene, roles, k, anchors = prepared
    rgb = cv2.imread(str(base / "recording" / "rgb" / f"{i:06d}.jpg"))
    masks = {}
    uvs = {}
    for ri, role in enumerate(roles):
        masks[role] = cv2.imread(str(base / "masks" / str(ri) / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
        pose = scene["poses"][ri, i] @ np.linalg.inv(scene["poses"][ri, 0])
        uvs[role] = project(transform_points(pose, anchors[role]), k)
    full = rgb.copy()
    for role in roles:
        overlay(full, masks[role], uvs[role])
    canvas = np.full((H, W, 3), BG, np.uint8)
    txt(canvas, "POSE CHECK: FIXED 3D LANDMARKS", 34, 80, 0.94, WHITE, 3)
    cv2.line(canvas, (34, 106), (1046, 106), CYAN, 3)
    scene_img = cv2.resize(full, (W, 608), interpolation=cv2.INTER_AREA)
    canvas[130:738] = scene_img
    for ri, role in enumerate(roles):
        x0 = 20 + ri * 530
        cv2.rectangle(canvas, (x0, 770), (x0 + 510, 1395), (39, 42, 45), -1)
        txt(canvas, NAMES.get(role, role.upper()), x0 + 17, 815, 0.91, WHITE, 3)
        yy, xx = np.nonzero(masks[role])
        cx = int(np.mean(xx)) if len(xx) else rgb.shape[1] // 2
        cy = int(np.mean(yy)) if len(yy) else rgb.shape[0] // 2
        side = 235
        left = max(0, min(rgb.shape[1] - side, cx - side // 2))
        top = max(0, min(rgb.shape[0] - side, cy - side // 2))
        crop = rgb[top : top + side, left : left + side].copy()
        crop = cv2.resize(crop, (480, 480), interpolation=cv2.INTER_LINEAR)
        # Inset image coordinates use this same fixed image-space crop transform.
        overlay(crop, masks[role], uvs[role], (left, top), 480 / side, True)
        canvas[850:1330, x0 + 15 : x0 + 495] = crop
    return canvas


DEMO = Path("/home/feit/.cache/huggingface/lerobot/demos/insertion/recording")
GROWTH = Path(
    "/home/feit/.cache/huggingface/lerobot/demos/insertion/flash/v1/tracker-runs/trackeverything-growth-ui-20261021-a"
)
RECORDS = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "docs/proofs/flash-cross-episode/te-demo-map/three-point-demo-frame.json"
    ).read_text()
)["frames"]


def dowel_frame(rec):
    fi = rec["frame"]
    rgb = cv2.imread(str(DEMO / "rgb" / f"{fi:06d}.jpg"))
    mask = cv2.imread(str(GROWTH / "segmentation/masks/0" / f"{fi:06d}.png"), cv2.IMREAD_GRAYSCALE) > 0
    uv = np.asarray(rec["anchor_px"], dtype=float)
    full = rgb.copy()
    overlay(full, mask, uv)
    canvas = np.full((H, W, 3), BG, np.uint8)
    txt(canvas, "HELD DOWEL: FIT VS POINT TRACKER", 34, 80, 0.92, WHITE, 3)
    cv2.line(canvas, (34, 106), (1046, 106), CYAN, 3)
    canvas[130:738] = cv2.resize(full, (W, 608), interpolation=cv2.INTER_AREA)
    raw = np.asarray(rec["raw_anchor_px"], dtype=float)
    for ri, (name, img, msk, anchors, support) in enumerate(
        (
            ("RIGID FIT", rgb, mask, uv, None),
            ("SAME THREE RAW IDS", rgb, mask, raw, rec["raw_supported"]),
        )
    ):
        x0 = 20 + ri * 530
        cv2.rectangle(canvas, (x0, 770), (x0 + 510, 1395), (39, 42, 45), -1)
        txt(canvas, name, x0 + 17, 815, 0.78, WHITE, 3)
        yy, xx = np.nonzero(msk)
        cx = int(np.mean(xx)) if len(xx) else img.shape[1] // 2
        cy = int(np.mean(yy)) if len(yy) else img.shape[0] // 2
        side = 215
        left = max(0, min(img.shape[1] - side, cx - side // 2))
        top = max(0, min(img.shape[0] - side, cy - side // 2))
        crop = cv2.resize(
            img[top : top + side, left : left + side], (480, 480), interpolation=cv2.INTER_LINEAR
        )
        overlay(crop, msk, anchors, (left, top), 480 / side, True, support)
        canvas[850:1330, x0 + 15 : x0 + 495] = crop
    return canvas


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
        for rec in RECORDS[::2]:
            rendered = dowel_frame(rec)
            if rec["frame"] in (361, 421, 481, 521):
                cv2.imwrite(str(ROOT / f"held-dowel-pose-{rec['frame']}.jpg"), rendered)
            for _ in range(2):
                proc.stdin.write(rendered.tobytes())
        print("held_dowel", len(RECORDS[::2]), flush=True)
        for name, limit in JOBS:
            prepared = read(name)
            for i in range(limit):
                rendered = frame(prepared, i)
                if i in (0, limit // 2, limit - 1):
                    cv2.imwrite(str(ROOT / f"{name}-pose-{i:03d}.jpg"), rendered)
                for _ in range(2):
                    proc.stdin.write(rendered.tobytes())
            print(name, limit, flush=True)
    finally:
        proc.stdin.close()
    if proc.wait():
        raise RuntimeError("ffmpeg failed")
    print(OUT, OUT.stat().st_size)


if __name__ == "__main__":
    main()
