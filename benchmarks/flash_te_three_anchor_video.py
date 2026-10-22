"""Render three material-point anchors carried through the insertion demo.

This offline illustration uses the saved TrackEverything result. Three
non-collinear surface tracks at frame 361 define a location and orientation.
Every later frame fits a rigid transform from *all* visible, consistent tracks;
the transform carries an anchor when its own TE track is occluded or rejected.
It does not establish demo-to-live point correspondence or change FLASH control.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flash_cross_episode_trackeverything import _project
from flash_te_insertion_relation_video import (
    BG,
    DEMO,
    DOWEL_COLOR,
    FPS,
    GROWTH,
    HEIGHT,
    INLIER_BAND_M,
    MUTED,
    PANEL,
    WHITE,
    WIDTH,
    collect_group,
    f,
)
from PIL import Image, ImageDraw

FIRST, LAST = 361, 535
ANCHOR_IDS = (295, 252, 261)  # A, B, C: validated TE IDs in this saved demo.
ANCHOR_COLORS = ((255, 190, 76), (255, 113, 217), (147, 221, 118))
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/proofs/flash-cross-episode/te-demo-map/three-point-demo-frame.mp4"
)


def camera_to_local(p: np.ndarray, anchor_fit: dict) -> np.ndarray:
    return (p - anchor_fit["shift"]) @ anchor_fit["rotation"]


def local_to_camera(p: np.ndarray, frame_fit: dict) -> np.ndarray:
    return p @ frame_fit["rotation"].T + frame_fit["shift"]


def make_records(scene: dict, k: np.ndarray) -> tuple[list[dict], dict]:
    fits, supports = collect_group("dowel", scene, k)
    base = scene["query_coords_m"][FIRST - int(scene["frames"][0]), list(ANCHOR_IDS)]
    local = camera_to_local(base, fits[FIRST])
    edges = np.linalg.norm(base[:, None, :] - base[None, :, :], axis=-1)
    triangle_area = float(np.linalg.norm(np.cross(base[1] - base[0], base[2] - base[0])))
    records = []
    for frame in range(FIRST, LAST + 1):
        fit = fits[frame]
        item = {
            "frame": frame,
            "fit_supported": fit["supported"],
            "fit_shared": fit["shared"],
            "fit_inliers": fit["inliers"],
            "fit_median_inlier_mm": fit["median_inlier_mm"],
            "support": supports[frame],
        }
        if fit["rotation"] is not None:
            anchors = local_to_camera(local, fit)
            primary = anchors[1] - anchors[0]
            primary /= np.linalg.norm(primary)
            secondary = anchors[2] - anchors[0]
            secondary -= np.dot(secondary, primary) * primary
            secondary /= np.linalg.norm(secondary)
            normal = np.cross(primary, secondary)
            index = frame - int(scene["frames"][0])
            raw = scene["query_coords_m"][index, list(ANCHOR_IDS)]
            residual_mm = np.linalg.norm(raw - anchors, axis=1) * 1000
            item.update(
                anchor_xyz_m=anchors.tolist(),
                anchor_px=_project(anchors, k).tolist(),
                raw_anchor_px=_project(raw, k).tolist(),
                raw_supported=[bool(supports[frame][i]) for i in ANCHOR_IDS],
                raw_residual_mm=residual_mm.tolist(),
                origin_camera_mm=(anchors[0] * 1000).tolist(),
                direction_camera=primary.tolist(),
                second_axis_camera=secondary.tolist(),
                third_axis_camera=normal.tolist(),
            )
        records.append(item)
    metadata = {
        "demo": str(DEMO),
        "trackeverything": str(GROWTH / "scene.npz"),
        "frames": [FIRST, LAST],
        "fps": FPS,
        "anchor_track_ids": {name: int(track_id) for name, track_id in zip("ABC", ANCHOR_IDS, strict=True)},
        "anchor_selection": "Three separated, non-collinear surface tracks from frame 361 whose co-motion was checked across the saved demo segment; not semantic task landmarks.",
        "initial_pairwise_distances_mm": (edges * 1000).tolist(),
        "initial_triangle_area_mm2": triangle_area * 1e6 / 2,
        "frame_rule": "origin=A; X=unit(B-A); Y=unit((C-A)-dot(C-A,X)X); Z=X cross Y",
        "propagation": "Each frame fits a rigid transform using all currently supported TrackEverything points, then applies it to fixed A/B/C reference coordinates.",
        "limitation": "This is within-demo propagation only. A live run still needs demonstrated cross-episode material-point correspondence; a rigid fit can also be wrong under occlusion.",
    }
    return records, metadata


def crop_xy(point: list[float]) -> tuple[float, float]:
    crop_box = (100, 50, 375, 320)
    return (
        (point[0] - crop_box[0]) * 715 / (crop_box[2] - crop_box[0]),
        (point[1] - crop_box[1]) * 702 / (crop_box[3] - crop_box[1]),
    )


def draw_cross(draw: ImageDraw.ImageDraw, x: float, y: float, color: tuple[int, int, int]) -> None:
    draw.line((x - 9, y - 9, x + 9, y + 9), fill=color, width=3)
    draw.line((x - 9, y + 9, x + 9, y - 9), fill=color, width=3)


def render(record: dict, scene: dict, k: np.ndarray) -> Image.Image:
    frame = record["frame"]
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((45, 23), "Three tracked points → location + direction", font=f(35, True), fill=WHITE)
    draw.text(
        (46, 77),
        f"Insertion demo  •  frame {frame}  •  {(frame - FIRST) / FPS:.2f} s",
        font=f(22),
        fill=MUTED,
    )
    draw.rounded_rectangle((43, 126, 792, 858), radius=18, fill=PANEL)
    rgb = cv2.cvtColor(cv2.imread(str(DEMO / "rgb" / f"{frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
    camera = Image.fromarray(rgb).crop((100, 50, 375, 320)).resize((715, 702), Image.Resampling.BILINEAR)
    cam = ImageDraw.Draw(camera)
    points = scene["query_coords_m"][frame - int(scene["frames"][0])]
    visible_ids = np.flatnonzero(record["support"])
    for pixel in _project(points[visible_ids], k):
        x, y = crop_xy(pixel)
        if 0 <= x < 715 and 0 <= y < 702:
            cam.ellipse((x - 3, y - 3, x + 3, y + 3), fill=DOWEL_COLOR, outline=(7, 19, 24), width=1)
    if "anchor_px" in record:
        pixels = [crop_xy(p) for p in record["anchor_px"]]
        ax, ay = pixels[0]
        bx, by = pixels[1]
        cx, cy = pixels[2]
        cam.line((ax, ay, bx, by), fill=ANCHOR_COLORS[1], width=5)
        cam.line((ax, ay, cx, cy), fill=ANCHOR_COLORS[2], width=3)
        cam.line((bx, by, cx, cy), fill=(235, 230, 244), width=2)
        cam.ellipse((bx - 5, by - 5, bx + 5, by + 5), fill=ANCHOR_COLORS[1])
        for n, ((x, y), color) in enumerate(zip(pixels, ANCHOR_COLORS, strict=True)):
            supported = record["raw_supported"][n]
            matched = supported and record["raw_residual_mm"][n] <= INLIER_BAND_M * 1000
            cam.ellipse(
                (x - 11, y - 11, x + 11, y + 11), fill=color if matched else None, outline=color, width=4
            )
            cam.text(
                (min(x + 14, 635), max(y - 29, 5)),
                "ABC"[n],
                font=f(23, True),
                fill=color,
                stroke_width=2,
                stroke_fill=(0, 0, 0),
            )
            if not matched:
                rx, ry = crop_xy(record["raw_anchor_px"][n])
                if 0 <= rx < 715 and 0 <= ry < 702:
                    draw_cross(cam, rx, ry, (249, 77, 77))
        cam.text(
            (min((ax + bx) / 2 + 10, 560), max((ay + by) / 2 - 30, 3)),
            "direction A → B",
            font=f(18, True),
            fill=ANCHOR_COLORS[1],
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    cam.rounded_rectangle((12, 641, 657, 693), radius=9, fill=(17, 26, 34))
    cam.text(
        (26, 651),
        "cyan: TE points  •  filled: agrees with fit  •  hollow: inferred",
        font=f(16, True),
        fill=WHITE,
    )
    canvas.paste(camera, (60, 142))

    draw.rounded_rectangle((817, 126, 1557, 858), radius=18, fill=PANEL)
    draw.text((845, 149), "A, B, C define one rigid frame", font=f(27, True), fill=WHITE)
    draw.text((845, 193), "Origin = A", font=f(24, True), fill=ANCHOR_COLORS[0])
    draw.text((845, 230), "Direction = unit(B − A)", font=f(24, True), fill=ANCHOR_COLORS[1])
    draw.text((845, 267), "C fixes rotation about that direction", font=f(20), fill=ANCHOR_COLORS[2])
    if "origin_camera_mm" in record:
        origin = record["origin_camera_mm"]
        axis = record["direction_camera"]
        draw.text((845, 319), "A location in camera frame (mm)", font=f(20), fill=MUTED)
        draw.text(
            (845, 350),
            f"[{origin[0]:+.1f}, {origin[1]:+.1f}, {origin[2]:+.1f}]",
            font=f(30, True),
            fill=WHITE,
        )
        draw.text((845, 407), "A → B direction (unit vector)", font=f(20), fill=MUTED)
        draw.text(
            (845, 438), f"[{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]", font=f(30, True), fill=WHITE
        )
    draw.line((845, 504, 1528, 504), fill=(65, 78, 92), width=2)
    draw.text((845, 522), "Anchor status this frame", font=f(24, True), fill=WHITE)
    for n, name in enumerate("ABC"):
        y = 567 + n * 52
        supported = record.get("raw_supported", [False] * 3)[n]
        residual = record.get("raw_residual_mm", [None] * 3)[n]
        consistent = supported and residual is not None and residual <= INLIER_BAND_M * 1000
        status = (
            f"observed  •  {residual:.1f} mm from fit"
            if consistent
            else "inferred; red X marks rejected raw track"
        )
        color = ANCHOR_COLORS[n]
        draw.ellipse((845, y + 8, 858, y + 21), fill=color if consistent else None, outline=color, width=3)
        draw.text((872, y), f"{name}: {status}", font=f(19, True), fill=color if not consistent else WHITE)
    draw.line((845, 733, 1528, 733), fill=(65, 78, 92), width=2)
    draw.text(
        (845, 751),
        f"Dowel fit: {record['fit_inliers']}/{record['fit_shared']} co-moving points",
        font=f(22, True),
        fill=WHITE,
    )
    draw.text(
        (845, 788), "A/B/C can be hidden; all visible points carry their frame.", font=f(18), fill=MUTED
    )
    if frame >= 500:
        draw.rounded_rectangle((1353, 148, 1530, 178), radius=6, fill=(84, 47, 34))
        draw.text((1360, 153), "FIT UNSTABLE", font=f(16, True), fill=ANCHOR_COLORS[0])
    draw.text(
        (55, 867),
        "A, B, C are arbitrary surface anchors, not object centers or task landmarks. This shows demo-time tracking only.",
        font=f(17),
        fill=MUTED,
    )
    return canvas


def main() -> None:
    k = np.loadtxt(DEMO / "cam_K.txt").reshape(3, 3)
    with np.load(GROWTH / "scene.npz", allow_pickle=False) as saved:
        scene = {key: saved[key] for key in saved.files}
    records, metadata = make_records(scene, k)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT.with_suffix(".json")
    json_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "frames": [{key: value for key, value in rec.items() if key != "support"} for rec in records],
            },
            indent=2,
        )
        + "\n"
    )
    ffmpeg = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{WIDTH}x{HEIGHT}",
            "-framerate",
            str(FPS),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            "29",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ],
        stdin=subprocess.PIPE,
    )
    assert ffmpeg.stdin is not None
    try:
        for record in records:
            ffmpeg.stdin.write(render(record, scene, k).tobytes())
    finally:
        ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(OUTPUT)
    print(json_path)


if __name__ == "__main__":
    main()
