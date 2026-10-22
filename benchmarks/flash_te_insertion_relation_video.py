"""Illustrate a point-group spatial action from the saved insertion demo.

This is an offline explanation, not the FLASH controller or a pose ground truth.
It uses the already-saved TrackEverything point tracks and RGB-D frames. A human
annotates one visible dowel end and one ring-hole location at frame 320; later
landmark positions are inferred from rigid fits to observed point groups.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import cv2
import numpy as np
from flash_cross_episode_trackeverything import _project, _rigid_fit
from flash_te_demo_map_consistency import supported_ids
from PIL import Image, ImageDraw, ImageFont

DEMO = Path("/home/feit/.cache/huggingface/lerobot/demos/insertion/recording")
GROWTH = Path(
    "/home/feit/.cache/huggingface/lerobot/demos/insertion/flash/v1/tracker-runs/"
    "trackeverything-growth-ui-20261021-a"
)
OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/proofs/flash-cross-episode/te-demo-map/insertion-relative-motion.mp4"
)
ANCHOR, END, FPS = 320, 535, 30
WIDTH, HEIGHT = 1600, 900
INLIER_BAND_M = 0.005  # Visualization diagnostic, not a FLASH acceptance threshold.
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
BG = (14, 20, 29)
PANEL = (24, 33, 45)
WHITE = (238, 242, 247)
MUTED = (158, 172, 189)
DOWEL_COLOR = (60, 220, 247)
RING_COLOR = (115, 238, 162)
TIP_COLOR = (255, 190, 76)
HOLE_COLOR = (239, 115, 219)


def f(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def point_from_depth(pixel: tuple[int, int], depth: np.ndarray, k: np.ndarray) -> np.ndarray:
    x, y = pixel
    z = float(depth[y, x]) * 0.001
    if z <= 0:
        raise ValueError(f"No aligned depth at {pixel}")
    return np.array([(x - k[0, 2]) * z / k[0, 0], (y - k[1, 2]) * z / k[1, 1], z])


def fit_group(
    reference: np.ndarray, measured: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a rigid map; retain every point only if its observed residual agrees."""
    if len(reference) < 3:
        raise ValueError("Fewer than three visible shared points")
    rng = np.random.default_rng(seed)
    best = np.zeros(len(reference), dtype=bool)
    for _ in range(120):
        sample = rng.choice(len(reference), 3, replace=False)
        if np.linalg.matrix_rank(reference[sample] - reference[sample].mean(axis=0)) < 2:
            continue
        rotation, shift = _rigid_fit(reference[sample], measured[sample])
        error = np.linalg.norm(reference @ rotation.T + shift - measured, axis=1)
        candidate = error <= INLIER_BAND_M
        if candidate.sum() > best.sum():
            best = candidate
    if best.sum() < 3:
        raise ValueError("No co-moving point group")
    rotation, shift = _rigid_fit(reference[best], measured[best])
    error = np.linalg.norm(reference @ rotation.T + shift - measured, axis=1)
    return rotation, shift, error <= INLIER_BAND_M


def collect_group(role: str, scene: dict, k: np.ndarray) -> tuple[dict[int, dict], dict[int, np.ndarray]]:
    mask_dir = GROWTH / "segmentation/masks" / ("0" if role == "dowel" else "1")
    base_index = ANCHOR - int(scene["frames"][0])
    support, _, _ = supported_ids(ANCHOR, scene, DEMO, mask_dir, k, role)
    canonical = np.full((len(support), 3), np.nan)
    canonical[support] = scene["query_coords_m"][base_index, support]
    output = {}
    supports = {}
    for frame in range(ANCHOR, END + 1):
        index = frame - int(scene["frames"][0])
        current_support, _, _ = supported_ids(frame, scene, DEMO, mask_dir, k, role)
        supports[frame] = current_support
        measured = scene["query_coords_m"][index]
        shared = current_support & np.isfinite(canonical).all(axis=1)
        ids = np.flatnonzero(shared)
        try:
            if frame == ANCHOR:
                rotation, shift = np.eye(3), np.zeros(3)
                inlier = np.ones(len(ids), dtype=bool)
            else:
                rotation, shift, inlier = fit_group(canonical[ids], measured[ids], frame)
            fit_residual = np.linalg.norm(canonical[ids] @ rotation.T + shift - measured[ids], axis=1)
            record = {
                "rotation": rotation,
                "shift": shift,
                "supported": int(current_support.sum()),
                "shared": len(ids),
                "inliers": int(inlier.sum()),
                "median_inlier_mm": float(np.median(fit_residual[inlier]) * 1000),
            }
            # A point born on a newly visible surface joins the same rigid frame.
            newborn = current_support & (scene["query_points_2d"][:, 0] == index)
            canonical[newborn] = (measured[newborn] - shift) @ rotation
        except ValueError:
            # Do not create a plausible-looking landmark from a missing fit.
            record = {
                "rotation": None,
                "shift": None,
                "supported": int(current_support.sum()),
                "shared": len(ids),
                "inliers": 0,
                "median_inlier_mm": None,
            }
        output[frame] = record
    return output, supports


def basis_from_normal(normal: np.ndarray) -> np.ndarray:
    x = np.array([1.0, 0.0, 0.0])
    x -= np.dot(x, normal) * normal
    x /= np.linalg.norm(x)
    y = np.cross(normal, x)
    return np.column_stack((x, y, normal))


def calculate_records(scene: dict, k: np.ndarray) -> tuple[list[dict], dict]:
    dowel, dowel_supports = collect_group("dowel", scene, k)
    ring, ring_supports = collect_group("ring", scene, k)
    base_index = ANCHOR - int(scene["frames"][0])
    depth = cv2.imread(str(DEMO / "depth" / f"{ANCHOR:06d}.png"), cv2.IMREAD_UNCHANGED)
    # These are hand-picked annotations for this explanatory video, not learned
    # semantic keypoints or validated ground truth.
    lower_end = point_from_depth((220, 182), depth, k)
    upper_end = point_from_depth((226, 103), depth, k)
    ring_points = scene["query_coords_m"][base_index, ring_supports[ANCHOR]]
    center = np.median(ring_points, axis=0)
    _, _, vt = np.linalg.svd(ring_points - center)
    normal = vt[-1]
    if normal[2] > 0:
        normal = -normal
    # The hole has no depth reading of its own: intersect its image ray with
    # an approximate ring surface plane. This is a geometric estimate.
    hole_pixel = (255, 202)
    ray = np.array([(hole_pixel[0] - k[0, 2]) / k[0, 0], (hole_pixel[1] - k[1, 2]) / k[1, 1], 1.0])
    hole = ray * (np.dot(center, normal) / np.dot(ray, normal))
    ring_basis = basis_from_normal(normal)
    records = []
    for frame in range(ANCHOR, END + 1):
        d, r = dowel[frame], ring[frame]
        item = {
            "frame": frame,
            "dowel": {key: value for key, value in d.items() if key not in {"rotation", "shift"}},
            "ring": {key: value for key, value in r.items() if key not in {"rotation", "shift"}},
            "dowel_support": dowel_supports[frame],
            "ring_support": ring_supports[frame],
        }
        if d["rotation"] is not None and r["rotation"] is not None:
            p = lower_end @ d["rotation"].T + d["shift"]
            upper = upper_end @ d["rotation"].T + d["shift"]
            h = hole @ r["rotation"].T + r["shift"]
            basis = r["rotation"] @ ring_basis
            relative_mm = (p - h) @ basis * 1000
            dowel_axis = (upper - p) / np.linalg.norm(upper - p)
            axis_error = math.degrees(math.acos(np.clip(np.dot(dowel_axis, basis[:, 2]), -1, 1)))
            item.update(
                tip_pixel=_project(p[None, :], k)[0].tolist(),
                upper_pixel=_project(upper[None, :], k)[0].tolist(),
                hole_pixel=_project(h[None, :], k)[0].tolist(),
                vector_mm=relative_mm.tolist(),
                lateral_mm=float(np.linalg.norm(relative_mm[:2])),
                axial_mm=float(relative_mm[2]),
                axis_error_deg=float(axis_error),
            )
        records.append(item)
    metadata = {
        "source": str(GROWTH / "scene.npz"),
        "demo": str(DEMO),
        "frame_range": [ANCHOR, END],
        "fps": FPS,
        "landmark_annotation": {
            "frame": ANCHOR,
            "dowel_surface_proxy_near_inserted_end_px": [220, 182],
            "dowel_upper_end_px": [226, 103],
            "ring_hole_px": list(hole_pixel),
        },
        "ring_hole_method": "image ray intersected with PCA plane of visible ring points",
        "meaning": "v_R(t)=R_ring(t)^T (P_dowel(t)-H_ring(t)), mm; axial axis points out of hole",
        "caveat": "Landmarks and PCA plane are approximate; inferred positions are not physical ground truth.",
    }
    return records, metadata


def line_plot(
    draw: ImageDraw.ImageDraw,
    values: list[float | None],
    upto: int,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    unit: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.text((x0, y0), label, font=f(22, True), fill=WHITE)
    usable = [v for v in values if v is not None]
    if not usable:
        return
    lo, hi = min(0.0, min(usable)), max(usable)
    if hi - lo < 1:
        hi = lo + 1
    gy0, gy1 = y0 + 36, y1 - 12
    draw.line((x0, gy1, x1, gy1), fill=(65, 78, 92), width=2)
    segments = []
    for i, value in enumerate(values[: upto + 1]):
        if value is None:
            if len(segments) > 1:
                draw.line(segments, fill=color, width=3)
            segments = []
            continue
        x = x0 + (x1 - x0) * i / max(1, len(values) - 1)
        y = gy1 - (gy1 - gy0) * (value - lo) / (hi - lo)
        segments.append((x, y))
    if len(segments) > 1:
        draw.line(segments, fill=color, width=3)
    current = values[upto]
    if current is not None:
        draw.ellipse(
            (segments[-1][0] - 5, segments[-1][1] - 5, segments[-1][0] + 5, segments[-1][1] + 5), fill=color
        )
        draw.text((x1 - 116, y0 - 3), f"{current:+.1f} {unit}", font=f(19, True), fill=color)


def render(record: dict, records: list[dict], index: int, scene: dict, k: np.ndarray) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(canvas)
    frame = record["frame"]
    draw.text((45, 25), "Insertion demo: what a point-group action stores", font=f(36, True), fill=WHITE)
    draw.text(
        (46, 78),
        f"Visible TrackEverything points  •  frame {frame}  •  {(frame - ANCHOR) / FPS:.2f} s",
        font=f(22),
        fill=MUTED,
    )
    camera_box = (43, 126, 792, 859)
    draw.rounded_rectangle(camera_box, radius=18, fill=PANEL)
    rgb = cv2.cvtColor(cv2.imread(str(DEMO / "rgb" / f"{frame:06d}.jpg")), cv2.COLOR_BGR2RGB)
    crop_box = (100, 50, 375, 320)
    crop = Image.fromarray(rgb).crop(crop_box).resize((715, 702), Image.Resampling.BILINEAR)
    camera = ImageDraw.Draw(crop)

    def to_crop(p: tuple[float, float] | list[float]) -> tuple[float, float]:
        return (
            (p[0] - crop_box[0]) * 715 / (crop_box[2] - crop_box[0]),
            (p[1] - crop_box[1]) * 702 / (crop_box[3] - crop_box[1]),
        )

    scene_index = frame - int(scene["frames"][0])
    all_points = scene["query_coords_m"][scene_index]
    for name, color in (("ring", RING_COLOR), ("dowel", DOWEL_COLOR)):
        ids = np.flatnonzero(record[name + "_support"])
        xy = _project(all_points[ids], k)
        for point in xy:
            px, py = to_crop(point)
            if 0 <= px < 715 and 0 <= py < 702:
                camera.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color, outline=(5, 18, 23), width=1)
    if "vector_mm" in record:
        hx, hy = to_crop(record["hole_pixel"])
        px, py = to_crop(record["tip_pixel"])
        ux, uy = to_crop(record["upper_pixel"])
        camera.line((hx, hy, px, py), fill=WHITE, width=4)
        camera.line((px, py, ux, uy), fill=TIP_COLOR, width=3)
        camera.ellipse((px - 12, py - 12, px + 12, py + 12), outline=TIP_COLOR, width=4)
        camera.line((px - 18, py, px + 18, py), fill=TIP_COLOR, width=3)
        camera.line((px, py - 18, px, py + 18), fill=TIP_COLOR, width=3)
        camera.polygon(
            [(hx, hy - 13), (hx + 13, hy), (hx, hy + 13), (hx - 13, hy)], fill=HOLE_COLOR, outline=WHITE
        )
        camera.text(
            (min(px + 18, 575), max(8, py - 32)),
            "P: inferred proxy" if frame > ANCHOR else "P: marked proxy",
            font=f(17, True),
            fill=TIP_COLOR,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        camera.text(
            (max(4, hx - 145), min(677, hy + 16)),
            "H: ring hole",
            font=f(18, True),
            fill=HOLE_COLOR,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    camera.rounded_rectangle((12, 641, 385, 693), radius=9, fill=(17, 26, 34))
    camera.ellipse((25, 654, 36, 665), fill=DOWEL_COLOR)
    camera.text((46, 649), "dowel", font=f(19, True), fill=WHITE)
    camera.ellipse((160, 654, 171, 665), fill=RING_COLOR)
    camera.text((180, 649), "ring", font=f(19, True), fill=WHITE)
    canvas.paste(crop, (60, 142))
    draw.rounded_rectangle((817, 126, 1557, 858), radius=18, fill=PANEL)
    draw.text((845, 148), "Example value to store", font=f(29, True), fill=WHITE)
    draw.text((845, 192), "v_R = R_ring^T (P_dowel − H_ring)", font=f(23), fill=WHITE)
    draw.text((845, 226), "3D offset in a ring-attached frame; no object center", font=f(17), fill=MUTED)
    if "vector_mm" in record:
        vector = record["vector_mm"]
        draw.text(
            (845, 265),
            f"v_R = [{vector[0]:+.1f}, {vector[1]:+.1f}, {vector[2]:+.1f}] mm",
            font=f(27, True),
            fill=WHITE,
        )
        draw.text(
            (845, 309), f"Lateral offset   {record['lateral_mm']:.1f} mm", font=f(26, True), fill=TIP_COLOR
        )
        draw.text(
            (845, 347), f"Axial offset      {record['axial_mm']:+.1f} mm", font=f(26, True), fill=HOLE_COLOR
        )
        draw.text(
            (845, 385),
            f"Axis difference  {record['axis_error_deg']:.1f}°",
            font=f(26, True),
            fill=DOWEL_COLOR,
        )
        draw.text((845, 423), "Axial + = above ring plane; − = past plane", font=f(17), fill=MUTED)
    else:
        draw.text((845, 292), "No supported rigid estimate this frame", font=f(23, True), fill=TIP_COLOR)
    lateral = [r.get("lateral_mm") for r in records]
    axial = [r.get("axial_mm") for r in records]
    angle = [r.get("axis_error_deg") for r in records]
    line_plot(draw, lateral, index, (845, 468, 1518, 566), TIP_COLOR, "Lateral", "mm")
    line_plot(draw, axial, index, (845, 572, 1518, 670), HOLE_COLOR, "Axial", "mm")
    line_plot(draw, angle, index, (845, 676, 1518, 773), DOWEL_COLOR, "Axis", "deg")
    draw.text(
        (845, 789),
        f"Visible TE points: dowel {record['dowel']['supported']}  •  ring {record['ring']['supported']}",
        font=f(18),
        fill=WHITE,
    )
    draw.text(
        (845, 817),
        f"Rigid-fit inliers: dowel {record['dowel']['inliers']}/{record['dowel']['shared']}  •  ring {record['ring']['inliers']}/{record['ring']['shared']}",
        font=f(18),
        fill=MUTED,
    )
    if frame >= 500:
        draw.rounded_rectangle((1353, 151, 1528, 180), radius=6, fill=(84, 47, 34))
        draw.text((1362, 154), "FIT UNSTABLE", font=f(16, True), fill=TIP_COLOR)
    draw.text(
        (55, 868),
        "P is a marked surface proxy near the inserted end. Propagated landmarks and ring plane are estimates, not ground truth.",
        font=f(17),
        fill=MUTED,
    )
    return canvas


def main() -> None:
    k = np.loadtxt(DEMO / "cam_K.txt").reshape(3, 3)
    with np.load(GROWTH / "scene.npz", allow_pickle=False) as saved:
        scene = {key: saved[key] for key in saved.files}
    records, metadata = calculate_records(scene, k)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT.with_suffix(".json")
    result_path.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "frames": [
                    {key: value for key, value in rec.items() if not key.endswith("_support")}
                    for rec in records
                ],
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
        for i, record in enumerate(records):
            frame = render(record, records, i, scene, k)
            ffmpeg.stdin.write(frame.tobytes())
    finally:
        ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(OUTPUT)
    print(result_path)


if __name__ == "__main__":
    main()
