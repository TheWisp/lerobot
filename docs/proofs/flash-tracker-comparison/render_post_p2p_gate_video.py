"""Show the unchanged P2P SDF and the separate post-output pose check.

The captured SDF is only available at selected frames, so this is an honest
frame-540 hold rather than an interpolated SDF animation. Magenta dots are
observed near-zero TSDF cells projected through the emitted P2P pose. The
bottom panel uses an independent first-view RGB-D surface to test that same
pose against the current aligned mask/depth; it does not fit a new pose.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from analyze_ray_consistency import reference_points
from replay_post_p2p_pose_rejection import ray_labels

HERE = Path(__file__).parent
FRAME = 540
SIZE = (720, 1480)
CROP = (120, 55, 460, 265)
PANEL_SIZE = (640, 395)
PANEL_X = 40
TOP_Y = 180
BOTTOM_Y = 815
SDF_BAND = 0.15  # Previously documented near-zero TSDF visualization band.
WHITE = (245, 245, 245)
GREEN = (100, 240, 145)
MAGENTA = (225, 70, 240)
RED = (75, 75, 250)
CYAN = (245, 220, 35)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def write_text(canvas: np.ndarray, value: str, x: int, y: int, scale: float, color=WHITE) -> None:
    cv2.putText(canvas, value, (x, y), FONT, scale, (5, 8, 14), 6, cv2.LINE_AA)
    cv2.putText(canvas, value, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    positive = points[:, 2] > 0
    result = np.full((len(points), 2), np.nan)
    result[positive] = points[positive, :2] / points[positive, 2, None] @ intrinsics[:2, :2].T
    result[positive] += intrinsics[:2, 2]
    return result


def panel_pixels(pixels: np.ndarray) -> np.ndarray:
    left, top, right, bottom = CROP
    scale = (PANEL_SIZE[0] / (right - left), PANEL_SIZE[1] / (bottom - top))
    return np.rint((pixels - (left, top)) * scale).astype(int)


def draw_points(
    panel: np.ndarray, pixels: np.ndarray, color: tuple[int, int, int], radius: int, alpha: float
) -> None:
    if alpha <= 0:
        return
    overlay = panel.copy()
    for x, y in panel_pixels(pixels):
        if 0 <= x < PANEL_SIZE[0] and 0 <= y < PANEL_SIZE[1]:
            cv2.circle(overlay, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, panel, 1 - alpha, 0, dst=panel)


def base_panel(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    left, top, right, bottom = CROP
    panel = cv2.resize(rgb[top:bottom, left:right], PANEL_SIZE, interpolation=cv2.INTER_CUBIC)
    cropped_mask = cv2.resize(mask[top:bottom, left:right], PANEL_SIZE, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, GREEN, 2, cv2.LINE_AA)
    return panel


def render(
    rgb: np.ndarray,
    mask: np.ndarray,
    sdf_pixels: np.ndarray,
    supported_pixels: np.ndarray,
    contradicted_pixels: np.ndarray,
    *,
    sdf_alpha: float,
    check_alpha: float,
    support_count: int,
    contradiction_count: int,
) -> np.ndarray:
    canvas = np.full((SIZE[1], SIZE[0], 3), (18, 23, 29), dtype=np.uint8)
    write_text(canvas, "FRAME 540: SAME CAMERA IMAGE", 40, 52, 0.83)
    write_text(canvas, "GREEN LINE: current object outline", 40, 91, 0.58, GREEN)

    write_text(canvas, "1  POINT2POSE'S STORED SDF", 40, 153, 0.77)
    top = base_panel(rgb, mask)
    draw_points(top, sdf_pixels, MAGENTA, 5, sdf_alpha)
    canvas[TOP_Y : TOP_Y + PANEL_SIZE[1], PANEL_X : PANEL_X + PANEL_SIZE[0]] = top
    write_text(canvas, "MAGENTA: SDF surface cells at P2P's pose", 40, 618, 0.54, MAGENTA)
    write_text(canvas, "Shown through the scene, including hidden cells", 40, 654, 0.53)

    write_text(canvas, "2  CHECK THAT POSE AGAINST DEPTH", 40, 787, 0.71)
    bottom = base_panel(rgb, mask)
    draw_points(bottom, supported_pixels, CYAN, 3, check_alpha)
    draw_points(bottom, contradicted_pixels, RED, 3, check_alpha)
    canvas[BOTTOM_Y : BOTTOM_Y + PANEL_SIZE[1], PANEL_X : PANEL_X + PANEL_SIZE[0]] = bottom
    write_text(canvas, "RED: empty space   CYAN: depth agrees", 40, 1252, 0.59)
    if check_alpha >= 0.77:
        write_text(canvas, "POSE REJECTED", 40, 1345, 1.15, RED)
        write_text(canvas, f"{contradiction_count} contradict   {support_count} support", 40, 1395, 0.73)
        write_text(canvas, "No corrected pose was generated.", 40, 1445, 0.63)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poster", required=True, type=Path)
    args = parser.parse_args()

    rgb = cv2.imread(str(args.dataset / "rgb" / f"{FRAME:06d}.jpg"))
    raw_depth = cv2.imread(str(args.dataset / "depth" / f"{FRAME:06d}.png"), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(args.masks / "dowel" / f"{FRAME:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if (
        rgb is None
        or raw_depth is None
        or mask is None
        or rgb.shape[:2] != raw_depth.shape
        or mask.shape != raw_depth.shape
    ):
        raise ValueError("missing aligned frame-540 RGB-D or object mask")
    intrinsics = np.load(HERE / "pose_traces.npz")["intrinsics"]

    with np.load(HERE / "p2p_dowel_internal_control_100_550.npz") as trace:
        index = int(np.flatnonzero(trace["frames"] == FRAME)[0])
        if trace["lost"][index]:
            raise ValueError("P2P did not emit a frame-540 pose")
        pose = trace["poses"][index].copy()
    with np.load(HERE / "p2p_dowel_sdf_release_snapshots.npz") as snapshot:
        if not np.array_equal(pose, snapshot["pose_540"]):
            raise ValueError("saved SDF and emitted pose are from different P2P runs")
        near_zero = (snapshot["weight_540"] > 0) & (np.abs(snapshot["tsdf_540"]) < SDF_BAND)
        object_points = snapshot["origin"] + np.argwhere(near_zero) * float(snapshot["voxel_size"])
    sdf_camera = object_points @ pose[:3, :3].T + pose[:3, 3]
    sdf_pixels = project(sdf_camera, intrinsics)
    sdf_pixels = sdf_pixels[np.isfinite(sdf_pixels).all(axis=1)]

    config = json.loads(
        (HERE.parents[2] / "src/lerobot/policies/flash/docs/trial_config.example.json").read_text()
    )
    tracking_config = config["causal_tracking"]
    first_view = reference_points(args.dataset, args.masks, "dowel", intrinsics)
    x, y, supported, contradicted = ray_labels(
        first_view,
        pose,
        raw_depth.astype(np.float32) * 0.001,
        mask > 0,
        intrinsics,
        depth_tolerance_m=float(tracking_config["fit_threshold_m"]),
        local_depth_span_m=float(tracking_config["max_pose_disagreement_translation_m"]),
    )
    support_count, contradiction_count = int(supported.sum()), int(contradicted.sum())
    with (HERE / "post_p2p_dowel_gate.csv").open(newline="") as stream:
        row = next(row for row in csv.DictReader(stream) if int(row["frame"]) == FRAME)
    if (row["verdict"], int(row["supported_rays"]), int(row["contradicted_rays"])) != (
        "rejected",
        support_count,
        contradiction_count,
    ):
        raise ValueError("visualized depth evidence differs from the saved post-output verdict")
    supported_pixels = np.column_stack((x[supported], y[supported]))
    contradicted_pixels = np.column_stack((x[contradicted], y[contradicted]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    poster = render(
        rgb,
        mask,
        sdf_pixels,
        supported_pixels,
        contradicted_pixels,
        sdf_alpha=0.87,
        check_alpha=0.77,
        support_count=support_count,
        contradiction_count=contradiction_count,
    )
    if not cv2.imwrite(str(args.poster), poster):
        raise OSError(f"could not write poster {args.poster}")
    command = [
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
        f"{SIZE[0]}x{SIZE[1]}",
        "-r",
        "15",
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for index in range(90):
            canvas = render(
                rgb,
                mask,
                sdf_pixels,
                supported_pixels,
                contradicted_pixels,
                sdf_alpha=min(0.87, 0.87 * index / 14),
                check_alpha=min(0.77, 0.77 * max(0, index - 15) / 14),
                support_count=support_count,
                contradiction_count=contradiction_count,
            )
            encoder.stdin.write(canvas.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoder failed")


if __name__ == "__main__":
    main()
