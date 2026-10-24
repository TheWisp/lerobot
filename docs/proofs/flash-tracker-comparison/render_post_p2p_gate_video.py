"""Show what happens when the unchanged P2P output fails a final pose check.

Both panels use the same recorded RGB-D frame and the exact same P2P pose.
The lower panel withholds the pose only when the post-output gate rejects it.
Cyan is the visible shaft axis fitted from current masked depth; orange is the
P2P proposal. Cyan does not measure the hidden insertion end.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from analyze_dowel_release_geometry import cloud_and_pixels, shaft_middle

HERE = Path(__file__).parent
SIZE = (720, 1480)
CROP = (120, 55, 360, 265)
PANEL_SIZE = (640, 560)
PANEL_X = 40
PANEL_YS = (180, 805)
CYAN = (245, 220, 35)
ORANGE = (45, 160, 255)
GREEN = (100, 240, 145)
RED = (75, 75, 250)
WHITE = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def write_text(image: np.ndarray, label: str, x: int, y: int, scale: float, color=WHITE) -> None:
    cv2.putText(image, label, (x, y), FONT, scale, (5, 8, 14), 6, cv2.LINE_AA)
    cv2.putText(image, label, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    xy = points[:, :2] / points[:, 2, None]
    return xy @ intrinsics[:2, :2].T + intrinsics[:2, 2]


def crop_projected_line(image: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    left, top, right, bottom = CROP
    scale_x = PANEL_SIZE[0] / (right - left)
    scale_y = PANEL_SIZE[1] / (bottom - top)
    xy = np.rint((points - (left, top)) * (scale_x, scale_y)).astype(int)
    valid, first, last = cv2.clipLine((0, 0, *PANEL_SIZE), tuple(xy[0]), tuple(xy[1]))
    if valid:
        cv2.line(image, first, last, color, 6, cv2.LINE_AA)
    for point in xy:
        if 0 <= point[0] < PANEL_SIZE[0] and 0 <= point[1] < PANEL_SIZE[1]:
            cv2.circle(image, tuple(point), 6, color, -1, cv2.LINE_AA)


def frame_panel(
    frame: int,
    dataset: Path,
    masks: Path,
    intrinsics: np.ndarray,
    shaft_points: np.ndarray,
    pose: np.ndarray | None,
    visible_line: np.ndarray,
) -> np.ndarray:
    rgb = cv2.imread(str(dataset / "rgb" / f"{frame:06d}.jpg"))
    mask = cv2.imread(str(masks / "dowel" / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
    if rgb is None or mask is None or mask.shape != rgb.shape[:2]:
        raise ValueError(f"missing aligned RGB/mask for frame {frame}")
    left, top, right, bottom = CROP
    panel = cv2.resize(rgb[top:bottom, left:right], PANEL_SIZE, interpolation=cv2.INTER_CUBIC)
    cropped_mask = cv2.resize(mask[top:bottom, left:right], PANEL_SIZE, interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(cropped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(panel, contours, -1, GREEN, 2, cv2.LINE_AA)
    crop_projected_line(panel, visible_line, CYAN)
    if pose is not None:
        camera_shaft = shaft_points @ pose[:3, :3].T + pose[:3, 3]
        crop_projected_line(panel, project(camera_shaft, intrinsics), ORANGE)
    return panel


def load_decisions(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="") as stream:
        return {int(row["frame"]): row for row in csv.DictReader(stream)}


def render(
    frame: int,
    dataset: Path,
    masks: Path,
    intrinsics: np.ndarray,
    shaft_points: np.ndarray,
    poses: np.ndarray,
    decisions: dict[int, dict[str, str]],
    *,
    freeze: bool,
) -> np.ndarray:
    canvas = np.full((SIZE[1], SIZE[0], 3), (18, 23, 29), dtype=np.uint8)
    write_text(canvas, "CHECK AFTER POINT2POSE", 40, 53, 0.92)
    write_text(canvas, "Same P2P pose; SDF unchanged", 40, 86, 0.64)
    write_text(canvas, "CYAN: visible shaft  ORANGE: P2P pose", 40, 115, 0.55, CYAN)
    points, pixels, _ = cloud_and_pixels(dataset, masks, frame, intrinsics)
    middle, _, visible_axis, _ = shaft_middle(points, pixels)
    visible_center = middle.mean(axis=0)
    visible_line = project(visible_center + np.array((-0.05, 0.05))[:, None] * visible_axis, intrinsics)
    decision = decisions[frame]
    rejected = decision["verdict"] == "rejected"
    for name, y, label in (
        ("proposal", PANEL_YS[0], "Point2Pose emitted pose"),
        ("checked", PANEL_YS[1], "After the final pose check"),
    ):
        write_text(canvas, label, 40, y - 16, 0.88)
        pose = poses[frame - 100] if name == "proposal" or not rejected else None
        panel = frame_panel(frame, dataset, masks, intrinsics, shaft_points, pose, visible_line)
        canvas[y : y + PANEL_SIZE[1], PANEL_X : PANEL_X + PANEL_SIZE[0]] = panel
        if name == "checked":
            cv2.rectangle(
                canvas,
                (PANEL_X + 10, y + PANEL_SIZE[1] - 53),
                (PANEL_X + 580, y + PANEL_SIZE[1] - 8),
                (8, 12, 17),
                -1,
            )
            verdict_label = (
                "REJECTED: pose withheld"
                if rejected
                else "Unresolved: pose not rejected"
                if decision["verdict"] == "unresolved"
                else "Supported: pose retained"
            )
            write_text(
                canvas,
                verdict_label,
                PANEL_X + 22,
                y + PANEL_SIZE[1] - 19,
                0.72,
                RED if rejected else WHITE,
            )
    if freeze:
        write_text(canvas, f"Frame 540: {decision['contradicted_rays']} rays contradict", 40, 1404, 0.76)
        write_text(canvas, f"the pose; {decision['supported_rays']} support it.", 40, 1445, 0.76)
    else:
        write_text(canvas, "No new pose is invented here.", 40, 1404, 0.72)
        write_text(canvas, f"Recorded frame {frame}", 40, 1445, 0.72, (185, 195, 205))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--masks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--poster", required=True, type=Path)
    args = parser.parse_args()
    with np.load(HERE / "pose_traces.npz") as features:
        intrinsics = features["intrinsics"].copy()
        center = features["dowel_reference_center_camera_m"].copy()
        axis = features["dowel_reference_shaft_camera"].copy()
        offsets = features["dowel_reference_end_offsets_m"].copy()
    shaft_points = center + offsets[:, None] * axis
    with np.load(HERE / "p2p_dowel_internal_control_100_550.npz") as original:
        poses = original["poses"].copy()
        source_lost = original["lost"].copy()
    with np.load(HERE / "post_p2p_dowel_gated_100_550.npz") as checked:
        if not np.array_equal(poses, checked["poses"]):
            raise ValueError("the post-output gate changed Point2Pose poses")
        if np.any(source_lost[400:441]):
            raise ValueError("source tracker lost one of the rendered frames")
    decisions = load_decisions(HERE / "post_p2p_dowel_gate.csv")
    if decisions[540]["verdict"] != "rejected":
        raise ValueError("frame 540 was not rejected by the post-output gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.poster.parent.mkdir(parents=True, exist_ok=True)
    poster = render(540, args.dataset, args.masks, intrinsics, shaft_points, poses, decisions, freeze=True)
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
        for frame in (500, *range(500, 541), *([540] * 42)):
            image = render(
                frame,
                args.dataset,
                args.masks,
                intrinsics,
                shaft_points,
                poses,
                decisions,
                freeze=frame == 540,
            )
            encoder.stdin.write(image.tobytes())
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise RuntimeError("video encoder failed")


if __name__ == "__main__":
    main()
