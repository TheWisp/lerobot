"""Render synchronized, *unfused* evidence for the full insertion demonstration.

The drawing compares candidate poses with the recorded SAM3 mask and RGB-D depth.
It deliberately does not turn agreement with a mask into a ground-truth 6D pose or
choose one candidate for robot control. Run with ``--output /path/video.mp4``.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path

import cv2
import numpy as np
from analyze_ray_consistency import score_rays

DATA = Path("/home/feit/.cache/huggingface/lerobot/demos/insertion")
PROOF = Path(__file__).parent
FLASH = DATA / "flash/v1"
MASKS = Path("/home/feit/.cache/flash-tracker-compare/sam3_masks")
CAUSAL = FLASH / "preparation-cache/insertion-causal-parity-20261023-6024d11a02a2.npz"
MODELS = FLASH / "prepared/insertion-causal-parity-20261023/models.npz"
P2P = {
    "dowel": FLASH / "tracker-candidates/point2pose-dowel-review-20261021/poses.npz",
    "ring": FLASH / "tracker-candidates/point2pose-ring-ui-e2e-20261021/poses.npz",
}
COLORS = {
    "sam_dowel": (66, 210, 255),
    "sam_ring": (240, 135, 220),
    "p2p": (255, 225, 45),
    "group": (90, 235, 90),
    "gripper": (30, 155, 255),
    "direct": (245, 245, 245),
}


def label(
    image: np.ndarray,
    value: str,
    at: tuple[int, int],
    size: float = 0.66,
    color: tuple[int, int, int] = (245, 245, 245),
    thickness: int = 2,
) -> None:
    cv2.putText(image, value, at, cv2.FONT_HERSHEY_SIMPLEX, size, color, thickness, cv2.LINE_AA)


def project(
    points: np.ndarray, base_from_object: np.ndarray, camera_from_base: np.ndarray, intrinsics: np.ndarray
) -> np.ndarray:
    xyz = points @ (camera_from_base @ base_from_object)[:3, :3].T
    xyz += (camera_from_base @ base_from_object)[:3, 3]
    uv = np.full((len(points), 2), -1000, dtype=np.int32)
    front = xyz[:, 2] > 0
    uv[front] = np.rint(
        xyz[front, :2] / xyz[front, 2, None] * np.diag(intrinsics)[:2] + intrinsics[:2, 2]
    ).astype(np.int32)
    return uv


def draw_dowel(
    image: np.ndarray,
    model: np.ndarray,
    pose: np.ndarray,
    camera_from_base: np.ndarray,
    intrinsics: np.ndarray,
    color: tuple[int, int, int],
    width: int,
) -> None:
    centered = model - np.median(model, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    axial = centered @ axis
    center = np.median(model, axis=0)
    ends = np.stack((center + np.quantile(axial, 0.03) * axis, center + np.quantile(axial, 0.97) * axis))
    pixels = project(ends, pose, camera_from_base, intrinsics)
    if np.all((pixels[:, 0] >= 0) & (pixels[:, 0] < 848) & (pixels[:, 1] >= 0) & (pixels[:, 1] < 480)):
        start, end = map(tuple, pixels)
        cv2.line(image, start, end, color, width, cv2.LINE_AA)
        cv2.circle(image, start, width + 2, color, -1, cv2.LINE_AA)
        cv2.circle(image, end, width + 2, color, -1, cv2.LINE_AA)


def draw_ring(
    image: np.ndarray,
    model: np.ndarray,
    pose: np.ndarray,
    camera_from_base: np.ndarray,
    intrinsics: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    pixel = project(np.median(model, axis=0)[None], pose, camera_from_base, intrinsics)[0]
    x, y = map(int, pixel)
    if 0 <= x < 848 and 0 <= y < 480:
        cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 18, 3, cv2.LINE_AA)


def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> None:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)


def bars(
    canvas: np.ndarray,
    start_y: int,
    rows: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    frame: int,
    end: int,
) -> None:
    left, right = 280, 1252
    cv2.line(canvas, (left, start_y - 14), (right, start_y - 14), (70, 70, 70), 1)
    for offset, (name, available, color) in enumerate(rows):
        y = start_y + offset * 28
        label(canvas, name, (24, y + 13), 0.55, color, 1)
        for first in range(0, end, 3):
            if available[first : first + 3].any():
                x1 = left + int((right - left) * first / end)
                x2 = left + int((right - left) * min(first + 3, end) / end)
                cv2.rectangle(canvas, (x1, y), (max(x1 + 1, x2), y + 13), color, -1)
        x = left + int((right - left) * frame / end)
        cv2.line(canvas, (x, start_y - 16), (x, start_y + 28 * len(rows) - 4), (250, 250, 250), 2)


def score(
    model: np.ndarray,
    pose: np.ndarray,
    depth: np.ndarray,
    valid: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    camera_from_base: np.ndarray,
    tolerance_m: float,
) -> tuple[int, int, int, int, int, int]:
    return score_rays(
        model, camera_from_base @ pose, depth, valid, mask, intrinsics, depth_tolerance_m=tolerance_m
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--depth-tolerance-mm", type=float, default=5.0)
    args = parser.parse_args()
    if args.depth_tolerance_mm <= 0:
        parser.error("depth tolerance must be positive")

    causal, models = np.load(CAUSAL), np.load(MODELS)
    traces = {role: np.load(path) for role, path in P2P.items()}
    recording = np.load(PROOF / "pose_traces.npz")
    intrinsics = recording["intrinsics"]
    camera_from_base = np.linalg.inv(recording["base_from_camera"])
    ee = recording["demo_ee_pose_base"]
    total = len(list((DATA / "recording/rgb").glob("*.jpg")))
    grasp, release, held_anchor = 229, 514, 340
    # This is an explicit diagnostic anchor. It is not an automatically validated grasp update.
    held_index = int(np.flatnonzero(traces["dowel"]["frames"] == held_anchor)[0])
    held_attachment = np.linalg.inv(ee[held_anchor]) @ traces["dowel"]["poses"][held_index]
    p2p_lookup = {
        role: {int(frame): i for i, frame in enumerate(trace["frames"])} for role, trace in traces.items()
    }

    exists = np.zeros(total, dtype=bool)
    exists[100:551] = True
    p2p_dowel = np.zeros(total, dtype=bool)
    p2p_ring = np.zeros(total, dtype=bool)
    for role, array in (("dowel", p2p_dowel), ("ring", p2p_ring)):
        trace = traces[role]
        in_demo = trace["frames"] < total
        array[trace["frames"][in_demo]] = ~trace["lost"][in_demo]
    group = np.zeros(total, dtype=bool)
    ring_group = np.zeros(total, dtype=bool)
    group[: len(causal["dowel_estimator"])] = causal["dowel_estimator"] == "environment_group_motion"
    ring_group[: len(causal["ring_estimator"])] = causal["ring_estimator"] == "environment_group_motion"
    rigid = np.zeros(total, dtype=bool)
    rigid[held_anchor:release] = True
    rows: list[tuple[object, ...]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "1280x720",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "26",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(args.output),
        ],
        stdin=subprocess.PIPE,
    )
    assert writer.stdin is not None
    try:
        for frame in range(total):
            rgb = cv2.imread(str(DATA / "recording/rgb" / f"{frame:06d}.jpg"))
            if rgb is None:
                raise FileNotFoundError(f"RGB frame {frame}")
            rgb = cv2.resize(rgb, (848, 480))
            screen = np.full((720, 1280, 3), (25, 27, 31), dtype=np.uint8)
            screen[:480, :848] = rgb
            screen[:55, :848] = (screen[:55, :848].astype(np.float32) * 0.28).astype(np.uint8)
            label(screen, "FULL DEMO  /  DOWEL INSERTION", (27, 36), 0.9)
            label(screen, f"{frame / 30:.1f}s", (647, 36), 0.75)
            depth = None
            frame_masks = {}
            valid = None
            if exists[frame]:
                depth = (
                    cv2.imread(
                        str(DATA / "recording/depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED
                    ).astype(np.float32)
                    * 0.001
                )
                lower = cv2.erode(depth, np.ones((3, 3), dtype=np.uint8))
                upper = cv2.dilate(depth, np.ones((3, 3), dtype=np.uint8))
                valid = (depth > 0) & (lower > 0) & (upper - lower <= 0.010)
                for role in ("dowel", "ring"):
                    mask = cv2.imread(str(MASKS / role / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
                    frame_masks[role] = mask > 0
                    draw_mask(screen[:480, :848], mask, COLORS[f"sam_{role}"])

            poses: dict[tuple[str, str], np.ndarray] = {}
            for role in ("dowel", "ring"):
                trace_index = p2p_lookup[role].get(frame)
                if trace_index is not None and not traces[role]["lost"][trace_index]:
                    poses[(role, "p2p")] = traces[role]["poses"][trace_index]
                if frame < len(causal[f"{role}_evidence"]) and causal[f"{role}_evidence"][frame] != "lost":
                    poses[(role, "causal")] = causal[f"{role}_poses"][frame]
            if rigid[frame]:
                poses[("dowel", "gripper")] = ee[frame] @ held_attachment

            image = screen[:480, :848]
            if ("dowel", "p2p") in poses:
                draw_dowel(
                    image,
                    models["points_0"],
                    poses[("dowel", "p2p")],
                    camera_from_base,
                    intrinsics,
                    COLORS["p2p"],
                    3,
                )
            if group[frame]:
                draw_dowel(
                    image,
                    models["points_0"],
                    poses[("dowel", "causal")],
                    camera_from_base,
                    intrinsics,
                    COLORS["group"],
                    3,
                )
            if rigid[frame]:
                draw_dowel(
                    image,
                    models["points_0"],
                    poses[("dowel", "gripper")],
                    camera_from_base,
                    intrinsics,
                    COLORS["gripper"],
                    3,
                )
            if ("ring", "p2p") in poses:
                draw_ring(
                    image,
                    models["points_1"],
                    poses[("ring", "p2p")],
                    camera_from_base,
                    intrinsics,
                    COLORS["p2p"],
                )
            if ("ring", "causal") in poses:
                color = COLORS["group"] if ring_group[frame] else COLORS["direct"]
                draw_ring(
                    image, models["points_1"], poses[("ring", "causal")], camera_from_base, intrinsics, color
                )

            panel = screen[:480, 848:]
            label(panel, "CAMERA EVIDENCE", (24, 43), 0.83)
            label(panel, "SAM3 mask + measured depth", (24, 78), 0.58, COLORS["sam_dowel"])
            label(panel, "P2P proposed pose", (24, 115), 0.68, COLORS["p2p"])
            label(panel, "scene-group fallback", (24, 150), 0.68, COLORS["group"])
            label(panel, "gripper motion prediction", (24, 185), 0.62, COLORS["gripper"])
            cv2.line(panel, (20, 205), (408, 205), (90, 90, 90), 1)
            label(panel, "Projected rays: fit / free space", (24, 228), 0.54, (170, 170, 170), 1)
            if frame < grasp:
                phase = "BEFORE GRASP"
            elif frame < 400:
                phase = "PICK UP + REORIENT"
            elif frame < release:
                phase = "INSERTION"
            else:
                phase = "RELEASE + RETREAT"
            label(panel, phase, (24, 260), 0.72)
            if depth is None:
                label(panel, "SAM/P2P trace unavailable", (24, 298), 0.58, (180, 180, 180))
            else:
                for role, model_index, base_y in (("dowel", 0, 282), ("ring", 1, 376)):
                    label(panel, role.upper(), (24, base_y + 8), 0.68)
                    source_names = ["p2p", "causal"]
                    if role == "dowel" and rigid[frame]:
                        source_names.append("gripper")
                    best = []
                    for source in source_names:
                        pose = poses.get((role, source))
                        if pose is None:
                            continue
                        counts = score(
                            models[f"points_{model_index}"],
                            pose,
                            depth,
                            valid,
                            frame_masks[role],
                            intrinsics,
                            camera_from_base,
                            args.depth_tolerance_mm / 1000,
                        )
                        rows.append((frame, role, source, *counts))
                        support = counts[2] / max(counts[1], 1)
                        free = counts[3] / max(counts[1], 1)
                        caption = source
                        if source == "causal":
                            estimator = causal[f"{role}_estimator"][frame]
                            caption = {
                                "environment_group_motion": "group",
                                "object_point_motion": "points",
                            }.get(estimator, "direct")
                        best.append((caption, support, free))
                    for i, (caption, support, free) in enumerate(best[:3]):
                        label(
                            panel,
                            f"{caption:7s} fit {support:3.0%}  free {free:3.0%}",
                            (24, base_y + 29 + 28 * i),
                            0.56,
                            (215, 215, 215),
                            1,
                        )
            cv2.line(screen, (0, 480), (1279, 480), (100, 100, 100), 2)
            label(screen, "SOURCE AVAILABILITY  /  CAMERA FRAMES", (24, 513), 0.72)
            bars(
                screen,
                538,
                [
                    ("SAM + DEPTH", exists, COLORS["sam_dowel"]),
                    ("P2P DOWEL", p2p_dowel, COLORS["p2p"]),
                    ("SCENE GROUP", group, COLORS["group"]),
                    ("GRIPPER PRED.", rigid, COLORS["gripper"]),
                    ("P2P RING", p2p_ring, (205, 190, 60)),
                    ("RING GROUP", ring_group, (90, 170, 90)),
                ],
                frame,
                total,
            )
            writer.stdin.write(screen.tobytes())
    finally:
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("ffmpeg video encoding failed")
    if args.metrics:
        args.metrics.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics.open("w", newline="") as stream:
            csv_writer = csv.writer(stream, lineterminator="\n")
            csv_writer.writerow(
                (
                    "frame",
                    "role",
                    "source",
                    "modeled_pixels",
                    "valid_depth_pixels",
                    "supported_pixels",
                    "free_space_pixels",
                    "occluded_pixels",
                    "mask_pixels",
                )
            )
            csv_writer.writerows(rows)
    print(f"{args.output} ({total} demo frames, {total / 30:.1f} seconds)")


if __name__ == "__main__":
    main()
