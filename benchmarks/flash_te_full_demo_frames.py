"""Show adjacent-frame TrackEverything fits for every object in a saved demo.

This is an offline diagnostic. TrackEverything used future video context; the
fits and local frames are not live measurements or a FLASH skill artifact.
When an object's supported adjacent-point graph breaks, its old frame ends.
The next visible segment starts an explicitly unlinked local frame.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
from flash_cross_episode_trackeverything import _project
from flash_te_demo_map_consistency import supported_ids
from flash_te_insertion_relation_video import DEMO, FPS, GROWTH, INLIER_BAND_M, f, fit_group
from PIL import Image, ImageDraw

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "docs/proofs/flash-cross-episode/te-demo-map/full-demo-adjacent-frames.mp4"
)
ROLE_COLORS = ((255, 185, 81), (124, 239, 163), (179, 168, 255))
POINT_COLOR = (106, 222, 235)
FIRST_DEMO_FRAME, LAST_DEMO_FRAME = 0, 712
WIDTH, HEIGHT = 848, 620


def load_scene() -> dict:
    with np.load(GROWTH / "scene.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def supported_tracks(scene: dict, intrinsics: np.ndarray, role: str, role_index: int) -> np.ndarray:
    first = int(scene["frames"][0])
    last = int(scene["frames"][-1])
    support = np.zeros((last - first + 1, len(scene["query_roles"])), dtype=bool)
    mask_dir = GROWTH / "segmentation/masks" / str(role_index)
    for frame in range(first, last + 1):
        if (mask_dir / f"{frame:06d}.png").exists():
            support[frame - first] = supported_ids(frame, scene, DEMO, mask_dir, intrinsics, role)[0]
    return support


def adjacent_edges(scene: dict, support: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first = int(scene["frames"][0])
    edges = np.full((len(support), 4, 4), np.nan)
    inliers = np.zeros(len(support), dtype=int)
    shared = np.zeros(len(support), dtype=int)
    edges[0] = np.eye(4)
    for index in range(1, len(support)):
        ids = np.flatnonzero(support[index - 1] & support[index])
        shared[index] = len(ids)
        if len(ids) < 3:
            continue
        previous = scene["query_coords_m"][index - 1, ids]
        current = scene["query_coords_m"][index, ids]
        try:
            rotation, shift, accepted = fit_group(previous, current, first + index)
        except ValueError:
            continue
        edges[index] = np.eye(4)
        edges[index, :3, :3] = rotation
        edges[index, :3, 3] = shift
        inliers[index] = accepted.sum()
    return edges, inliers, shared


def linked_segments(support: np.ndarray, edges: np.ndarray, first: int) -> list[tuple[int, int]]:
    segments = []
    start = None
    for index in range(len(support)):
        measurable = support[index].sum() >= 3
        connected = index > 0 and np.isfinite(edges[index, 0, 0])
        if not measurable:
            if start is not None:
                segments.append((start, first + index - 1))
                start = None
        elif start is None:
            start = first + index
        elif not connected:
            segments.append((start, first + index - 1))
            start = first + index
    if start is not None:
        segments.append((start, first + len(support) - 1))
    # A lone supported image cannot demonstrate frame-to-frame object motion.
    return [(start, end) for start, end in segments if end > start]


def anchor_ids(
    scene: dict,
    support: np.ndarray,
    start: int,
    end: int,
    seed: int,
    transforms: dict[int, np.ndarray],
) -> tuple[int, int, int]:
    """Choose separated tracks from the more self-consistent half of one segment."""
    first = int(scene["frames"][0])
    ids = np.flatnonzero(support[seed - first])
    points = scene["query_coords_m"][seed - first]
    scores = []
    for track_id in ids:
        reference = points[track_id]
        residuals = []
        for frame in range(start, end + 1):
            if not support[frame - first, track_id]:
                continue
            transform = transforms[frame]
            predicted = transform[:3, :3] @ reference + transform[:3, 3]
            measured = scene["query_coords_m"][frame - first, track_id]
            residuals.append(np.linalg.norm(predicted - measured))
        scores.append(float(np.quantile(residuals, 0.9)))
    ids = ids[np.asarray(scores) <= np.median(scores)]
    cloud = points[ids]
    center = np.median(cloud, axis=0)
    a = int(np.argmin(np.linalg.norm(cloud - center, axis=1)))
    b = int(np.argmax(np.linalg.norm(cloud - cloud[a], axis=1)))
    axis = cloud[b] - cloud[a]
    line_distance = np.linalg.norm(np.cross(cloud - cloud[a], axis), axis=1) / np.linalg.norm(axis)
    c = int(np.argmax(line_distance))
    return int(ids[a]), int(ids[b]), int(ids[c])


def build_role(scene: dict, intrinsics: np.ndarray, role: str, role_index: int) -> dict:
    first = int(scene["frames"][0])
    support = supported_tracks(scene, intrinsics, role, role_index)
    edges, inliers, shared = adjacent_edges(scene, support)
    segments = linked_segments(support, edges, first)
    anchors_by_frame = np.full((len(support), 3, 3), np.nan)
    observed_by_frame = np.zeros((len(support), 3), dtype=bool)
    residual_by_frame = np.full((len(support), 3), np.nan)
    segment_by_frame = np.zeros(len(support), dtype=int)
    descriptions = []
    for segment_number, (start, end) in enumerate(segments, start=1):
        seed = (start + end) // 2  # Offline reference; keeps it away from a segment boundary.
        seed_index = seed - first
        transforms = {seed: np.eye(4)}
        for frame in range(seed + 1, end + 1):
            transforms[frame] = edges[frame - first] @ transforms[frame - 1]
        for frame in range(seed - 1, start - 1, -1):
            transforms[frame] = np.linalg.inv(edges[frame + 1 - first]) @ transforms[frame + 1]
        ids = anchor_ids(scene, support, start, end, seed, transforms)
        base = scene["query_coords_m"][seed_index, list(ids)]
        for frame in range(start, end + 1):
            index = frame - first
            transform = transforms[frame]
            anchors = base @ transform[:3, :3].T + transform[:3, 3]
            raw = scene["query_coords_m"][index, list(ids)]
            anchors_by_frame[index] = anchors
            observed_by_frame[index] = support[index, list(ids)]
            residual_by_frame[index] = np.linalg.norm(raw - anchors, axis=1) * 1000
            segment_by_frame[index] = segment_number
        a, b, c = base
        base_height_mm = np.linalg.norm(np.cross(b - a, c - a)) / np.linalg.norm(b - a) * 1000
        observed = observed_by_frame[start - first : end - first + 1]
        residual = residual_by_frame[start - first : end - first + 1]
        description = {
            "number": segment_number,
            "frames": [start, end],
            "seed_frame": seed,
            "anchor_track_ids": list(ids),
            "initial_ab_mm": float(np.linalg.norm(b - a) * 1000),
            "initial_c_distance_from_ab_mm": float(base_height_mm),
            "raw_anchor_supported_frames": observed.sum(axis=0).tolist(),
            "raw_anchor_residual_p90_mm": [
                float(np.quantile(residual[observed[:, n], n], 0.9)) if observed[:, n].any() else None
                for n in range(3)
            ],
            "adjacent_fit_inliers_min": int(inliers[start - first + 1 : end - first + 1].min())
            if end > start
            else None,
        }
        descriptions.append(description)
    return {
        "role": role,
        "support": support,
        "inliers": inliers,
        "shared": shared,
        "segments": descriptions,
        "segment_by_frame": segment_by_frame,
        "anchors_by_frame": anchors_by_frame,
        "observed_by_frame": observed_by_frame,
        "residual_by_frame": residual_by_frame,
    }


def draw_frame(frame: int, scene: dict, intrinsics: np.ndarray, roles: list[dict]) -> Image.Image:
    first = int(scene["frames"][0])
    image = Image.open(DEMO / "rgb" / f"{frame:06d}.jpg").convert("RGB")
    canvas = Image.new("RGB", (WIDTH, HEIGHT), (17, 25, 33))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDTH, 38), fill=(14, 21, 30))
    draw.text((14, 8), "FLASH demo — adjacent-frame object motion", font=f(20, True), fill="white")
    draw.text((WIDTH - 176, 11), f"{frame:03d}/712  {frame / FPS:05.2f}s", font=f(15), fill="white")
    for role_index, result in enumerate(roles):
        color = ROLE_COLORS[role_index % len(ROLE_COLORS)]
        y = 493 + role_index * 42
        if frame < first:
            status = f"{result['role']}: no saved point tracks yet"
        else:
            index = frame - first
            support = result["support"][index]
            points = scene["query_coords_m"][index, support]
            for px, py in _project(points, intrinsics):
                if 0 <= px < 848 and 38 <= py < 480:
                    draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=POINT_COLOR)
            segment = result["segment_by_frame"][index]
            if segment == 0:
                status = f"{result['role']}: no supported pose / track gap"
            else:
                anchors = result["anchors_by_frame"][index]
                pixels = _project(anchors, intrinsics)
                draw.line([tuple(pixels[n]) for n in (0, 1, 2, 0)], fill=color, width=3)
                for point_number, (px, py) in enumerate(pixels):
                    if 0 <= px < 848 and 38 <= py < 480:
                        agrees = (
                            result["observed_by_frame"][index, point_number]
                            and result["residual_by_frame"][index, point_number] <= INLIER_BAND_M * 1000
                        )
                        draw.ellipse(
                            (px - 7, py - 7, px + 7, py + 7),
                            fill=color if agrees else None,
                            outline=color,
                            width=3,
                        )
                        draw.text((px + 9, py - 12), "ABC"[point_number], font=f(16, True), fill=color)
                if frame == result["segments"][segment - 1]["frames"][0] and segment > 1:
                    draw.text(
                        (15, 448 - 26 * role_index),
                        f"{result['role']}: new segment; prior frame not linked",
                        font=f(18, True),
                        fill=color,
                        stroke_width=2,
                        stroke_fill=(0, 0, 0),
                    )
                segment_start = result["segments"][segment - 1]["frames"][0]
                evidence = (
                    "new local frame"
                    if frame == segment_start
                    else f"adjacent fit {result['inliers'][index]}/{result['shared'][index]}"
                )
                status = (
                    f"{result['role']}: segment {segment} · {int(support.sum())} surface points · {evidence}"
                )
        draw.text((15, y), status, font=f(18, True), fill=color)
    draw.text(
        (15, 581),
        "Offline, future-aware diagnostic. Hollow A/B/C = raw ID absent or disagrees; gaps are not bridged.",
        font=f(14),
        fill=(172, 186, 200),
    )
    draw.rectangle((0, 612, WIDTH * (frame + 1) / (LAST_DEMO_FRAME + 1), 619), fill=(235, 190, 96))
    return canvas


def main() -> None:
    scene = load_scene()
    intrinsics = np.loadtxt(DEMO / "cam_K.txt").reshape(3, 3)
    roles = [build_role(scene, intrinsics, str(role), index) for index, role in enumerate(scene["roles"])]
    summary = {
        "demo": str(DEMO),
        "tracker_run": str(GROWTH / "scene.npz"),
        "full_demo_frames": [FIRST_DEMO_FRAME, LAST_DEMO_FRAME],
        "saved_tracker_frames": [int(scene["frames"][0]), int(scene["frames"][-1])],
        "method": "Mask/depth-supported TrackEverything points; robust adjacent-frame rigid fits. Each connected segment seeds a local triplet at its midpoint from its more temporally self-consistent half of supported IDs. Broken segments are not registered to each other.",
        "limitations": "This is a future-aware offline diagnostic, not a causal live tracker, a verified 6D material frame, or a demo-to-live object registration.",
        "objects": {
            item["role"]: {
                "segments": item["segments"],
                "unlinked_frames": [
                    int(index + scene["frames"][0]) for index in np.flatnonzero(item["segment_by_frame"] == 0)
                ],
            }
            for item in roles
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    process = subprocess.Popen(
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
            "35",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUTPUT),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame in range(FIRST_DEMO_FRAME, LAST_DEMO_FRAME + 1):
            process.stdin.write(draw_frame(frame, scene, intrinsics, roles).tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    print(OUTPUT)
    print(OUTPUT.with_suffix(".json"))


if __name__ == "__main__":
    main()
