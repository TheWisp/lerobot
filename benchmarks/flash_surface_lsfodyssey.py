"""Import a labelled LSFOdyssey RGB-D clip for visible-point evaluation.

The source supplies pixel/depth trajectories, visibility, depth images and
camera calibration. The point labels are never passed to model adapters except
for the common first-frame query pixels. Later object masks are unavailable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _digest
from benchmarks.flash_surface_nested_budget import farthest_order


def build(source: Path, output: Path, count: int) -> dict:
    with np.load(source / "deps.npz", allow_pickle=False) as saved:
        depth_m = np.asarray(saved["deps"][:, 0], dtype=np.float32)
    with np.load(source / "intris.npz", allow_pickle=False) as saved:
        intrinsics = np.asarray(saved["intris"][0], dtype=np.float32)
    with np.load(source / "track.npz", allow_pickle=False) as saved:
        tracks = np.asarray(saved["track_g"], dtype=np.float32)
    if depth_m.ndim != 3 or tracks.shape[0] != len(depth_m) or intrinsics.shape != (len(depth_m), 3, 3):
        raise ValueError("LSFOdyssey frames, tracks and camera matrices disagree")
    if not np.allclose(intrinsics, intrinsics[0], atol=1e-5):
        raise ValueError("this adapter requires a fixed intrinsic matrix for the interval")
    t, h, w = depth_m.shape
    if tracks.ndim != 3 or tracks.shape[2] != 5 or count < 3:
        raise ValueError("expected T x N x 5 labelled point tracks")
    initial = tracks[0]
    initial_x = np.clip(np.rint(initial[:, 0]).astype(int), 0, w - 1)
    initial_y = np.clip(np.rint(initial[:, 1]).astype(int), 0, h - 1)
    source_depth = depth_m[0, initial_y, initial_x]
    eligible_initial = (
        (initial[:, 3] > 0)
        & (initial[:, 4] > 0)
        & (initial[:, 0] >= 0)
        & (initial[:, 0] < w)
        & (initial[:, 1] >= 0)
        & (initial[:, 1] < h)
        & (initial[:, 2] > 0)
        & np.isfinite(source_depth)
        & (source_depth > 0)
        & (source_depth < 65.535)
    )
    candidate_ids = np.flatnonzero(eligible_initial)
    if len(candidate_ids) < count:
        raise ValueError("fewer initially visible labelled points than requested")
    chosen = candidate_ids[farthest_order(initial[candidate_ids, :2])[:count]]
    selected = tracks[:, chosen]
    video = cv2.VideoCapture(str(source / "rgb.mp4"))
    if not video.isOpened():
        raise ValueError("cannot open source RGB video")
    fps = video.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("RGB video has no declared frame rate")
    recording = output / "recording"
    for folder in (recording / "rgb", recording / "depth"):
        folder.mkdir(parents=True, exist_ok=True)
    for index in range(t):
        ok, bgr = video.read()
        if not ok or bgr.shape[:2] != (h, w):
            raise ValueError(f"missing or unaligned video frame {index}")
        millimetres = np.where(
            np.isfinite(depth_m[index]) & (depth_m[index] > 0) & (depth_m[index] < 65.535),
            np.rint(depth_m[index] * 1000),
            0,
        ).astype(np.uint16)
        if not cv2.imwrite(str(recording / "rgb" / f"{index:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("cannot save normalized source RGB")
        if not cv2.imwrite(str(recording / "depth" / f"{index:06d}.png"), millimetres):
            raise OSError("cannot save aligned metric depth")
    if video.read()[0]:
        raise ValueError("video has more frames than depth and point labels")
    video.release()
    np.savetxt(recording / "cam_K.txt", intrinsics[0])
    np.savetxt(recording / "times.txt", np.arange(t) / fps, fmt="%.9f")
    xy = selected[..., :2]
    z = selected[..., 2]
    inside = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    truth_visible = (selected[..., 3] > 0) & (selected[..., 4] > 0) & inside & np.isfinite(z) & (z > 0)
    k = intrinsics[0]
    truth_xyz = np.stack(
        ((xy[..., 0] - k[0, 2]) * z / k[0, 0], (xy[..., 1] - k[1, 2]) * z / k[1, 1], z),
        axis=-1,
    )
    query_xy = xy[0]
    query_x = np.clip(np.rint(query_xy[:, 0]).astype(int), 0, w - 1)
    query_y = np.clip(np.rint(query_xy[:, 1]).astype(int), 0, h - 1)
    measured_start = depth_m[0, query_y, query_x]
    if np.any(~np.isfinite(measured_start)) or np.any(measured_start <= 0):
        raise ValueError("source query pixels lack measured depth")
    x = np.clip(np.rint(xy[..., 0]).astype(int), 0, w - 1)
    y = np.clip(np.rint(xy[..., 1]).astype(int), 0, h - 1)
    depth_gap = np.abs(depth_m[np.arange(t)[:, None], y, x] - z)
    np.savez_compressed(
        output / "queries.npz",
        frames=np.arange(t),
        capture_s=np.arange(t) / fps,
        queries=np.column_stack((np.zeros(count), query_xy)).astype(np.float32),
        intrinsics=k,
        initial_depth_m=measured_start,
        truth_xy=xy,
        truth_xyz=truth_xyz,
        truth_visible=truth_visible,
        reference_depth_gap_m=depth_gap,
    )
    manifest = {
        "schema": "flash-visible-points-v1",
        "recording": str(recording.resolve()),
        "source": str(source.resolve()),
        "source_hw": [h, w],
        "first_frame": 0,
        "last_frame": t - 1,
        "query_count": count,
        "source_track_ids": chosen.tolist(),
        "source_rgb_sha256": _digest(source / "rgb.mp4"),
        "source_depth_sha256": _digest(source / "deps.npz"),
        "source_tracks_sha256": _digest(source / "track.npz"),
        "source_intrinsics_sha256": _digest(source / "intris.npz"),
        "declared_video_fps": float(fps),
        "ground_truth": "LSFOdyssey supplied point trajectories and visibility, camera-frame metric xyz",
        "reference_depth_gap_m_median_visible": float(np.median(depth_gap[truth_visible])),
        "queries_sha256": _digest(output / "queries.npz"),
        "first_rgb_sha256": _digest(recording / "rgb" / "000000.jpg"),
        "first_depth_sha256": _digest(recording / "depth" / "000000.png"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--points", type=int, default=128)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.points), indent=2))


if __name__ == "__main__":
    main()
