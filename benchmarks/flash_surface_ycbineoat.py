"""Build labelled RGB-D point clips from the public YCBInEOAT recordings.

Reference material points use the first measured depth and the dataset's
annotated object poses. The dataset mask and raw depth determine visibility.
The pose, mask and reference tracks are never inputs to point models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _digest, prepare_local


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = points[..., :2] / points[..., 2:3]
    return xy * [intrinsics[0, 0], intrinsics[1, 1]] + [intrinsics[0, 2], intrinsics[1, 2]]


def build(
    source: Path, first: int, last: int, count: int, visibility_tolerance_m: float, output: Path
) -> dict:
    if first < 0 or last < first or visibility_tolerance_m <= 0:
        raise ValueError("invalid source interval or visibility tolerance")
    names = [p.name for p in sorted((source / "rgb").glob("*.png"))]
    if last >= len(names):
        raise ValueError("source interval exceeds available RGB-D frames")
    chosen = names[first : last + 1]
    source_poses = sorted((source / "annotated_poses").glob("*.txt"))
    if len(source_poses) != len(names):
        raise ValueError("pose count and image count differ; cannot pair by ordinal index")
    recording = output / "recording"
    masks = output / "masks"
    for directory in (recording / "rgb", recording / "depth", masks):
        directory.mkdir(parents=True, exist_ok=True)
    k = np.loadtxt(source / "cam_K.txt").reshape(3, 3)
    np.savetxt(recording / "cam_K.txt", k)
    source_times = np.array([int(name.removesuffix(".png")) for name in chosen], np.int64)
    # YCBInEOAT filenames are capture timestamps in nanoseconds.
    if np.any(np.diff(source_times) <= 0):
        raise ValueError("RGB filenames are not ordered nanosecond timestamps")
    np.savetxt(recording / "times.txt", (source_times - source_times[0]) * 1e-9, fmt="%.9f")
    poses = []
    for index, name in enumerate(chosen):
        image = cv2.imread(str(source / "rgb" / name), cv2.IMREAD_UNCHANGED)
        depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(source / "gt_mask" / name), cv2.IMREAD_GRAYSCALE)
        pose = np.loadtxt(source_poses[first + index]).reshape(4, 4)
        if image is None or depth is None or mask is None or depth.dtype != np.uint16:
            raise ValueError(f"missing or invalid source RGB-D frame {name}")
        if image.shape[:2] != depth.shape or mask.shape != depth.shape:
            raise ValueError(f"unaligned source RGB-D frame {name}")
        if not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"bad annotated pose at {name}")
        poses.append(pose)
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR) if image.shape[-1] == 4 else image
        if not cv2.imwrite(str(recording / "rgb" / f"{index:06d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError("cannot write normalized RGB")
        (recording / "depth" / f"{index:06d}.png").symlink_to((source / "depth" / name).resolve())
        (masks / f"{index:06d}.png").symlink_to((source / "gt_mask" / name).resolve())
    manifest = prepare_local(recording, masks, 0, len(chosen) - 1, count, output)
    reference_path = output / "queries.npz"
    with np.load(reference_path, allow_pickle=False) as data:
        reference = {key: data[key] for key in data.files}
    query = reference["queries"]
    z = reference["initial_depth_m"]
    source_xyz = np.column_stack(
        ((query[:, 1] - k[0, 2]) * z / k[0, 0], (query[:, 2] - k[1, 2]) * z / k[1, 1], z)
    )
    object_points = (np.linalg.inv(poses[0]) @ np.column_stack((source_xyz, np.ones(count))).T).T
    truth_xyz = np.array([(pose @ object_points.T).T[:, :3] for pose in poses])
    truth_xy = _project(truth_xyz, k)
    truth_visible = np.zeros((len(chosen), count), bool)
    depth_gap = np.full((len(chosen), count), np.nan)
    for index, name in enumerate(chosen):
        mask = cv2.imread(str(source / "gt_mask" / name), cv2.IMREAD_GRAYSCALE)
        depth = cv2.imread(str(source / "depth" / name), cv2.IMREAD_UNCHANGED).astype(float) * 0.001
        x = np.rint(truth_xy[index, :, 0]).astype(int)
        y = np.rint(truth_xy[index, :, 1]).astype(int)
        inside = (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        ids = np.flatnonzero(inside)
        gaps = np.abs(depth[y[ids], x[ids]] - truth_xyz[index, ids, 2])
        depth_gap[index, ids] = gaps
        truth_visible[index, ids] = (
            (mask[y[ids], x[ids]] > 0) & (depth[y[ids], x[ids]] > 0) & (gaps <= visibility_tolerance_m)
        )
    if not np.all(truth_visible[0]):
        raise ValueError("some initial query points lack annotated surface support")
    np.savez_compressed(
        reference_path,
        **reference,
        truth_xy=truth_xy.astype(np.float32),
        truth_xyz=truth_xyz.astype(np.float32),
        truth_visible=truth_visible,
        reference_depth_gap_m=depth_gap.astype(np.float32),
    )
    manifest.update(
        {
            "queries_sha256": _digest(reference_path),
            "ground_truth": "YCBInEOAT annotated object poses; source-depth seed; GT mask/raw-depth visibility",
            "visibility_tolerance_m": visibility_tolerance_m,
            "source": str(source.resolve()),
            "source_first_index": first,
            "source_last_index": last,
            "source_first_rgb_sha256": _digest(source / "rgb" / chosen[0]),
            "labelled_visible_fraction": float(truth_visible.mean()),
            "reference_depth_gap_m_median_on_visible": float(np.median(depth_gap[truth_visible])),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--first", type=int, required=True)
    parser.add_argument("--last", type=int, required=True)
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--visibility-tolerance-m", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.source, args.first, args.last, args.points, args.visibility_tolerance_m, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
