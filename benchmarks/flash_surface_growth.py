"""Build labelled later-birth query cohorts on an existing YCBInEOAT clip.

New queries come from each birth frame's GT object mask and measured depth.
Previously seeded material points are projected into that frame and excluded
within a configurable pixel spacing. The pose annotation supplies evaluation
labels only; it is never passed to the point tracker.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks.flash_surface_benchmark import _digest, _load, select_queries


def _project(xyz: np.ndarray, k: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = xyz[..., :2] / xyz[..., 2:3]
    return xy * [k[0, 0], k[1, 1]] + [k[0, 2], k[1, 2]]


def _select_new(
    mask: np.ndarray,
    depth: np.ndarray,
    occupied: np.ndarray,
    count: int,
    min_spacing_px: float,
) -> np.ndarray:
    """Spread queries across currently observed surface, excluding old anchors."""
    if count < 1 or min_spacing_px < 0:
        raise ValueError("count must be positive and spacing nonnegative")
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    if len(occupied):
        finite = occupied[np.isfinite(occupied).all(axis=1)]
        if len(finite):
            # A distance transform makes the geometric exclusion independent of
            # the number of pixels and existing points.
            free = np.ones(mask.shape, np.uint8)
            xy = np.rint(finite).astype(int)
            inside = (
                (xy[:, 0] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < mask.shape[0])
            )
            free[xy[inside, 1], xy[inside, 0]] = 0
            separation = cv2.distanceTransform(free, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            valid &= separation >= min_spacing_px
    return select_queries(valid.astype(np.uint8), depth, count)


def build(
    base: Path,
    output: Path,
    births: list[int],
    points_per_birth: int,
    min_spacing_px: float,
) -> dict:
    manifest = json.loads((base / "manifest.json").read_text())
    if not str(manifest.get("ground_truth", "")).startswith("YCBInEOAT annotated object poses"):
        raise ValueError("later-birth reference currently requires YCBInEOAT object-pose labels")
    source = _load(base / "queries.npz")
    if not births or sorted(set(births)) != births or births[0] <= 0 or births[-1] >= len(source["frames"]):
        raise ValueError("births must be distinct, ordered, nonzero in-clip frame indices")
    if points_per_birth < 1 or min_spacing_px < 0:
        raise ValueError("points per birth must be positive and spacing nonnegative")
    recording, mask_dir = Path(manifest["recording"]), Path(manifest["mask_dir"])
    pose_paths = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    indices = [manifest["source_first_index"] + int(index) for index in source["frames"]]
    poses = np.stack([np.loadtxt(pose_paths[index]).reshape(4, 4) for index in indices])
    k = source["intrinsics"]
    queries = source["queries"].copy()
    seed_depth = source["initial_depth_m"].copy()
    existing_xyz = source["truth_xyz"].copy()
    additions = []
    for birth in births:
        frame = int(source["frames"][birth])
        mask = cv2.imread(str(mask_dir / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        if mask is None or raw is None or raw.dtype != np.uint16 or mask.shape != raw.shape:
            raise ValueError(f"missing aligned mask/depth on birth frame {frame}")
        xy = _select_new(mask, raw, _project(existing_xyz[birth], k), points_per_birth, min_spacing_px)
        z = raw[xy[:, 1].astype(int), xy[:, 0].astype(int)].astype(float) * 0.001
        xyz = np.column_stack(((xy[:, 0] - k[0, 2]) * z / k[0, 0], (xy[:, 1] - k[1, 2]) * z / k[1, 1], z))
        local = (np.linalg.inv(poses[birth]) @ np.column_stack((xyz, np.ones(len(xyz)))).T).T
        trajectory = np.array([(pose @ local.T).T[:, :3] for pose in poses])
        existing_xyz = np.concatenate((existing_xyz, trajectory), axis=1)
        additions.append((birth, xy, z, trajectory))
    queries = np.concatenate(
        (queries, *(np.column_stack((np.full(len(xy), birth), xy)) for birth, xy, _, _ in additions))
    ).astype(np.float32)
    seed_depth = np.concatenate((seed_depth, *(z for _, _, z, _ in additions))).astype(np.float32)
    truth_xyz = existing_xyz.astype(np.float32)
    truth_xy = _project(truth_xyz, k).astype(np.float32)
    truth_visible = np.zeros(truth_xy.shape[:2], bool)
    depth_gap = np.full(truth_xy.shape[:2], np.nan, np.float32)
    for index, frame in enumerate(source["frames"]):
        mask = cv2.imread(str(mask_dir / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        raw = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        xy = np.rint(truth_xy[index]).astype(int)
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < raw.shape[1]) & (xy[:, 1] >= 0) & (xy[:, 1] < raw.shape[0])
        ids = np.flatnonzero(inside & (queries[:, 0] <= index))
        measured = raw[xy[ids, 1], xy[ids, 0]].astype(float) * 0.001
        gap = np.abs(measured - truth_xyz[index, ids, 2])
        depth_gap[index, ids] = gap
        truth_visible[index, ids] = (
            (mask[xy[ids, 1], xy[ids, 0]] > 0) & (measured > 0) & (gap <= manifest["visibility_tolerance_m"])
        )
    if not np.all(truth_visible[queries[:, 0].astype(int), np.arange(len(queries))]):
        raise ValueError("a seeded point has no surface support on its birth frame")
    truth_xyz[np.arange(len(source["frames"]))[:, None] < queries[:, 0]] = np.nan
    truth_xy[np.arange(len(source["frames"]))[:, None] < queries[:, 0]] = np.nan
    output.mkdir(parents=True, exist_ok=True)
    query_path = output / "queries.npz"
    np.savez_compressed(
        query_path,
        frames=source["frames"],
        capture_s=source["capture_s"],
        intrinsics=k,
        queries=queries,
        initial_depth_m=seed_depth,
        truth_xy=truth_xy,
        truth_xyz=truth_xyz,
        truth_visible=truth_visible,
        reference_depth_gap_m=depth_gap,
    )
    output_manifest = {
        **manifest,
        "query_count": len(queries),
        "queries_sha256": _digest(query_path),
        "base_query_sha256": manifest["queries_sha256"],
        "query_birth_schedule": {str(birth): points_per_birth for birth in births},
        "minimum_old_query_spacing_px": min_spacing_px,
        "labelled_visible_fraction": float(
            truth_visible.sum() / np.count_nonzero(np.arange(len(source["frames"]))[:, None] >= queries[:, 0])
        ),
        "reference_depth_gap_m_median_on_visible": float(np.median(depth_gap[truth_visible])),
    }
    (output / "manifest.json").write_text(json.dumps(output_manifest, indent=2) + "\n")
    return output_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--births", nargs="+", type=int, required=True)
    parser.add_argument("--points-per-birth", type=int, required=True)
    parser.add_argument("--min-spacing-px", type=float, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            build(args.base, args.output, args.births, args.points_per_birth, args.min_spacing_px), indent=2
        )
    )


if __name__ == "__main__":
    main()
