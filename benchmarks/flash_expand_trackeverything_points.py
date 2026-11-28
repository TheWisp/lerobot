"""Experiment: seed TrackEverything queries on newly exposed object surfaces.

This deliberately evaluates point identities, not a 6D pose. A baseline trace
supplies original queries; SAM masks and RGB-D select a later, uncovered view.
The model then tracks both query sets in one inference call. Only points that
the model marks visible and that agree with the mask and measured depth count
in the report; hidden predictions remain in the raw trace for audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flash_trackeverything_pose import _query_pixels


def _project(
    points: np.ndarray, intrinsics: np.ndarray, shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = np.isfinite(points).all(axis=1) & (points[:, 2] > 0)
    z = np.where(finite, points[:, 2], 1.0)
    x = np.rint(intrinsics[0, 0] * points[:, 0] / z + intrinsics[0, 2]).astype(int)
    y = np.rint(intrinsics[1, 1] * points[:, 1] / z + intrinsics[1, 2]).astype(int)
    inside = finite & (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
    return x, y, inside


def _supported(
    points: np.ndarray,
    visibility: np.ndarray,
    mask: np.ndarray,
    depth: np.ndarray,
    intrinsics: np.ndarray,
    depth_tolerance_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, y, inside = _project(points, intrinsics, mask.shape)
    supported = inside & (visibility > 0)
    indices = np.flatnonzero(supported)
    supported[indices] = mask[y[indices], x[indices]] & (depth[y[indices], x[indices]] > 0)
    indices = np.flatnonzero(supported)
    supported[indices] &= np.abs(depth[y[indices], x[indices]] - points[indices, 2]) <= depth_tolerance_m
    return supported, x, y


def _novel_pixels(
    mask: np.ndarray,
    depth: np.ndarray,
    points: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
    depth_tolerance_m: float,
    query_count: int,
) -> tuple[np.ndarray, int]:
    valid = mask & np.isfinite(depth) & (depth > 0)
    supported, x, y = _supported(points, visibility, mask, depth, intrinsics, depth_tolerance_m)
    occupied = np.zeros(mask.shape, dtype=np.uint8)
    occupied[y[supported], x[supported]] = 1
    # One sample's nominal mask coverage derives from the requested query density.
    spacing_px = np.sqrt(valid.sum() / query_count)
    distance = cv2.distanceTransform(1 - occupied, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return valid & (distance > spacing_px), int(supported.sum())


def _read_mask(mask_dir: Path, frame: int, shape: tuple[int, int]) -> np.ndarray | None:
    path = mask_dir / f"{frame:06d}.png"
    if not path.is_file():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        raise ValueError(f"mask does not align with RGB-D frame {frame}: {path}")
    return mask > 0


def select_birth_frame(
    frames: np.ndarray,
    old_points: np.ndarray,
    old_visibility: np.ndarray,
    old_birth_indices: np.ndarray,
    depths: np.ndarray,
    masks: Path,
    intrinsics: np.ndarray,
    depth_tolerance_m: float,
    query_count: int,
) -> tuple[int, np.ndarray, dict]:
    """Choose the view with most valid object pixels uncovered by prior queries."""
    best: tuple[int, int, np.ndarray, dict] | None = None
    for index, frame in enumerate(frames):
        if index <= old_birth_indices.max():
            continue
        mask = _read_mask(masks, int(frame), depths[index].shape)
        if mask is None:
            continue
        novel, supported = _novel_pixels(
            mask,
            depths[index],
            old_points[index],
            np.where(old_birth_indices <= index, old_visibility[index], -np.inf),
            intrinsics,
            depth_tolerance_m,
            query_count,
        )
        score = int(novel.sum())
        if best is None or score > best[0]:
            best = (
                score,
                index,
                novel,
                {
                    "mask_pixels": int(mask.sum()),
                    "novel_pixels": score,
                    "old_mask_depth_supported": supported,
                },
            )
    if best is None or best[0] == 0:
        raise ValueError("no newly exposed depth-backed masked surface was found")
    return best[1], best[2], best[3]


def run(args: argparse.Namespace) -> dict:
    if args.depth_tolerance_m <= 0 or args.query_count < 3:
        raise ValueError("depth tolerance and query count must be positive")
    if args.output.exists():
        raise FileExistsError(f"experiment output already exists: {args.output}")
    with np.load(args.baseline, allow_pickle=False) as baseline:
        frames = baseline["frames"]
        old_queries = baseline["query_points_2d"]
        old_points = baseline["query_coords_m"]
        old_visibility = baseline["query_visibility_logits"]
        metadata = json.loads(str(baseline["metadata"]))
        roles = baseline["roles"].tolist()
        query_roles = (
            baseline["query_roles"]
            if "query_roles" in baseline
            else np.concatenate([np.repeat(role, int(metadata["query_counts"][role])) for role in roles])
        )
        original_query_count = (
            int(baseline["original_query_count"]) if "original_query_count" in baseline else len(old_queries)
        )
    if args.role not in roles:
        raise ValueError(f"role {args.role!r} is not in baseline {roles}")
    old_role_indices = np.flatnonzero(query_roles == args.role)
    count = len(old_role_indices)
    if not count or len(query_roles) != len(old_queries):
        raise ValueError("baseline point roles do not match its queries")
    rgbs, depths = [], []
    for frame in frames:
        name = f"{frame:06d}"
        rgb = cv2.imread(str(args.recording / "rgb" / f"{name}.jpg"))
        depth = cv2.imread(str(args.recording / "depth" / f"{name}.png"), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None or rgb.shape[:2] != depth.shape:
            raise ValueError(f"missing aligned RGB-D frame {frame}")
        rgbs.append(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
        depths.append(depth.astype(np.float32) * 0.001)
    rgbs = np.stack(rgbs)
    depths = np.stack(depths)
    intrinsics = np.loadtxt(args.recording / "cam_K.txt").reshape(3, 3)
    birth_index, novel, selection = select_birth_frame(
        frames,
        old_points[:, old_role_indices],
        old_visibility[:, old_role_indices],
        old_queries[old_role_indices, 0].astype(int),
        depths,
        args.masks,
        intrinsics,
        args.depth_tolerance_m,
        args.query_count,
    )
    new_xy = _query_pixels(novel, depths[birth_index], args.query_count)
    new_queries = np.column_stack((np.full(len(new_xy), birth_index), new_xy))
    queries = np.concatenate((old_queries, new_queries))
    expanded_roles = np.concatenate((query_roles, np.repeat(args.role, len(new_queries))))
    sys.path.insert(0, str(args.repository.parent))
    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415

    model = TrackEverything.from_pretrained(str(args.checkpoint), device="cuda")
    started = time.perf_counter()
    with torch.inference_mode():
        tracked = model.infer(
            video=rgbs,
            depths=depths,
            intrinsics=np.repeat(intrinsics[None], len(frames), axis=0),
            extrinsics=np.repeat(np.eye(4)[None], len(frames), axis=0),
            query_points_2d=queries,
            mode="dense",
        )
    inference_s = time.perf_counter() - started
    points = tracked.query_coords[0].float().cpu().numpy() * tracked.norm_factor
    visibility = tracked.query_visibs[0].float().cpu().numpy()
    if points.shape != (len(frames), len(queries), 3) or visibility.shape != points.shape[:2]:
        raise ValueError("TrackEverything returned an unexpected point trace")
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.masks / f"{frames[birth_index]:06d}.png", args.output / "seed-mask.png")
    np.savez_compressed(
        args.output / "expanded-points.npz",
        frames=frames,
        roles=np.asarray(roles),
        query_roles=expanded_roles,
        query_points_2d=queries,
        query_coords_m=points,
        query_visibility_logits=visibility,
        original_query_count=original_query_count,
        new_query_role=np.asarray(args.role),
        norm_factor=np.asarray(tracked.norm_factor),
        metadata=np.asarray(
            json.dumps(
                {
                    "query_counts": {role: int((expanded_roles == role).sum()) for role in roles},
                    "source": "multi_view_visible_surface_queries",
                }
            )
        ),
    )
    rows = []
    new_slice = slice(len(old_queries), len(queries))
    for index, frame in enumerate(frames):
        mask = _read_mask(args.masks, int(frame), depths[index].shape)
        if mask is None:
            continue
        row = {"frame": int(frame), "mask_pixels": int(mask.sum())}
        for label, part in (("old", old_role_indices), ("new", new_slice)):
            observed, _, _ = _supported(
                points[index, part],
                visibility[index, part],
                mask,
                depths[index],
                intrinsics,
                args.depth_tolerance_m,
            )
            births = queries[part, 0].astype(int)
            row[f"{label}_visible"] = int(((visibility[index, part] > 0) & (births <= index)).sum())
            row[f"{label}_mask_depth_supported"] = int((observed & (births <= index)).sum())
        rows.append(row)
    with (args.output / "coverage.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    after_birth = [row for row in rows if row["frame"] >= int(frames[birth_index])]
    sample_indices = sorted({0, len(after_birth) // 2, len(after_birth) - 1})
    report = {
        "role": args.role,
        "birth_frame": int(frames[birth_index]),
        "new_queries": len(new_queries),
        "old_queries": count,
        "selection": selection,
        "inference_s": round(inference_s, 3),
        "depth_tolerance_m": args.depth_tolerance_m,
        "baseline": str(args.baseline),
        "mask_source": str(args.masks),
        "coverage_samples": [after_birth[index] for index in sample_indices],
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True, help="Directory of numbered object masks")
    parser.add_argument("--role", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--depth-tolerance-m", type=float, default=0.01)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
