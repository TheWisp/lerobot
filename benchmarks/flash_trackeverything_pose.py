"""Fit object poses from TrackEverything query tracks on a recorded RGB-D demo.

This is offline, windowed inference. A result for an early frame may use later
frames in the model's window; it is suitable for demo extraction experiments,
not a causal live pose source. The output follows FLASH's camera-relative trace
contract (frames, poses, lost) and retains fitting diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from lerobot.policies.flash.registration import DegenerateRegistrationError, register_object


def _query_pixels(mask: np.ndarray, depth: np.ndarray, count: int) -> np.ndarray:
    """Select depth-backed mask pixels across its spatial extent, deterministically."""
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < 3:
        raise ValueError("reference mask needs at least three depth-backed pixels")
    pixels = np.column_stack((xs, ys))
    side = max(1, int(np.ceil(np.sqrt(count))))
    grid_x = np.linspace(xs.min(), xs.max(), side)
    grid_y = np.linspace(ys.min(), ys.max(), side)
    chosen: set[int] = set()
    for y in grid_y:
        for x in grid_x:
            chosen.add(int(np.argmin((xs - x) ** 2 + (ys - y) ** 2)))
    return pixels[sorted(chosen)][:count].astype(np.float32)


def run(
    recording: Path,
    mask_path: Path | None,
    *,
    repository: Path,
    checkpoint: Path,
    first_frame: int,
    last_frame: int,
    output: Path,
    query_count: int,
    fit_threshold_m: float,
    min_inliers: int,
    fit_trials: int,
    reference_specs: list[dict] | None = None,
    mode: str = "dense",
) -> dict:
    if mode not in ("dense", "sparse"):
        raise ValueError("TrackEverything mode must be dense or sparse")
    if not 0 <= first_frame <= last_frame or query_count < min_inliers or min_inliers < 3:
        raise ValueError("invalid demo span or fit support settings")
    if not checkpoint.is_file() or not (repository / "demo_hf.py").is_file():
        raise ValueError("TrackEverything source and checkpoint must be installed locally")
    sys.path.insert(0, str(repository.parent))
    from trackeverything.demo_hf import TrackEverything  # noqa: PLC0415

    frames = np.arange(first_frame, last_frame + 1)
    rgbs, depths = [], []
    for index in frames:
        stem = f"{index:06d}"
        bgr = cv2.imread(str(recording / "rgb" / f"{stem}.jpg"))
        raw_depth = cv2.imread(str(recording / "depth" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        if bgr is None or raw_depth is None or bgr.shape[:2] != raw_depth.shape:
            raise ValueError(f"missing aligned RGB-D frame {index}")
        rgbs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        depths.append(raw_depth.astype(np.float32) * 0.001)
    specs = reference_specs or [{"role": "object", "frame_index": first_frame, "mask_path": str(mask_path)}]
    if len({spec["role"] for spec in specs}) != len(specs):
        raise ValueError("reference object names must be distinct")
    queries_by_role = []
    for spec in specs:
        reference_index = int(spec["frame_index"]) - first_frame
        if not 0 <= reference_index < len(frames):
            raise ValueError("object reference frame is outside the video")
        reference_mask = cv2.imread(str(spec["mask_path"]), cv2.IMREAD_GRAYSCALE)
        if reference_mask is None or reference_mask.shape != depths[reference_index].shape:
            raise ValueError("reference mask is missing or does not align with RGB-D")
        query_xy = _query_pixels(reference_mask > 0, depths[reference_index], query_count)
        queries_by_role.append(np.column_stack((np.full(len(query_xy), reference_index), query_xy)))
    queries = np.concatenate(queries_by_role)
    intrinsics = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    cameras = np.repeat(np.eye(4)[None], len(frames), axis=0)
    model = TrackEverything.from_pretrained(str(checkpoint), device="cuda")
    start = time.perf_counter()
    with torch.inference_mode():
        tracked = model.infer(
            video=np.stack(rgbs),
            depths=np.stack(depths),
            intrinsics=np.repeat(intrinsics[None], len(frames), axis=0),
            extrinsics=cameras,
            query_points_2d=queries,
            mode=mode,
        )
    elapsed_s = time.perf_counter() - start
    points = tracked.query_coords[0].float().cpu().numpy() * tracked.norm_factor
    visibility = tracked.query_visibs[0].float().cpu().numpy()
    if points.shape != (len(frames), len(queries), 3) or visibility.shape != points.shape[:2]:
        raise ValueError("TrackEverything returned an unexpected query-track shape")
    traces, offset = [], 0
    for spec, role_queries in zip(specs, queries_by_role, strict=True):
        end = offset + len(role_queries)
        object_points = points[:, offset:end]
        object_visibility = visibility[:, offset:end]
        trace = _fit_tracks(
            object_points,
            object_visibility,
            int(spec["frame_index"]) - first_frame,
            fit_threshold_m=fit_threshold_m,
            min_inliers=min_inliers,
            fit_trials=fit_trials,
        )
        trace["temporal_conflicts"] = _temporal_motion_conflicts(
            object_points,
            object_visibility,
            trace["poses"],
            trace["lost"],
            reference_index=int(spec["frame_index"]) - first_frame,
            fit_threshold_m=fit_threshold_m,
            min_inliers=min_inliers,
        )
        traces.append(trace)
        offset = end
    arrays = {key: np.stack([trace[key] for trace in traces]) for key in traces[0]}
    if reference_specs is None:
        arrays = {key: value[0] for key, value in arrays.items()}
    else:
        arrays["roles"] = np.asarray([spec["role"] for spec in specs])
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        frames=frames,
        query_points_2d=queries,
        query_coords_m=points,
        query_visibility_logits=visibility,
        **arrays,
        elapsed_s=np.asarray(elapsed_s),
        metadata=np.asarray(
            json.dumps(
                {
                    "tracker": "TrackEverything",
                    "windowed_noncausal": True,
                    "mode": mode,
                    "first_frame": first_frame,
                    "query_counts": {
                        spec["role"]: len(query) for spec, query in zip(specs, queries_by_role, strict=True)
                    },
                }
            )
        ),
    )
    return {
        "frames": len(frames),
        "valid_by_role": {
            spec["role"]: int((~trace["lost"]).sum()) for spec, trace in zip(specs, traces, strict=True)
        },
        "elapsed_s": elapsed_s,
        "output": str(output),
    }


def _fit_tracks(points, visibility, reference_index, *, fit_threshold_m, min_inliers, fit_trials):
    reference_points = points[reference_index]
    poses, lost, inliers, residuals = [], [], [], []
    for index, (current, visible) in enumerate(zip(points, visibility, strict=True)):
        usable = np.isfinite(current).all(axis=1) & np.isfinite(reference_points).all(axis=1)
        usable &= visible > 0  # released API calls these visibility logits
        try:
            if usable.sum() < min_inliers:
                raise DegenerateRegistrationError("too few visible object points")
            fit = register_object(
                reference_points[usable],
                current[usable],
                threshold_m=fit_threshold_m,
                min_inliers=min_inliers,
                trials=fit_trials,
                seed=index,
            )
            poses.append(fit.base_from_demo_object)
            lost.append(False)
            inliers.append(len(fit.inlier_indices))
            residuals.append(fit.median_error_m)
        except (DegenerateRegistrationError, ValueError):
            poses.append(np.eye(4))
            lost.append(True)
            inliers.append(0)
            residuals.append(float("nan"))
    return {
        "poses": np.stack(poses),
        "lost": np.asarray(lost, dtype=bool),
        "inliers": np.asarray(inliers),
        "median_fit_error_m": np.asarray(residuals),
    }


def _temporal_motion_conflicts(
    points: np.ndarray,
    visibility: np.ndarray,
    poses: np.ndarray,
    lost: np.ndarray,
    *,
    reference_index: int,
    fit_threshold_m: float,
    min_inliers: int,
) -> np.ndarray:
    """Flag pose changes contradicted by the tracked motion of prior supporters.

    Each independently fitted pose can be within one registration tolerance of
    a point. Two adjacent poses therefore permit at most twice that discrepancy
    in the point's measured displacement. No robot speed limit is assumed.
    """
    conflicts = np.zeros(len(points), dtype=bool)
    reference = points[reference_index]
    for index in range(reference_index + 1, len(points)):
        if lost[index - 1] or lost[index]:
            continue
        before = reference @ poses[index - 1, :3, :3].T + poses[index - 1, :3, 3]
        after = reference @ poses[index, :3, :3].T + poses[index, :3, 3]
        prior_support = (visibility[index - 1] > 0) & (visibility[index] > 0)
        prior_support &= np.linalg.norm(before - points[index - 1], axis=1) <= fit_threshold_m
        if prior_support.sum() < min_inliers:
            continue
        displacement_error = np.linalg.norm((after - before) - (points[index] - points[index - 1]), axis=1)
        conflicts[index] = np.median(displacement_error[prior_support]) > 2 * fit_threshold_m
    return conflicts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-mask", type=Path)
    source.add_argument("--reference-specs", type=Path, help="JSON list of role, frame_index and mask_path")
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--first-frame", type=int, required=True)
    parser.add_argument("--last-frame", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--query-count", type=int, default=128)
    parser.add_argument("--mode", choices=("dense", "sparse"), default="dense")
    parser.add_argument("--fit-threshold-m", type=float, required=True)
    parser.add_argument("--min-inliers", type=int, required=True)
    parser.add_argument("--fit-trials", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.recording,
                args.reference_mask,
                repository=args.repository,
                checkpoint=args.checkpoint,
                first_frame=args.first_frame,
                last_frame=args.last_frame,
                output=args.output,
                query_count=args.query_count,
                mode=args.mode,
                fit_threshold_m=args.fit_threshold_m,
                min_inliers=args.min_inliers,
                fit_trials=args.fit_trials,
                reference_specs=json.loads(args.reference_specs.read_text())
                if args.reference_specs
                else None,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
