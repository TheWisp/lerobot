"""Prepare and score visible-surface point tracking experiments.

This deliberately scores raw predictions before any SAM/depth/rigid-fit gate.
Visibility and truth belong to the reference, never to the tracker being tested.
Run ``python benchmarks/flash_surface_benchmark.py --help`` for commands.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "flash-visible-points-v1"
MM = 1000.0


def _quantiles(values: np.ndarray) -> dict[str, float] | None:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(np.max(values)),
    }


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_queries(mask: np.ndarray, depth: np.ndarray, count: int) -> np.ndarray:
    """Spatially spread reference pixels; an input choice, never truth."""
    if count < 1:
        raise ValueError("point count must be positive")
    valid = (mask > 0) & np.isfinite(depth) & (depth > 0)
    ys, xs = np.nonzero(valid)
    if len(xs) < count:
        raise ValueError(f"only {len(xs)} depth-backed mask pixels for {count} queries")
    side = int(np.ceil(np.sqrt(count)))
    targets = np.array(
        np.meshgrid(np.linspace(xs.min(), xs.max(), side), np.linspace(ys.min(), ys.max(), side))
    )
    targets = targets.reshape(2, -1).T
    chosen: list[int] = []
    used: set[int] = set()
    for target in targets:
        order = np.argsort((xs - target[0]) ** 2 + (ys - target[1]) ** 2, kind="stable")
        index = next(int(i) for i in order if int(i) not in used)
        used.add(index)
        chosen.append(index)
        if len(chosen) == count:
            break
    if len(chosen) < count:
        raise ValueError("spatial query selection could not fill the requested budget")
    return np.column_stack((xs[chosen], ys[chosen])).astype(np.float32)


def prepare_local(recording: Path, mask_dir: Path, first: int, last: int, count: int, output: Path) -> dict:
    """Create immutable source/query manifest without inventing track labels."""
    if first < 0 or last < first:
        raise ValueError("invalid frame interval")
    frames = np.arange(first, last + 1)
    rgb_paths = [recording / "rgb" / f"{frame:06d}.jpg" for frame in frames]
    depth_paths = [recording / "depth" / f"{frame:06d}.png" for frame in frames]
    mask_paths = [mask_dir / f"{frame:06d}.png" for frame in frames]
    for path in (*rgb_paths, *depth_paths, *mask_paths, recording / "cam_K.txt", recording / "times.txt"):
        if not path.is_file():
            raise FileNotFoundError(path)
    rgb = cv2.imread(str(rgb_paths[0]), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_paths[0]), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(mask_paths[0]), cv2.IMREAD_GRAYSCALE)
    if (
        rgb is None
        or depth is None
        or mask is None
        or rgb.shape[:2] != depth.shape
        or depth.shape != mask.shape
    ):
        raise ValueError("initial RGB/depth/mask do not align")
    if depth.dtype != np.uint16:
        raise ValueError("local recording depth must be uint16 millimetres")
    xy = select_queries(mask, depth, count)
    queries = np.column_stack((np.zeros(count), xy)).astype(np.float32)
    times = np.loadtxt(recording / "times.txt", dtype=float)
    if len(times) <= last or not np.all(np.diff(times[first : last + 1]) > 0):
        raise ValueError("recording timestamps are missing or unordered")
    intrinsics = np.loadtxt(recording / "cam_K.txt").reshape(3, 3)
    if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
        raise ValueError("invalid camera focal length")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "queries.npz",
        frames=frames,
        capture_s=times[frames] - times[first],
        queries=queries,
        intrinsics=intrinsics,
        initial_depth_m=depth[xy[:, 1].astype(int), xy[:, 0].astype(int)] * 0.001,
    )
    manifest = {
        "schema": SCHEMA,
        "recording": str(recording.resolve()),
        "mask_dir": str(mask_dir.resolve()),
        "first_frame": first,
        "last_frame": last,
        "query_count": count,
        "source_hw": list(depth.shape),
        "depth_unit": "uint16 millimetres",
        "queries_sha256": _digest(output / "queries.npz"),
        "first_rgb_sha256": _digest(rgb_paths[0]),
        "first_depth_sha256": _digest(depth_paths[0]),
        "first_mask_sha256": _digest(mask_paths[0]),
        "ground_truth": "unlabelled; mask/depth are diagnostics only",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def validate_prediction(reference: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> None:
    frames, queries = reference["frames"], reference["queries"]
    if not np.array_equal(prediction["frames"], frames) or not np.array_equal(prediction["queries"], queries):
        raise ValueError("prediction must retain exact source frames and query identities")
    t, n = len(frames), len(queries)
    if prediction["xy"].shape != (t, n, 2) or prediction["visible"].shape != (t, n):
        raise ValueError("prediction has wrong point or visibility shape")
    if "xyz" in prediction and prediction["xyz"].shape != (t, n, 3):
        raise ValueError("prediction has wrong 3D point shape")
    if prediction["visible"].dtype != bool:
        raise ValueError("visible must contain boolean predictions")


def _eligibility(reference: dict[str, np.ndarray]) -> np.ndarray:
    if "truth_visible" not in reference:
        raise ValueError("independent truth_visible required for accuracy scoring")
    frames, queries = reference["frames"], reference["queries"]
    visible = reference["truth_visible"]
    if visible.shape != (len(frames), len(queries)) or visible.dtype != bool:
        raise ValueError("truth_visible must be boolean [frames, queries]")
    if "truth_xy" not in reference or reference["truth_xy"].shape != (len(frames), len(queries), 2):
        raise ValueError("independent truth_xy required for accuracy scoring")
    eligible = visible & (np.arange(len(frames))[:, None] >= queries[:, 0].astype(int))
    if not np.isfinite(reference["truth_xy"][eligible]).all():
        raise ValueError("visible reference points need finite 2D labels")
    return eligible


def score(reference: dict[str, np.ndarray], prediction: dict[str, np.ndarray]) -> dict:
    """Missing visible predictions stay in the denominator for every threshold."""
    validate_prediction(reference, prediction)
    eligible = _eligibility(reference)
    predicted = prediction["visible"] & np.isfinite(prediction["xy"]).all(axis=-1)
    numerator = eligible & predicted
    count = int(eligible.sum())
    if not count:
        raise ValueError("reference contains no eligible visible points")
    error_px = np.linalg.norm(prediction["xy"] - reference["truth_xy"], axis=-1)
    result = {
        "eligible_point_frames": count,
        "predicted_visible": int(numerator.sum()),
        "available_fraction": float(numerator.sum() / count),
        "error_px_when_available": _quantiles(error_px[numerator]),
        "coverage_px": {
            str(limit): float(np.count_nonzero(numerator & (error_px <= limit)) / count)
            for limit in (1, 2, 4, 8)
        },
    }
    if "truth_xyz" in reference and "xyz" in prediction:
        if reference["truth_xyz"].shape != prediction["xyz"].shape:
            raise ValueError("3D truth has wrong shape")
        # Invalid reference depth is separately reported; never silently scored as zero error.
        eligible_3d = eligible & np.isfinite(reference["truth_xyz"]).all(axis=-1)
        valid_3d = eligible_3d & predicted & np.isfinite(prediction["xyz"]).all(axis=-1)
        error_mm = np.linalg.norm(prediction["xyz"] - reference["truth_xyz"], axis=-1) * MM
        denominator = int(eligible_3d.sum())
        result["eligible_3d_point_frames"] = denominator
        result["3d_reference_fraction"] = float(denominator / count)
        result["error_mm_when_available"] = _quantiles(error_mm[valid_3d])
        result["coverage_mm"] = {
            str(limit): float(np.count_nonzero(valid_3d & (error_mm <= limit)) / denominator)
            if denominator
            else None
            for limit in (1, 2, 5, 10, 20)
        }
    return result


def _project(points: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = points[..., :2] / points[..., 2:3]
    return xy * [intrinsics[0, 0], intrinsics[1, 1]] + [intrinsics[0, 2], intrinsics[1, 2]]


def lift_measured_depth(xy: np.ndarray, depth_m: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Lift predicted pixels with aligned measured depth, preserving invalid samples."""
    xyz = np.full((len(xy), 3), np.nan, dtype=np.float32)
    finite = np.isfinite(xy).all(axis=1)
    x = np.rint(np.nan_to_num(xy[:, 0], nan=-1)).astype(int)
    y = np.rint(np.nan_to_num(xy[:, 1], nan=-1)).astype(int)
    valid = finite & (x >= 0) & (x < depth_m.shape[1]) & (y >= 0) & (y < depth_m.shape[0])
    ids = np.flatnonzero(valid)
    z = depth_m[y[ids], x[ids]]
    ids = ids[np.isfinite(z) & (z > 0)]
    z = depth_m[y[ids], x[ids]]
    xyz[ids] = np.column_stack(
        (
            (xy[ids, 0] - intrinsics[0, 2]) * z / intrinsics[0, 0],
            (xy[ids, 1] - intrinsics[1, 2]) * z / intrinsics[1, 1],
            z,
        )
    )
    return xyz


def diagnostic(
    recording: Path,
    mask_dir: Path,
    reference: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    tolerance_m: float,
) -> dict:
    """Predicted-surface check; never substitute for independently labelled accuracy."""
    validate_prediction(reference, prediction)
    if "xyz" not in prediction:
        return {"status": "3d output unavailable"}
    points, frames = prediction["xyz"], reference["frames"]
    geometry_source = str(prediction.get("geometry_source", "unknown"))
    xy = prediction["xy"]
    t, n = points.shape[:2]
    in_mask = np.zeros((t, n), bool)
    depth_valid = np.zeros((t, n), bool)
    depth_gap_mm = np.full((t, n), np.nan)
    for index, frame in enumerate(frames):
        depth = cv2.imread(str(recording / "depth" / f"{frame:06d}.png"), cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(str(mask_dir / f"{frame:06d}.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None or mask is None or depth.shape != mask.shape:
            raise ValueError(f"missing aligned depth/mask at frame {frame}")
        x = np.rint(np.nan_to_num(xy[index, :, 0], nan=-1)).astype(int)
        y = np.rint(np.nan_to_num(xy[index, :, 1], nan=-1)).astype(int)
        valid = (
            prediction["visible"][index]
            & np.isfinite(points[index]).all(axis=1)
            & (points[index, :, 2] > 0)
            & (x >= 0)
            & (x < depth.shape[1])
            & (y >= 0)
            & (y < depth.shape[0])
        )
        ids = np.flatnonzero(valid)
        in_mask[index, ids] = mask[y[ids], x[ids]] > 0
        depth_valid[index, ids] = depth[y[ids], x[ids]] > 0
        depth_gap_mm[index, ids] = np.abs(depth[y[ids], x[ids]] * 0.001 - points[index, ids, 2]) * MM
    births = reference["queries"][:, 0].astype(int)
    born = np.arange(t)[:, None] >= births
    supported = born & in_mask & depth_valid & (depth_gap_mm <= tolerance_m * MM)
    return {
        "status": "diagnostic only; mask and depth cannot prove point identity",
        "geometry_source": geometry_source,
        "eligible_birth_to_end": int(born.sum()),
        "predicted_visible_fraction": float(np.count_nonzero(born & prediction["visible"]) / born.sum()),
        "mask_fraction": float(np.count_nonzero(born & in_mask) / born.sum()),
        "mask_and_depth_fraction": float(np.count_nonzero(born & in_mask & depth_valid) / born.sum()),
        "within_depth_tolerance_fraction": (
            float(np.count_nonzero(supported) / born.sum()) if geometry_source == "model_3d" else None
        ),
        "depth_gap_mm_when_masked": (
            _quantiles(depth_gap_mm[born & in_mask & depth_valid]) if geometry_source == "model_3d" else None
        ),
    }


def timing(
    capture_s: np.ndarray, source_indices: np.ndarray, publication_s: np.ndarray, consumer_hz: float = 30.0
) -> dict:
    """Replay a timestamped publication trace against an independent consumer."""
    capture_s = np.asarray(capture_s, float)
    source_indices = np.asarray(source_indices, int)
    publication_s = np.asarray(publication_s, float)
    if consumer_hz <= 0 or len(capture_s) == 0 or len(publication_s) != len(source_indices):
        raise ValueError("invalid timing trace")
    if np.any(np.diff(capture_s) < 0) or np.any(np.diff(publication_s) < 0):
        raise ValueError("capture/publication times must be ordered")
    if np.any(np.diff(source_indices) < 0):
        raise ValueError("publication source indices must not move backward")
    if np.any(source_indices < 0) or np.any(source_indices >= len(capture_s)):
        raise ValueError("publication source index outside capture stream")
    if np.any(publication_s < capture_s[source_indices]):
        raise ValueError("result cannot precede its source frame")
    steps = np.arange(int(np.ceil(max(capture_s[-1], publication_s[-1]) * consumer_hz)) + 1) / consumer_hz
    latest_publication = np.searchsorted(publication_s, steps, side="right") - 1
    fresh = latest_publication >= 0
    effective_age = steps[fresh] - capture_s[source_indices[latest_publication[fresh]]]
    unique = np.concatenate(([True], source_indices[1:] != source_indices[:-1]))
    gaps = np.diff(publication_s[unique])
    return {
        "source_frame_count": len(capture_s),
        "publications": len(publication_s),
        "unique_source_publications": int(unique.sum()),
        "unique_publication_fps": float(
            unique.sum() / max(publication_s[-1] - publication_s[0], 1 / consumer_hz)
        ),
        "publication_age_ms": _quantiles((publication_s - capture_s[source_indices]) * 1000),
        "effective_consumer_age_ms": _quantiles(effective_age * 1000),
        "update_gap_ms": _quantiles(gaps * 1000),
        "consumer_ticks_before_first_result": int((~fresh).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-local")
    prepare.add_argument("--recording", type=Path, required=True)
    prepare.add_argument("--masks", type=Path, required=True)
    prepare.add_argument("--first", type=int, required=True)
    prepare.add_argument("--last", type=int, required=True)
    prepare.add_argument("--points", type=int, default=64)
    prepare.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--experiment", type=Path, required=True)
    evaluate.add_argument("--prediction", type=Path, required=True)
    evaluate.add_argument("--depth-tolerance-m", type=float, default=0.01)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-local":
        report = prepare_local(args.recording, args.masks, args.first, args.last, args.points, args.output)
    else:
        experiment = args.experiment
        manifest = json.loads((experiment / "manifest.json").read_text())
        reference = _load(experiment / "queries.npz")
        prediction = _load(args.prediction)
        report = {"schema": SCHEMA, "manifest": manifest, "prediction": str(args.prediction.resolve())}
        report["diagnostic"] = diagnostic(
            Path(manifest["recording"]),
            Path(manifest["mask_dir"]),
            reference,
            prediction,
            args.depth_tolerance_m,
        )
        if "truth_visible" in reference:
            report["accuracy"] = score(reference, prediction)
        if "publication_s" in prediction:
            report["timing"] = timing(
                reference["capture_s"], prediction["publication_source_index"], prediction["publication_s"]
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
