"""Score complete Point2Pose poses and existing common point fits on labelled RGB-D clips.

The YCBInEOAT object-pose annotations define relative motion from the first
frame. A cloud error transports identical first-frame measured surface points
with estimated and annotated transforms; it is not a contact-tip error.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

POINT_FITS = ("lk", "litetracker", "bootstapir", "tapnext", "cotracker3", "trackeverything")
MODES = ("oracle_masks", "internal_masks")


def source_points(reference: dict[str, np.ndarray]) -> np.ndarray:
    k, queries, z = (reference[name] for name in ("intrinsics", "queries", "initial_depth_m"))
    if np.any(queries[:, 0] != 0):
        raise ValueError("all query points must start in the source frame")
    return np.column_stack(
        ((queries[:, 1] - k[0, 2]) * z / k[0, 0], (queries[:, 2] - k[1, 2]) * z / k[1, 1], z)
    )


def score_transforms(
    transforms: np.ndarray, available: np.ndarray, truth: np.ndarray, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if transforms.shape != truth.shape or available.shape != (len(truth),):
        raise ValueError("pose trace frame count or shape does not match truth")
    cloud = np.full(len(truth), np.nan)
    angle = np.full(len(truth), np.nan)
    center = np.full(len(truth), np.nan)
    centroid = source.mean(axis=0)
    for i in np.flatnonzero(available):
        estimated = (transforms[i, :3, :3] @ source.T).T + transforms[i, :3, 3]
        expected = (truth[i, :3, :3] @ source.T).T + truth[i, :3, 3]
        cloud[i] = np.median(np.linalg.norm(estimated - expected, axis=1)) * 1000
        center[i] = (
            np.linalg.norm(
                transforms[i, :3, :3] @ centroid
                + transforms[i, :3, 3]
                - truth[i, :3, :3] @ centroid
                - truth[i, :3, 3]
            )
            * 1000
        )
        delta = transforms[i, :3, :3] @ truth[i, :3, :3].T
        angle[i] = np.rad2deg(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))
    return cloud, angle, center


def summarize(available: np.ndarray, cloud: np.ndarray, angle: np.ndarray, center: np.ndarray) -> dict:
    # Initialization is identical by construction and would artificially improve every score.
    valid = available.copy()
    valid[0] = False
    scored = int(valid.sum())
    total = len(valid) - 1
    if not scored:
        return {"available": 0, "total": total}
    return {
        "available": scored,
        "total": total,
        "cloud_mm_median": float(np.median(cloud[valid])),
        "cloud_mm_p90": float(np.quantile(cloud[valid], 0.9)),
        "angle_deg_median": float(np.median(angle[valid])),
        "angle_deg_p90": float(np.quantile(angle[valid], 0.9)),
        "center_mm_median": float(np.median(center[valid])),
        "within_10mm_10deg": int(np.sum(valid & (cloud <= 10) & (angle <= 10))),
        "within_20mm_20deg": int(np.sum(valid & (cloud <= 20) & (angle <= 20))),
        "final_cloud_mm": float(cloud[-1]) if available[-1] else None,
        "final_angle_deg": float(angle[-1]) if available[-1] else None,
    }


def serializable_trace(values: np.ndarray) -> list[float | None]:
    return [float(value) if np.isfinite(value) else None for value in values]


def score_case(case: Path, output_root: Path) -> dict:
    manifest = json.loads((case / "manifest.json").read_text())
    if not str(manifest.get("ground_truth", "")).startswith("YCBInEOAT annotated object poses"):
        raise ValueError(f"case lacks annotated pose truth: {case}")
    with np.load(case / "queries.npz", allow_pickle=False) as loaded:
        reference = {name: loaded[name] for name in loaded.files}
    frames = reference["frames"]
    source = source_points(reference)
    paths = sorted((Path(manifest["source"]) / "annotated_poses").glob("*.txt"))
    poses = np.stack(
        [
            np.loadtxt(paths[int(manifest["source_first_index"]) + int(frame)]).reshape(4, 4)
            for frame in frames
        ]
    )
    truth = poses @ np.linalg.inv(poses[0])
    result: dict = {
        "case": case.name,
        "source": manifest["source"],
        "source_first_index": manifest["source_first_index"],
        "frames": frames.tolist(),
        "reference_visible_fraction": float(np.mean(reference["truth_visible"])),
        "reference_depth_gap_mm_median_when_visible": float(
            np.nanmedian(reference["reference_depth_gap_m"][reference["truth_visible"]]) * 1000
        ),
        "annotated_motion_mm": float(
            np.linalg.norm((truth[-1] @ np.r_[source.mean(axis=0), 1])[:3] - source.mean(axis=0)) * 1000
        ),
        "annotated_motion_deg": float(
            np.rad2deg(np.arccos(np.clip((np.trace(truth[-1, :3, :3]) - 1) / 2, -1, 1)))
        ),
        "methods": {},
    }
    for mode in MODES:
        pose_path = output_root / case.name / mode / "p2p_object_pose.npz"
        if not pose_path.exists():
            provenance = output_root / case.name / mode / "provenance.json"
            if provenance.exists() and json.loads(provenance.read_text()).get("exit_code") != 0:
                lines = (provenance.parent / "run.log").read_text().strip().splitlines()
                result["methods"]["p2p_" + mode] = {
                    "summary": {"available": 0, "total": len(frames) - 1},
                    "failure": lines[-1] if lines else "Point2Pose process failed",
                }
            continue
        with np.load(pose_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["frames"], frames):
                raise ValueError(f"Point2Pose frame mismatch: {pose_path}")
            transforms, available = saved["poses"].copy(), ~saved["lost"].copy()
            milliseconds = saved["ms"].copy()
        cloud, angle, center = score_transforms(transforms, available, truth, source)
        metrics = summarize(available, cloud, angle, center)
        metrics["step_ms_median"] = float(np.median(milliseconds[1:]))
        result["methods"]["p2p_" + mode] = {
            "summary": metrics,
            "cloud_mm": serializable_trace(cloud),
            "angle_deg": serializable_trace(angle),
            "center_mm": serializable_trace(center),
        }
    for name in POINT_FITS:
        fit_path = case / "fits-10mm" / f"{name}.npz"
        if not fit_path.exists():
            continue
        with np.load(fit_path, allow_pickle=False) as saved:
            if not np.array_equal(saved["frames"], frames):
                raise ValueError(f"point-fit frame mismatch: {fit_path}")
            transforms, available = saved["transforms"].copy(), saved["available"].copy()
            prior_cloud = saved["cloud_error_mm"].copy()
            prior_angle = saved["angular_error_deg"].copy()
        cloud, angle, center = score_transforms(transforms, available, truth, source)
        if not np.allclose(cloud[available], prior_cloud[available], atol=0.01) or not np.allclose(
            angle[available], prior_angle[available], atol=0.01
        ):
            raise ValueError(f"shared scoring contract does not reproduce {fit_path}")
        result["methods"][name + "_common_fit"] = {
            "summary": summarize(available, cloud, angle, center),
            "cloud_mm": serializable_trace(cloud),
            "angle_deg": serializable_trace(angle),
            "center_mm": serializable_trace(center),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "schema": "flash-full-pose-benchmark-v1",
        "cases": [score_case(case, args.output_root) for case in args.cases],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
