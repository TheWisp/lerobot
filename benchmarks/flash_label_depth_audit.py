"""Check a supplied mesh/pose against recorded depth, independently of tracking.

This measures observation agreement, not pose accuracy. Model error, label error,
camera calibration, mask error and sensor noise can all contribute. Only the
frontmost mesh surface is compared, at pixels inside the supplied object mask.
The mask must describe the visible object; rendered dataset masks can include
unmodelled occluders such as a sleeve. Missing depth is counted, never filled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from numba import njit

from benchmarks.flash_experiment_cache import file_hash
from benchmarks.flash_experiment_inputs import samples
from benchmarks.flash_experiment_labels import load_pose_labels


@njit(cache=True)
def render_depth(vertices, faces, intrinsics, height, width):
    """Triangle z-buffer at integer pixel centres, with perspective-correct depth.

    Vertices must already be in camera coordinates and wholly in front of it.
    Both triangle windings are accepted; back surfaces lose to the nearest one.
    """
    if np.any(vertices[:, 2] <= 0):
        raise ValueError("Audit requires a mesh wholly in front of the camera")
    depth = np.full((height, width), np.inf)
    uv = vertices @ intrinsics.T
    for i in range(len(uv)):
        uv[i, 0] /= uv[i, 2]
        uv[i, 1] /= uv[i, 2]
    for face in faces:
        a, b, c = uv[face[0]], uv[face[1]], uv[face[2]]
        za, zb, zc = vertices[face[0], 2], vertices[face[1], 2], vertices[face[2], 2]
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        xmin = max(0, int(np.ceil(min(a[0], b[0], c[0]))))
        xmax = min(width - 1, int(np.floor(max(a[0], b[0], c[0]))))
        ymin = max(0, int(np.ceil(min(a[1], b[1], c[1]))))
        ymax = min(height - 1, int(np.floor(max(a[1], b[1], c[1]))))
        for y in range(ymin, ymax + 1):
            for x in range(xmin, xmax + 1):
                w0 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det
                w1 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det
                w2 = 1 - w0 - w1
                if min(w0, w1, w2) < -1e-9:
                    continue
                z = 1 / (w0 / za + w1 / zb + w2 / zc)
                if z < depth[y, x]:
                    depth[y, x] = z
    return depth


def residuals(rendered, measured, mask, border_px=0):
    if border_px < 0:
        raise ValueError("Mask border must be nonnegative")
    mask = cv2.erode(
        np.asarray(mask, np.uint8), np.ones((2 * border_px + 1, 2 * border_px + 1), np.uint8)
    ).astype(bool)
    sensor = mask & np.isfinite(measured) & (measured > 0)
    common = sensor & np.isfinite(rendered) & (rendered > 0)
    counts = {
        "mask_pixels": int(mask.sum()),
        "depth_valid_pixels": int(sensor.sum()),
        "mesh_and_depth_pixels": int(common.sum()),
        "valid_depth_without_mesh_pixels": int((sensor & ~common).sum()),
    }
    return (rendered[common] - measured[common]) * 1000, counts


def statistics(values_mm):
    values = np.asarray(values_mm)
    if not len(values):
        return {"pixels": 0}
    absolute = np.abs(values)
    return {
        "pixels": len(values),
        "signed_median_mm": float(np.median(values)),
        "median_mm": float(np.median(absolute)),
        "p90_mm": float(np.percentile(absolute, 90)),
        **{f"within_{t}mm_pct": float(100 * np.mean(absolute <= t)) for t in (2, 5, 10, 20)},
    }


def audit(case, mesh_path, mesh_unit_m, role, output, borders=(0, 2)):
    import trimesh

    if not np.isfinite(mesh_unit_m) or mesh_unit_m <= 0:
        raise ValueError("Specify the mesh's positive metres-per-unit scale")
    mesh = trimesh.load(mesh_path, process=False, skip_materials=True)
    vertices = np.asarray(mesh.vertices) * mesh_unit_m
    faces = np.asarray(mesh.faces)
    # Evaluation labels remain outside the tracker; the mesh and poses must use
    # the same object origin. No registration is performed to improve this audit.
    rows = [{"frame": f.frame_id, "time_s": f.source_time_s} for f, _ in samples(case)]
    labels, provenance = load_pose_labels(case, rows)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {str(b): [] for b in borders}
    results = []
    for frame, masks in samples(case):
        pose = labels[role][frame.frame_id]
        depth = frame.metric_depth()
        rendered = render_depth(
            vertices @ pose[:3, :3].T + pose[:3, 3], faces, frame.intrinsics, *depth.shape
        )
        result = {"frame": frame.frame_id}
        for border in borders:
            values, counts = residuals(rendered, depth, masks[role], border)
            aggregate[str(border)].append(values)
            result[str(border)] = {**statistics(values), **counts}
        results.append(result)
    report = {
        "metric": "Frontmost mesh depth minus recorded depth at the same pixel; not pose accuracy",
        "mask": "Input object mask; its errors and unmodelled occluders can contaminate this check",
        "mask_border_px": list(borders),
        "missing_depth": "Excluded and counted; never filled",
        "case": case,
        "labels": provenance,
        "mesh": {"path": str(mesh_path), "sha256": file_hash(mesh_path), "unit_m": mesh_unit_m},
        "aggregate": {k: statistics(np.concatenate(v)) for k, v in aggregate.items()},
        "frames": results,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("mesh_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mesh-unit-m", type=float, required=True)
    parser.add_argument("--role", default="object")
    args = vars(parser.parse_args())
    args["case"] = json.loads(args["case"].read_text())
    print(json.dumps(audit(**args)["aggregate"], indent=2))


if __name__ == "__main__":
    main()
