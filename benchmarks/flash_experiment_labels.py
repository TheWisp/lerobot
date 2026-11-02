"""Evaluation-only pose labels. Never import these readers into a tracker.

Case ``evaluation`` maps object roles to explicit label files. Poses map object
coordinates into the label camera; ``camera_from_label_camera`` converts them
to the RGB-D camera. Rigid point labels already use metric camera coordinates.
"""

from __future__ import annotations

import hashlib

import numpy as np

from benchmarks.flash_experiment_inputs import path


def _validate_poses(poses):
    poses = np.asarray(poses, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError("evaluation poses must be finite 4x4 transforms")
    rotation = poses[:, :3, :3]
    # Allow small non-orthogonality in supplied calibration/pose matrices
    # (Laval's determinant differs from 1 by 2.4e-5). Preserve their values;
    # this 0.01% file-validity check is not a tracking acceptance threshold.
    if (
        not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-6, rtol=0)
        or not np.allclose(rotation @ rotation.transpose(0, 2, 1), np.eye(3), atol=1e-4, rtol=0)
        or not np.allclose(np.linalg.det(rotation), 1, atol=1e-4, rtol=0)
    ):
        raise ValueError("evaluation labels contain a non-rigid transform")
    return poses


def _rigid_point_motion(points):
    """Recover exact motion from labelled identities, not predicted matches."""
    if points.ndim != 3 or points.shape[2] != 3 or points.shape[1] < 3 or not np.isfinite(points).all():
        raise ValueError("rigid point labels require at least three finite points per frame")
    source = points[0]
    centered = source - source.mean(axis=0)
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError("collinear point labels cannot define a full rigid pose")
    poses = []
    for target in points:
        u, _, vt = np.linalg.svd(centered.T @ (target - target.mean(axis=0)))
        rotation = vt.T @ np.diag([1, 1, np.linalg.det(vt.T @ u.T)]) @ u.T
        translation = target.mean(axis=0) - rotation @ source.mean(axis=0)
        # Generated float32 labels should agree to a micrometre. Do not turn
        # noisy/deforming point annotations into supposedly exact pose truth.
        if np.max(np.linalg.norm(source @ rotation.T + translation - target, axis=1)) > 1e-6:
            raise ValueError("point labels do not describe exact rigid motion")
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = rotation, translation
        poses.append(pose)
    return np.asarray(poses)


def load_pose_labels(case, rows):
    labels, provenance = {}, {}
    if case["kind"] == "ycb_annotated_pose":
        files = sorted((path(case["source_root"]) / "annotated_poses").glob("*.txt"))
        start = case["start_frame"]
        files = files[start : start + len(rows)]
        if len(files) != len(rows):
            raise ValueError("missing evaluation pose annotations")
        # Preserve the established YCB evaluation exactly. Some supplied
        # annotation rotations are slightly non-orthogonal; normalizing them
        # or adding a new rejection threshold would change the baseline.
        poses = np.stack([np.loadtxt(f).reshape(4, 4) for f in files])
        for role in case["roles"]:
            labels[role] = poses
            provenance[role] = {
                "format": "ycb_annotated_pose",
                "source": str(path(case["source_root"])),
                "start_frame": start,
                "sha256": hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest(),
            }
    for role, spec in case.get("evaluation", {}).items():
        if role not in case["roles"] or role in labels:
            raise ValueError(f"unknown or duplicate evaluation role: {role}")
        if [row.get("frame") for row in rows] != list(range(len(rows))):
            raise ValueError("saved prediction frames are not consecutive and ordered")
        source = path(spec["path"])
        start = spec.get("start_frame", 0)
        if type(start) is not int or start < 0:
            raise ValueError("evaluation start_frame must be a nonnegative integer")
        indices = np.arange(start, start + len(rows))
        if spec["format"] == "poses_npy":
            poses = np.load(source, allow_pickle=False).reshape(-1, 4, 4)[start : start + len(rows)]
        elif spec["format"] == "rigid_points_npz":
            with np.load(source, allow_pickle=False) as data:
                if not np.array_equal(data["frames"][start : start + len(rows)], indices):
                    raise ValueError("point labels and predictions use different frame indices")
                times = data["capture_s"][start : start + len(rows)]
                recorded = np.asarray([row["time_s"] for row in rows])
                if not np.allclose(times - times[0], recorded - recorded[0], atol=1e-6, rtol=0):
                    raise ValueError("point labels and predictions use different timestamps")
                poses = _rigid_point_motion(data["truth_xyz"][start : start + len(rows)].astype(float))
        else:
            raise ValueError(f"unsupported evaluation format: {spec['format']}")
        if len(poses) != len(rows):
            raise ValueError("missing evaluation pose annotations")
        conversion = _validate_poses(np.asarray(spec.get("camera_from_label_camera", np.eye(4)))[None])[0]
        labels[role] = _validate_poses(conversion @ poses)
        provenance[role] = {
            **spec,
            "path": str(source),
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    return labels, provenance
