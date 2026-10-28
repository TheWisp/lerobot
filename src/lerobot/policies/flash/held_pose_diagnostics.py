"""Describe disagreement between observed held poses and a demo's grasp assumption.

These statistics do not choose which estimate is correct. They give the editor
source-time evidence for deciding whether the demonstrated attachment needs
review before its carried-object motion is used as a skill.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .geometry import interpolate, inverse, transform
from .registration import transform_points
from .trajectory import Evidence, PoseEstimate


def _arm_pose_at(times: np.ndarray, poses: np.ndarray, at_s: float) -> np.ndarray | None:
    right = int(np.searchsorted(times, at_s))
    if right == len(times):
        return poses[-1] if at_s == times[-1] else None
    if right == 0:
        return poses[0] if at_s == times[0] else None
    if at_s == times[right]:
        return poses[right]
    fraction = (at_s - times[right - 1]) / (times[right] - times[right - 1])
    return interpolate(poses[right - 1], poses[right], fraction)


def compare_demo_held_pose(
    *,
    role: str,
    estimates: Sequence[Mapping[str, PoseEstimate]],
    arm_times_s: np.ndarray,
    arm_poses: np.ndarray,
    grasp_time_s: float,
    release_time_s: float,
    grasp_attachment: np.ndarray,
    object_center_m: np.ndarray,
) -> dict:
    """Compare camera-frame observations with a carried rigid-grasp prediction.

    Each contiguous run is reported separately: combining observations across
    a hidden slip or a lost tracking interval would make one median misleading.
    The object center is the fixed reference cloud's mean, not a mass center.
    """
    times = np.asarray(arm_times_s, dtype=float)
    poses = np.asarray(arm_poses, dtype=float)
    center = np.asarray(object_center_m, dtype=float)
    if (
        not role
        or len(times) != len(poses)
        or len(times) < 2
        or not np.isfinite(times).all()
        or not np.all(np.diff(times) > 0)
        or center.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite([grasp_time_s, release_time_s]).all()
        or grasp_time_s >= release_time_s
    ):
        raise ValueError("held-pose comparison needs valid motor times, object center and event times")
    attachment = transform(grasp_attachment)
    records: list[dict] = []
    for index, row in enumerate(estimates):
        estimate = row.get(role)
        if (
            estimate is None
            or estimate.evidence != Evidence.OBSERVED
            or estimate.pose is None
            or not grasp_time_s <= estimate.source_time_s < release_time_s
        ):
            continue
        ee = _arm_pose_at(times, poses, estimate.source_time_s)
        if ee is None:
            continue
        observed = transform(estimate.pose)
        predicted = ee @ attachment
        observed_center = transform_points(observed, center[None])[0]
        predicted_center = transform_points(predicted, center[None])[0]
        gripper_center = transform_points(inverse(ee) @ observed, center[None])[0]
        relative_rotation = predicted[:3, :3].T @ observed[:3, :3]
        turn = float(np.arccos(np.clip((np.trace(relative_rotation) - 1) / 2, -1, 1)))
        records.append(
            {
                "frame_index": index,
                "time_s": estimate.source_time_s,
                "center_gap_m": float(np.linalg.norm(observed_center - predicted_center)),
                "rotation_gap_rad": turn,
                "gripper_center_m": gripper_center,
            }
        )

    def summarize_run(run: list[dict]) -> dict:
        center_gaps = [item["center_gap_m"] for item in run]
        turns = [item["rotation_gap_rad"] for item in run]
        centers = np.stack([item["gripper_center_m"] for item in run])
        scatter = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
        return {
            "first_frame": run[0]["frame_index"],
            "last_frame": run[-1]["frame_index"],
            "first_time_s": run[0]["time_s"],
            "last_time_s": run[-1]["time_s"],
            "observed_frames": len(run),
            "center_gap_median_m": float(np.median(center_gaps)),
            "center_gap_p90_m": float(np.percentile(center_gaps, 90)),
            "rotation_gap_median_rad": float(np.median(turns)),
            "rotation_gap_p90_rad": float(np.percentile(turns, 90)),
            "gripper_relative_center_scatter_p90_m": float(np.percentile(scatter, 90)),
        }

    runs: list[dict] = []
    current: list[dict] = []
    for record in records:
        if current and record["frame_index"] != current[-1]["frame_index"] + 1:
            runs.append(summarize_run(current))
            current = []
        current.append(record)
    if current:
        runs.append(summarize_run(current))
    return {"role": role, "observed_frames": len(records), "runs": runs}
