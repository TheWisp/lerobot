"""Experimental vision-only surface tracking, independent of cameras and robot control.

One fixed partial reference is transported by UFM then corrected against the
current SAM mask and measured depth. No complete object, mass center, or
calibrated confidence is inferred. A new reference requires an explicit reset.
"""

from __future__ import annotations

import time
from dataclasses import asdict

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .surface_correspondence import SurfaceMotionConfig, crop_pair, robust_fit, ufm_pairs
from .surface_grounding import ShapeConfig, WholeSurfaceFit, project
from .surface_observation import (
    SurfaceObservationConfig,
    initial_landmarks,
    move,
    observation,
    unproject,
    voxel_sample,
)


def load_ufm():
    """Load the same published checkpoint as the offline comparison.

    Call only in the dedicated inference process: UFM's constructor initializes
    DINO weights that its complete checkpoint immediately replaces.
    """
    import torch
    from uniflowmatch.models.ufm import UniFlowMatchConfidence

    original = torch.hub.load

    def backbone(repo, name, *args, **kwargs):
        if repo == "facebookresearch/dinov2" and name.startswith("dinov2_"):
            kwargs["pretrained"] = False
        return original(repo, name, *args, **kwargs)

    torch.hub.load = backbone
    try:
        model = UniFlowMatchConfidence.from_pretrained("infinity1096/UFM-Base-DINOv2L-init")
    finally:
        torch.hub.load = original
    return model.cuda().float().eval()


class SurfaceTrackingSession:
    def __init__(self, matcher, *, observation_config=None, shape_config=None, motion_config=None):
        self.matcher = matcher
        self.motion_config = motion_config or SurfaceMotionConfig()
        self.observation_config = observation_config or SurfaceObservationConfig()
        self.shape_config = shape_config or ShapeConfig()
        self.reference = None
        self.previous = None
        self.pose = np.eye(4)
        self.anchors = None
        self.basis = None
        self.reference_frame_id = None

    def update(self, rgb, depth_m, mask, intrinsics, frame_id):
        start = time.perf_counter()
        obs = observation(depth_m, mask, intrinsics, asdict(self.observation_config))
        all_points = unproject(depth_m, intrinsics)[obs["valid_image"]]
        all_points = all_points[voxel_sample(all_points, self.shape_config.surface_spacing_m)]
        connected = np.zeros(mask.shape, np.uint8)
        px = obs["pixels"]
        connected[px[:, 1], px[:, 0]] = 1
        connected = (
            cv2.dilate(connected, np.ones((self.motion_config.connected_dilation_px,) * 2, np.uint8)) > 0
        )
        current = {
            "rgb": rgb,
            "depth": depth_m,
            "mask": mask,
            "obs": obs,
            "usable": obs["valid_image"] & connected,
            "all_surface_points": all_points,
        }
        detail = {
            key: obs[key]
            for key in ("raw_count", "boundary_count", "edge_count", "clean_count", "detached_count")
        }
        detail.update(surface_points=len(all_points), reference_frame_id=self.reference_frame_id)
        if not mask.any():
            return {"status": "no_mask", "diagnostics": detail}
        points = obs["points"]
        if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
            return {"status": "insufficient_depth", "diagnostics": detail}
        detail["preprocess_ms"] = (time.perf_counter() - start) * 1000
        if self.reference is None:
            self.reference = points.copy()
            self.anchors, self.basis = initial_landmarks(self.reference)
            self.reference_frame_id = frame_id
            detail["reference_frame_id"] = frame_id
            status = "reference"
        else:
            start = time.perf_counter()
            a, b, crop = crop_pair(self.previous, current, self.motion_config)
            source, target = ufm_pairs(
                self.matcher, a, b, crop, self.previous, current, intrinsics, self.motion_config
            )
            step, inliers, residual = robust_fit(source, target, self.motion_config)
            detail.update(
                matches=len(source),
                inliers=inliers,
                match_residual_mm=residual * 1000 if np.isfinite(residual) else None,
                match_ms=(time.perf_counter() - start) * 1000,
                matcher_available=step is not None,
            )
            proposal = self.pose if step is None else step @ self.pose
            start = time.perf_counter()
            self.pose, fit = WholeSurfaceFit(self.reference, current, intrinsics, self.shape_config).correct(
                proposal
            )
            detail.update(fit, fit_ms=(time.perf_counter() - start) * 1000)
            status = "fitted" if step is not None else "fitted_without_rgb_motion"
        self.previous = current
        predicted = move(self.reference, self.pose)
        distances = cKDTree(predicted).query(all_points)[0] * 1000
        detail.update(
            surface_median_mm=float(np.median(distances)), surface_p90_mm=float(np.percentile(distances, 90))
        )
        anchors = move(self.anchors, self.pose)
        # The display frame is fixed at A, with axes chosen once at initialization.
        # Its size follows the initial cloud extent; it is not a physical object size.
        axis_length = float(np.linalg.norm(np.ptp(self.reference, axis=0)) / 3)
        axes = anchors[0] + (self.pose[:3, :3] @ self.basis).T * axis_length
        object_pose = self.pose.copy()
        object_pose[:3, :3] = self.pose[:3, :3] @ self.basis
        object_pose[:3, 3] = anchors[0]
        return {
            "status": status,
            "diagnostics": detail,
            "camera_from_reference": self.pose.tolist(),
            "camera_from_object": object_pose.tolist(),
            "observed_m": all_points.tolist(),
            "reference_m": predicted.tolist(),
            "anchors_m": anchors.tolist(),
            "axes_m": axes.tolist(),
            "anchors_px": project(anchors, intrinsics).tolist(),
            "axes_px": project(axes, intrinsics).tolist(),
            "reference_px": project(predicted, intrinsics).tolist(),
        }
