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
from scipy.spatial.transform import Rotation

from .surface_correspondence import SurfaceMotionConfig, crop_pair, robust_fit, ufm_pairs
from .surface_grounding import PreparedReferenceSurface, ShapeConfig, WholeSurfaceFit, project
from .surface_observation import (
    SurfaceFrameFilters,
    SurfaceObservationConfig,
    initial_landmarks,
    move,
    observation,
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
    def __init__(
        self,
        matcher,
        *,
        observation_config=None,
        shape_config=None,
        motion_config=None,
        matcher_lock=None,
    ):
        self.matcher = matcher
        self.matcher_lock = matcher_lock
        self.motion_config = motion_config or SurfaceMotionConfig()
        self.observation_config = observation_config or SurfaceObservationConfig()
        self.shape_config = shape_config or ShapeConfig()
        self.reference = None
        self.previous = None
        self.pose = np.eye(4)
        self.anchors = None
        self.basis = None
        self.reference_frame_id = None
        self.prepared_reference = None

    def update(
        self,
        rgb,
        depth_m,
        mask,
        intrinsics,
        frame_id,
        *,
        frame_filters: SurfaceFrameFilters | None = None,
    ):
        return self.finish(
            self.prepare(rgb, depth_m, mask, intrinsics, frame_id, frame_filters=frame_filters)
        )

    def prepare(self, rgb, depth_m, mask, intrinsics, frame_id, *, frame_filters=None):
        """Measure image motion without reading or changing the fitted pose.

        Called in source order by one producer. Its previous observation belongs
        exclusively to this stage; finish may concurrently fit the prior frame.
        """
        start = time.perf_counter()
        obs = observation(
            depth_m, mask, intrinsics, asdict(self.observation_config), frame_filters=frame_filters
        )
        # Lift only usable pixels; observation computed normals inside the mask bounds.
        y, x = np.nonzero(obs["valid_image"])
        z = depth_m[y, x]
        all_points = np.column_stack(
            ((x - intrinsics[0, 2]) * z / intrinsics[0, 0], (y - intrinsics[1, 2]) * z / intrinsics[1, 1], z)
        )
        spacing = min(obs["spacing_m"], self.shape_config.surface_spacing_m)
        all_points = all_points[voxel_sample(all_points, spacing)]
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
        detail.update(
            surface_points=len(all_points),
            sample_spacing_mm=spacing * 1000,
        )
        prepared = {
            "current": current,
            "intrinsics": intrinsics,
            "frame_id": frame_id,
            "diagnostics": detail,
            "step": None,
            "status": None,
        }
        if not mask.any():
            prepared["status"] = "no_mask"
            return prepared
        points = obs["points"]
        if len(points) < 3 or np.linalg.matrix_rank(points - points.mean(0)) < 2:
            prepared["status"] = "insufficient_depth"
            return prepared
        detail["preprocess_ms"] = (time.perf_counter() - start) * 1000
        if self.previous is not None:
            start = time.perf_counter()
            a, b, crop = crop_pair(self.previous, current, self.motion_config)
            match_args = (self.matcher, a, b, crop, self.previous, current, intrinsics, self.motion_config)
            if self.matcher_lock is None:
                source, target = ufm_pairs(*match_args)
            else:
                wait_start = time.perf_counter()
                with self.matcher_lock:
                    detail["matcher_wait_ms"] = (time.perf_counter() - wait_start) * 1000
                    source, target = ufm_pairs(*match_args)
            step, inliers, residual = robust_fit(source, target, self.motion_config)
            detail.update(
                matches=len(source),
                inliers=inliers,
                match_residual_mm=residual * 1000 if np.isfinite(residual) else None,
                match_ms=(time.perf_counter() - start) * 1000,
                matcher_available=step is not None,
            )
            prepared["step"] = step
        self.previous = current
        return prepared

    def finish(self, prepared):
        """Apply prepared motion to the last completed pose, strictly in order."""
        current, intrinsics = prepared["current"], prepared["intrinsics"]
        detail = dict(prepared["diagnostics"])
        detail["reference_frame_id"] = self.reference_frame_id
        if prepared["status"] is not None:
            return {"status": prepared["status"], "diagnostics": detail}
        if self.reference is None:
            self.reference = current["obs"]["points"].copy()
            self.anchors, self.basis = initial_landmarks(self.reference)
            self.reference_frame_id = prepared["frame_id"]
            detail["reference_frame_id"] = self.reference_frame_id
            status = "reference"
        else:
            step = prepared["step"]
            proposal = self.pose if step is None else step @ self.pose
            prior = self.pose.copy()
            start = time.perf_counter()
            if self.prepared_reference is None:
                self.prepared_reference = PreparedReferenceSurface.build(self.reference, self.shape_config)
            self.pose, fit = WholeSurfaceFit(
                self.reference,
                current,
                intrinsics,
                self.shape_config,
                prepared_reference=self.prepared_reference,
            ).correct(proposal)
            detail.update(fit, fit_ms=(time.perf_counter() - start) * 1000)

            def angle(a, b):
                return float(np.degrees(Rotation.from_matrix(a[:3, :3] @ b[:3, :3].T).magnitude()))

            detail.update(
                rgb_rotation_deg=angle(proposal, prior) if step is not None else None,
                surface_correction_deg=angle(self.pose, proposal),
                final_rotation_deg=angle(self.pose, prior),
            )
            status = "fitted" if step is not None else "fitted_without_rgb_motion"
        all_points = current["all_surface_points"]
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
