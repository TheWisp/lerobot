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
from .surface_motion import measured_surface_information
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


def reference_depth_support(
    reference_camera_points: np.ndarray,
    clean_depth_pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[int, float | None]:
    """Compare a transported reference with reliable depth on this object.

    The current-to-reference nearest-neighbour score can be low for a small
    visible patch even when the rest of the saved reference is unsupported.
    This reverse check uses only pixels retained by the existing mask/depth
    observation filter; occluded or invalid pixels supply no positive evidence.
    """
    residual_mm = reference_depth_residual_mm(
        reference_camera_points, clean_depth_pixels, depth_m, intrinsics
    )
    return len(residual_mm), float(np.percentile(np.abs(residual_mm), 90)) if len(residual_mm) else None


def reference_depth_residual_mm(
    reference_camera_points: np.ndarray,
    clean_depth_pixels: np.ndarray,
    depth_m: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Signed frontmost-reference minus measured depth, in millimetres.

    Negative means the saved surface lies in front of measured depth. Positive
    may mean a hidden reference surface, so it is not a free-space conflict.
    Only clean pixels on the measured object's mask are compared.
    """
    positive = np.isfinite(reference_camera_points).all(axis=1) & (reference_camera_points[:, 2] > 0)
    projected = np.rint(project(reference_camera_points[positive], intrinsics)).astype(int)
    points = reference_camera_points[positive]
    height, width = depth_m.shape
    inside = (
        (projected[:, 0] >= 0)
        & (projected[:, 0] < width)
        & (projected[:, 1] >= 0)
        & (projected[:, 1] < height)
    )
    projected = projected[inside]
    points = points[inside]
    # One image pixel is one depth measurement even if the reference contains
    # several points along its ray. Compare the nearest predicted surface.
    linear = projected[:, 1] * width + projected[:, 0]
    pixels, inverse = np.unique(linear, return_inverse=True)
    front_depth = np.full(len(pixels), np.inf)
    np.minimum.at(front_depth, inverse, points[:, 2])
    y, x = np.divmod(pixels, width)
    clean = clean_depth_pixels[y, x]
    return (front_depth[clean] - depth_m[y[clean], x[clean]]) * 1000


class SurfaceTrackingSession:
    def __init__(
        self,
        matcher,
        *,
        observation_config=None,
        shape_config=None,
        motion_config=None,
        matcher_lock=None,
        fit_kernels=None,
        search_pool=None,
        information_correspondence_m=None,
    ):
        self.matcher = matcher
        self.matcher_lock = matcher_lock
        self.fit_kernels = fit_kernels
        self.search_pool = search_pool
        self.motion_config = motion_config or SurfaceMotionConfig()
        self.observation_config = observation_config or SurfaceObservationConfig()
        self.shape_config = shape_config or ShapeConfig()
        self.information_correspondence_m = (
            self.shape_config.depth_tolerance_m
            if information_correspondence_m is None
            else information_correspondence_m
        )
        if not np.isfinite(self.information_correspondence_m) or self.information_correspondence_m <= 0:
            raise ValueError("surface information correspondence distance must be positive")
        self.reference = None
        self.previous = None
        self.pose = np.eye(4)
        # Keep an independent motion-only lineage from the last RGB-D accepted
        # fit. The visual fit remains free to recover after a rejected frame.
        self.last_accepted_pose = None
        self.motion_from_accepted = np.eye(4)
        self.motion_chain_complete = True
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
        result = self.finish(
            self.prepare(rgb, depth_m, mask, intrinsics, frame_id, frame_filters=frame_filters)
        )
        # A standalone inspection session has no outer RGB-D publication gate.
        self.record_publication("camera_from_reference" in result)
        return result

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
            compiled_surface_fit=self.fit_kernels is not None,
            reused_search_workers=self.search_pool.workers if self.search_pool is not None else 0,
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
        # A small visible sliver can match a reference with a low nearest-surface
        # residual while leaving its orientation effectively unconstrained.
        # Measure two independent spans of the *clean connected* observation;
        # the second is compared with the fit's own error before claiming a
        # fully observed rigid pose. No object axis or nominal size is assumed.
        centered = points - points.mean(axis=0)
        variances, directions = np.linalg.eigh(centered.T @ centered / len(points))
        detail["visible_lateral_spread_mm"] = float(np.sqrt(max(variances[-2], 0.0)) * 1000)
        # The visible patch may constrain its long direction without
        # constraining a complete rigid pose. Retain that measured direction
        # and the full spread spectrum even when the full-pose gate abstains.
        # No object axis is assumed here; this is only the patch's principal
        # direction, with its sign intentionally meaningless.
        detail["principal_surface_axis_camera"] = directions[:, -1].tolist()
        detail["principal_surface_spreads_m"] = np.sqrt(np.maximum(variances, 0.0)).tolist()
        detail["principal_surface_points"] = len(points)
        split_axes = []
        for subset in (obs["fit"], ~obs["fit"]):
            if np.count_nonzero(subset) < 3:
                break
            partition = points[subset] - points[subset].mean(axis=0)
            _, partition_directions = np.linalg.eigh(partition.T @ partition / len(partition))
            split_axes.append(partition_directions[:, -1])
        if len(split_axes) == 2:
            detail["principal_surface_split_axes_camera"] = [axis.tolist() for axis in split_axes]
            detail["principal_surface_split_gap_rad"] = float(
                np.arccos(np.clip(abs(split_axes[0] @ split_axes[1]), -1, 1))
            )
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
            if self.last_accepted_pose is not None:
                if step is None:
                    self.motion_chain_complete = False
                elif self.motion_chain_complete:
                    self.motion_from_accepted = step @ self.motion_from_accepted
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
                fit_kernels=self.fit_kernels,
                search_pool=self.search_pool,
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
        motion_only_prior = (
            self.motion_from_accepted @ self.last_accepted_pose
            if self.last_accepted_pose is not None and self.motion_chain_complete
            else None
        )
        all_points = current["all_surface_points"]
        predicted = move(self.reference, self.pose)
        information_start = time.perf_counter()
        information = measured_surface_information(
            self.reference,
            self.pose,
            current["obs"]["points"],
            current["obs"]["normals"],
            correspondence_distance_m=self.information_correspondence_m,
        )
        detail["conditional_surface_information_camera"] = {
            "pivot_m": information.pivot.tolist(),
            "twist_basis": information.basis.tolist(),
            "strength": information.strength.tolist(),
            "matched_reference_points": information.matched_points,
            "reference_points": information.reference_points,
            "correspondence_distance_m": self.information_correspondence_m,
        }
        detail["information_ms"] = (time.perf_counter() - information_start) * 1000
        distances = cKDTree(predicted).query(all_points)[0] * 1000
        detail.update(
            surface_median_mm=float(np.median(distances)), surface_p90_mm=float(np.percentile(distances, 90))
        )
        reference_depth_pixels, reference_depth_p90_mm = reference_depth_support(
            predicted, current["obs"]["valid_image"], current["depth"], intrinsics
        )
        detail.update(
            reference_depth_pixels=reference_depth_pixels,
            reference_depth_p90_mm=reference_depth_p90_mm,
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
            "motion_only_prior_camera_from_reference": (
                motion_only_prior.tolist() if motion_only_prior is not None else None
            ),
            "camera_from_object": object_pose.tolist(),
            "observed_m": all_points.tolist(),
            "reference_m": predicted.tolist(),
            "anchors_m": anchors.tolist(),
            "axes_m": axes.tolist(),
            "anchors_px": project(anchors, intrinsics).tolist(),
            "axes_px": project(axes, intrinsics).tolist(),
            "reference_px": project(predicted, intrinsics).tolist(),
        }

    def record_publication(self, accepted: bool) -> None:
        """Anchor the shadow prediction only after the RGB-D publication gate."""
        if accepted:
            self.last_accepted_pose = self.pose.copy()
            self.motion_from_accepted = np.eye(4)
            self.motion_chain_complete = True
