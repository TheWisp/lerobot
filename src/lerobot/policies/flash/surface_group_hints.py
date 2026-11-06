"""Small adapter from the existing scene point tracker to surface-session hints.

Object points are checked against the current mask. Surrounding points can
predict a hidden object after their motion agrees across direct observations,
including when both remain stationary. Proximity alone is not enough.
All poses here are hypotheses: optical flow and the association can both drift.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.policies.flash.causal_points import CausalRigidPointTracker
from lerobot.policies.flash.experimental_surface_session import PointEvidence, PoseHint
from lerobot.policies.flash.geometry import transform
from lerobot.policies.flash.pose_disagreement import PoseComparisonVerdict, compare_pose_proposals
from lerobot.policies.flash.scene_groups import EnvironmentRelationship, SceneTracks
from lerobot.policies.flash.surface_grounding import ShapeConfig, WholeSurfaceFit
from lerobot.policies.flash.surface_kalman import MeasuredGroupMotion
from lerobot.policies.flash.surface_observation import move, spatial_sample
from lerobot.policies.flash.surface_uncertainty import PoseUncertaintyConfig, check_surface_contradiction


@dataclass(frozen=True)
class GroupConfig:
    # Same initial settings as the sealed scene tracker. These are experiment
    # parameters, not calibrated confidence or rules for a particular object.
    context_radius_m: float = 0.25
    min_inliers: int = 12
    fit_threshold_m: float = 0.005
    object_max_points: int = 120
    object_max_frame_gap_s: float = 0.1
    object_fb_threshold_px: float = 1.0
    object_image_scale: float = 1.0
    depth_support_radius_px: int = 1
    # Experimental duration of continuous observed agreement before admission.
    # Source timestamps, not processing time; separation has no waiting period.
    support_confirmation_s: float = 1.0
    # Per independent source, normalized by its point count.
    point_evidence_weight: float = 1.0

    def __post_init__(self):
        if type(self.min_inliers) is not int or self.min_inliers < 3:
            raise ValueError("groups need at least three points")
        if not np.isfinite(self.support_confirmation_s) or self.support_confirmation_s < 0:
            raise ValueError("support confirmation duration must be finite and nonnegative")
        if any(not np.isfinite(v) or v <= 0 for v in (self.context_radius_m, self.fit_threshold_m)):
            raise ValueError("group distances must be positive and finite")
        if not np.isfinite(self.point_evidence_weight) or self.point_evidence_weight < 0:
            raise ValueError("point evidence weight must be finite and nonnegative")
        # Validate with the actual consumer, including point count and image scale.
        CausalRigidPointTracker(**self.point_tracking_options())

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        # Historical experiments required displacement before association.
        # Continuous relative-position evidence replaces that requirement.
        values.pop("min_motion_m", None)
        return cls(**values)

    def point_tracking_options(self):
        return {
            "max_points": self.object_max_points,
            "min_inliers": self.min_inliers,
            "max_frame_gap_s": self.object_max_frame_gap_s,
            "fb_threshold_px": self.object_fb_threshold_px,
            "fit_threshold_m": self.fit_threshold_m,
            "image_scale": self.object_image_scale,
            "depth_support_radius_px": self.depth_support_radius_px,
        }


def select(scene, keep):
    return SceneTracks(
        scene.frame_id, scene.source_time_s, scene.ids[keep], scene.xy[keep], scene.base_xyz[keep]
    )


def inside(scene, mask):
    xy = scene.xy.astype(int)
    return mask[xy[:, 1], xy[:, 0]]


class GroupHints:
    """Independent measured point motion; a fitted pose only initializes a reference.

    During an uninterrupted track, final surface poses cannot rewrite point
    coordinates, group associations or source motion. Reinitialization after
    all independent sources fail is explicit and starts a new point generation.
    """

    def __init__(self, config, *, shared_image_points=False, uncertainty_config=None):
        self.uncertainty_config = uncertainty_config or PoseUncertaintyConfig()
        self.shared_image_points = shared_image_points
        self.config = config
        self.object_points = None
        self.environment = None
        self.environment_prediction = None
        self.point_tracker = None
        self.point_anchor = None
        self.point_motion = None
        self.point_generation = 0
        self.object_prediction = None
        self.object_support = None
        self.reference = None
        self.support_evidence = {}
        self.separated_ids = set()
        self.previous_scene = None

    def _predict(self, relation, scene):
        if relation is None:
            return None, np.empty(0, dtype=int)
        _, left, right = np.intersect1d(relation.ids, scene.ids, return_indices=True)
        if len(left) < self.config.min_inliers:
            return None, np.empty(0, dtype=int)
        # Reuse the old consensus fitter, without its separate translation-only
        # shortcut: a group rotating around its centre still measures rotation.
        fit = EnvironmentRelationship._fit_motion(
            relation.anchor_xyz[left],
            scene.base_xyz[right],
            self.config.fit_threshold_m,
            self.config.min_inliers,
        )
        if fit is None:
            return None, np.empty(0, dtype=int)
        delta, inliers = fit
        return delta @ relation.anchor_pose, right[inliers]

    def predict(
        self,
        frame,
        scene,
        masks,
        role,
        reference_frame,
        *,
        object_usable=None,
        observation=None,
        shape_config=None,
        reference_views=(),
        surface_reference=None,
        prepared_geometry=None,
    ):
        geometry = None

        def measured_geometry():
            nonlocal geometry
            if geometry is None:
                shared = prepared_geometry
                geometry = WholeSurfaceFit(
                    self.reference if surface_reference is None else surface_reference,
                    observation,
                    frame.intrinsics,
                    shape_config or ShapeConfig(),
                    reference_views=shared.reference_views if shared is not None else reference_views,
                    prepared_reference=shared.prepared_reference if shared is not None else None,
                    prepared_target=shared.prepared_target if shared is not None else None,
                    prepared_reverse_fits=shared.reverse_fits if shared is not None else None,
                    fit_kernels=shared.fit_kernels if shared is not None else None,
                    search_pool=shared.search_pool if shared is not None else None,
                )
            return geometry

        previous_scene, previous_object_support = self.previous_scene, self.object_support
        self.previous_scene = scene
        self.environment_prediction = None
        self.object_prediction = None
        self.point_motion = None
        self.separated_ids = set()
        details = {
            "source": None,
            "object_inliers": 0,
            "environment_inliers": 0,
            "ids": [],
            "pixels": [],
            "camera_xyz": [],
            "display_ids": [],
            "display_pixels": [],
            "display_camera_xyz": [],
            "object_ids": [],
            "environment_ids": [],
            "pending_environment_points": len(self.support_evidence)
            - (len(self.environment.ids) if self.environment is not None else 0),
            "environment_separated": False,
        }
        if reference_frame is None:
            return None, details
        object_keep = inside(scene, masks[role])
        if object_usable is not None:
            object_keep &= inside(scene, object_usable)
        object_scene = select(scene, object_keep)
        object_pose, object_indices = self._predict(self.object_points, object_scene)
        object_support = select(object_scene, object_indices)
        camera_from_base = np.linalg.inv(frame.base_from_camera)
        evidence = []
        if object_pose is not None:
            _, left, right = np.intersect1d(self.object_points.ids, object_support.ids, return_indices=True)
            evidence.append(
                PointEvidence(
                    move(self.object_points.anchor_xyz[left], np.linalg.inv(self.object_points.anchor_pose)),
                    move(object_support.base_xyz[right], camera_from_base),
                    "object_point_group",
                    "measured",
                    self.config.fit_threshold_m,
                    self.config.point_evidence_weight,
                )
            )
        if self.point_tracker is not None:
            try:
                motion = self.point_tracker.update(frame, masks[role])
            except ValueError:
                # A camera gap invalidates these identities; a later direct
                # observation can reseed them. Surrounding support is separate.
                self.point_tracker = None
            else:
                if motion.base_motion_from_seed is not None:
                    self.point_motion = motion.base_motion_from_seed
                    object_pose = self.point_motion @ self.point_anchor.anchor_pose
                    ids, pixels, xyz = self.point_tracker.inlier_samples()
                    # Negative IDs distinguish dedicated object seeds from the
                    # shared scene pool and break trails when a seed changes.
                    ids = -(self.point_generation * self.config.object_max_points + ids + 1)
                    object_support = SceneTracks(frame.frame_id, frame.source_time_s, ids, pixels, xyz)
                    object_indices = np.arange(len(ids))
                    original, measured = self.point_tracker.inlier_point_pairs()
                    evidence = [
                        PointEvidence(
                            move(original, np.linalg.inv(self.point_anchor.anchor_pose)),
                            move(measured, camera_from_base),
                            "object_point_group",
                            "measured",
                            self.config.fit_threshold_m,
                            self.config.point_evidence_weight,
                        )
                    ]
        self.object_prediction = object_pose
        self.object_support = object_support
        details["object_inliers"] = len(object_indices)
        # Other tracked objects can support this one too: ownership does not
        # prevent two bodies from sharing motion. Only exclude this object.
        outside = ~inside(scene, masks[role])
        environment_scene = select(scene, outside)
        environment_pose, environment_indices = self._predict(self.environment, environment_scene)
        if environment_pose is not None and observation is not None and self.uncertainty_config.enabled:
            geometry = measured_geometry()
            check = check_surface_contradiction(
                geometry, camera_from_base @ environment_pose, self.uncertainty_config
            )
            details["environment_observation_check"] = check
            if not check["accepted"]:
                # Current measured pixels, not a fitted object's pose, disprove
                # this relationship even when object image tracks disappeared.
                self.separated_ids = set(self.environment.ids.tolist())
                self.environment = None
                self.support_evidence = {}
                environment_pose, environment_indices = None, np.empty(0, dtype=int)
                details["environment_separated"] = True
        if object_pose is not None and environment_pose is not None:
            # Test the measurements themselves. Extrapolating the rotation of
            # a tiny visible fragment to hidden object points (or to distant
            # surrounding points) magnifies an unconstrained rotation into a
            # false separation. No final surface pose participates here.
            measured = [p for p in evidence if p.kind == "measured"]
            gap = float(
                np.median(
                    np.concatenate(
                        [
                            np.linalg.norm(
                                move(p.reference, camera_from_base @ environment_pose) - p.current, axis=1
                            )
                            for p in measured
                        ]
                    )
                )
            )
            details["object_environment_difference_mm"] = 1000 * gap
            if gap > self.config.fit_threshold_m:
                # Image matches can slide across a visible surface while still
                # passing mask/depth membership. Their disagreement alone is
                # not evidence that the object's attachment actually broke.
                # Compare BOTH independently predicted locations to today's
                # measured surface before discarding established support.
                object_supported = True
                if observation is not None:
                    geometry = measured_geometry()
                    object_cost = geometry.score(camera_from_base @ object_pose)
                    environment_cost = geometry.score(camera_from_base @ environment_pose)
                    details["separation_surface_costs"] = {
                        "object_matches": object_cost,
                        "surrounding_group": environment_cost,
                    }
                    object_supported = object_cost <= environment_cost
                    if self.uncertainty_config.enabled:
                        # A lower scalar cost alone does not establish separation
                        # when the visible geometry cannot distinguish the poses.
                        object_supported = not check_surface_contradiction(
                            geometry, camera_from_base @ environment_pose, self.uncertainty_config
                        )["accepted"]
                if object_supported:
                    self.separated_ids = set(self.environment.ids.tolist())
                    self.environment = None
                    self.support_evidence = {}
                    environment_pose, environment_indices = None, np.empty(0, dtype=int)
                    details["environment_separated"] = True
                else:
                    # The rejected image proposal must not still pull the fit
                    # through a second, "measured points" residual block.
                    object_pose, object_indices = None, np.empty(0, dtype=int)
                    object_support = select(object_scene, object_indices)
                    evidence = [replace(p, weight=0.0) for p in evidence]
                    self.object_prediction = None
                    self.object_support = object_support
                    details["object_inliers"] = 0
                    details["object_matches_rejected_by_observation"] = True
        details["environment_inliers"] = len(environment_indices)
        self.environment_prediction = environment_pose
        if environment_pose is not None:
            reference = self.reference[spatial_sample(self.reference, self.config.object_max_points)]
            evidence.append(
                PointEvidence(
                    reference,
                    move(reference, camera_from_base @ environment_pose),
                    "environment_point_group",
                    "predicted",
                    self.config.fit_threshold_m,
                    self.config.point_evidence_weight,
                )
            )
        environment_support = select(environment_scene, environment_indices)
        supports = [object_support, environment_support]
        details.update(
            object_ids=object_support.ids.tolist(),
            environment_ids=environment_support.ids.tolist(),
            display_ids=np.concatenate([s.ids for s in supports]).tolist(),
            display_pixels=np.concatenate([s.xy for s in supports]).tolist(),
            display_camera_xyz=move(
                np.concatenate([s.base_xyz for s in supports]), np.linalg.inv(frame.base_from_camera)
            ).tolist(),
        )
        if object_pose is not None:
            pose, source = object_pose, "object_point_group"
            relation = self.point_anchor if self.point_motion is not None else self.object_points
            support = object_support
        elif environment_pose is not None:
            pose, relation, source = environment_pose, self.environment, "environment_point_group"
            support = environment_support
        else:
            return None, details
        details.update(
            source=source,
            anchor_time_s=relation.anchor_time_s,
            ids=support.ids.tolist(),
            pixels=support.xy.tolist(),
            camera_xyz=move(support.base_xyz, np.linalg.inv(frame.base_from_camera)).tolist(),
        )
        # Recorded camera matrices can be rounded, and composing two of them
        # carries that tiny non-orthogonality into the prediction. Validate
        # first, then project numerical roundoff back onto a proper rotation.
        # This rejects actual scale/shear rather than silently repairing it.
        camera_pose = transform(np.linalg.inv(frame.base_from_camera) @ pose)
        camera_pose[:3, :3] = Rotation.from_matrix(camera_pose[:3, :3]).as_matrix()
        details["camera_from_reference"] = camera_pose.tolist()
        motion = None
        if self.uncertainty_config.enabled and self.uncertainty_config.mode == "kalman":
            # A surrounding group's measured step takes precedence over object
            # image matches when its relationship survived observation checks.
            motion_support = environment_support if environment_pose is not None else object_support
            motion_source = (
                "environment_point_group" if environment_pose is not None else "object_point_group"
            )
            old = (
                previous_object_support
                if len(motion_support.ids) and np.all(motion_support.ids < 0)
                else previous_scene
            )
            if old is not None and len(motion_support.ids):
                _, left, right = np.intersect1d(old.ids, motion_support.ids, return_indices=True)
                if len(left) >= self.config.min_inliers:
                    fitted = EnvironmentRelationship._fit_motion(
                        old.base_xyz[left],
                        motion_support.base_xyz[right],
                        self.config.fit_threshold_m,
                        self.config.min_inliers,
                    )
                    if fitted is not None:
                        _, inliers = fitted
                        motion = MeasuredGroupMotion(
                            old.base_xyz[left[inliers]],
                            motion_support.base_xyz[right[inliers]],
                            old.source_time_s,
                            frame.source_time_s,
                            motion_source,
                        )
                        details["kalman_motion"] = {
                            "source": motion_source,
                            "points": len(inliers),
                            "from_time_s": old.source_time_s,
                        }
        return PoseHint(
            camera_pose,
            source,
            frame.source_time_s,
            relation.anchor_time_s,
            reference_frame,
            tuple(evidence),
            motion,
        ), details

    def compare_candidate(
        self, frame, mask, hint, points, candidate, *, observation=None, shape_config=None, reference_views=()
    ):
        """Compare hypotheses using the fitter's geometry objective when supplied.

        The archived Pareto comparison remains available for the weighted-sum
        ablation. A custom session callback still has its normal veto semantics.
        """
        if observation is not None:
            geometry = WholeSurfaceFit(
                points,
                observation,
                frame.intrinsics,
                shape_config or ShapeConfig(),
                reference_views=reference_views,
            )
            check = check_surface_contradiction(geometry, candidate, self.uncertainty_config)
            return check["accepted"], {"observation_check": check}
        comparison = self._compare(
            frame,
            mask,
            points,
            frame.base_from_camera @ candidate,
            frame.base_from_camera @ hint.camera_from_reference,
        )

        def support(value):
            return {
                "mask_fraction": value.projected_point_mask_fraction,
                "depth_points": value.masked_depth_points,
                "tested_depth_points": value.tested_masked_depth_points,
                "depth_p90_mm": 1000 * value.depth_residual_p90_m
                if np.isfinite(value.depth_residual_p90_m)
                else None,
            }

        return comparison.verdict in (PoseComparisonVerdict.AGREE, PoseComparisonVerdict.CANDIDATE_FAVORED), {
            "verdict": comparison.verdict.value,
            "candidate": support(comparison.candidate),
            "group": support(comparison.prior),
            "translation_difference_mm": 1000 * comparison.object_center_difference_m,
            "rotation_difference_deg": float(np.rad2deg(comparison.rotation_difference_rad)),
        }

    def _compare(self, frame, mask, points, candidate, prior):
        center = points.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((points - center) ** 2, axis=1))))
        return compare_pose_proposals(
            frame,
            mask,
            points,
            candidate,
            prior,
            max_agreement_offset_m=self.config.fit_threshold_m,
            max_agreement_angle_rad=float(
                np.arctan2(self.config.fit_threshold_m, max(radius, np.finfo(float).eps))
            ),
            min_masked_depth_points=self.config.min_inliers,
            depth_occlusion_tolerance_m=self.config.fit_threshold_m,
        )

    def _retain_established(self):
        established = set(self.environment.ids.tolist()) if self.environment is not None else set()
        self.support_evidence = {
            point_id: evidence
            for point_id, evidence in self.support_evidence.items()
            if point_id in established
        }

    def observe(self, frame, scene, masks, role, result, reference, *, object_usable=None):
        initializing = self.reference is None
        direct_fit = result["status"] in {"initialized", "observed_candidate"}
        if not direct_fit and self.object_prediction is None:
            self._retain_established()
            return
        self.reference = reference
        version = str(result["diagnostics"]["reference_frame"])
        # Prefer an independently tracked reference. The final fit is used only
        # to establish coordinates at initialization or full track loss.
        source_pose = self.object_prediction
        if source_pose is None:
            source_pose = self.environment_prediction
        if source_pose is None:
            source_pose = frame.base_from_camera @ np.asarray(result["camera_from_reference"])
        if not self.shared_image_points and (self.point_tracker is None or self.point_motion is None):
            self.point_tracker = CausalRigidPointTracker(**self.config.point_tracking_options())
            self.point_generation += 1
            if self.point_tracker.seed(frame, masks[role]) < self.config.min_inliers:
                self.point_tracker = None
            self.point_anchor = EnvironmentRelationship(
                np.empty(0, dtype=int), np.empty((0, 3)), source_pose.copy(), frame.source_time_s, version
            )
            result["diagnostics"]["point_reference_reset"] = {
                "generation": self.point_generation,
                "source": "object_points"
                if self.object_prediction is not None
                else (
                    "surrounding_points" if self.environment_prediction is not None else "direct_surface_fit"
                ),
            }
        object_keep = inside(scene, masks[role])
        if object_usable is not None:
            object_keep &= inside(scene, object_usable)
        current = select(scene, object_keep)
        # Renew scene identities in the independently tracked coordinates, never
        # in coordinates corrected by the downstream surface optimizer.
        self.object_points = (
            EnvironmentRelationship(
                current.ids.copy(), current.base_xyz.copy(), source_pose.copy(), frame.source_time_s, version
            )
            if len(current.ids) >= self.config.min_inliers
            else None
        )
        if not initializing and self.object_prediction is None:
            # A surrounding prediction cannot confirm its own attachment.
            # Wait for measured object tracks after a new reference is seeded.
            self._retain_established()
            return
        # A support point's coordinates relative to the object should stay
        # constant. Keep its first measured coordinates, not a moving average:
        # otherwise small frame-to-frame errors could conceal slow separation.
        # Once surrounding motion agrees with measured object points, use that
        # established reference for membership. Otherwise a noisy fragment's
        # fitted rotation can discard stable distant support despite passing the
        # measured separation check above. Object tracks remain independently anchored.
        association_pose = (
            self.environment_prediction if self.environment_prediction is not None else source_pose
        )
        center = move(reference.mean(axis=0, keepdims=True), association_pose)[0]
        allowed = ~inside(scene, masks[role]) & (
            np.linalg.norm(scene.base_xyz - center, axis=1) <= self.config.context_radius_m
        )
        nearby = select(scene, allowed)
        local = move(nearby.base_xyz, np.linalg.inv(association_pose))
        evidence, mature = {}, []
        for index, (point_id, xyz) in enumerate(zip(nearby.ids.tolist(), local, strict=True)):
            if point_id in self.separated_ids:
                continue
            previous = self.support_evidence.get(point_id)
            if previous is None:
                evidence[point_id] = (xyz.copy(), frame.source_time_s)
                continue  # One observation only establishes a candidate.
            anchor, since = previous
            if np.linalg.norm(xyz - anchor) > self.config.fit_threshold_m:
                continue  # Immediate removal; a later observation can start again.
            evidence[point_id] = previous
            if frame.source_time_s - since >= self.config.support_confirmation_s:
                mature.append(index)
        self.support_evidence = evidence
        self.environment = (
            EnvironmentRelationship(
                nearby.ids[mature].copy(),
                nearby.base_xyz[mature].copy(),
                (
                    self.environment_prediction if self.environment_prediction is not None else source_pose
                ).copy(),
                frame.source_time_s,
                version,
            )
            if len(mature) >= self.config.min_inliers
            else None
        )
