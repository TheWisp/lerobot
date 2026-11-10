"""Object association consumes scene groups; it cannot change group membership.

The pose estimator receives the same PoseHint/PointEvidence contract that other
motion sources can implement. This layer alone knows the selected object's mask
and reference coordinates. Groups themselves never receive either.
"""

from typing import Protocol

import numpy as np

from .experimental_surface_session import PointEvidence, PoseHint
from .motion_groups import fit_point_motion
from .surface_kalman import MeasuredGroupMotion
from .surface_observation import move, spatial_sample


class ObjectMotionSource(Protocol):
    def query(self, group_frame, frame, scene, mask, reference_frame, *, object_usable=None): ...

    def record_observation(self, frame, scene, mask, result, reference, *, object_usable=None): ...


def inside(scene, mask):
    xy = scene.xy.astype(int)
    valid = (xy[:, 0] >= 0) & (xy[:, 1] >= 0) & (xy[:, 0] < mask.shape[1]) & (xy[:, 1] < mask.shape[0])
    keep = np.zeros(len(xy), bool)
    keep[valid] = mask[xy[valid, 1], xy[valid, 0]]
    return keep


class ObjectGroupAssociation:
    """Bind an object's measured points to group coordinates, then query motion.

    Initialization/recovery can establish object coordinates from an accepted
    observation. Subsequent predictions use fixed point anchors, never the last
    fitted object pose. All writes here belong to the object association only.
    """

    def __init__(self, config, *, point_evidence_weight=1.0, min_association_points=12):
        self.config = config
        self.point_evidence_weight = point_evidence_weight
        self.min_association_points = min_association_points
        self.anchors = {}
        self.reference = None
        self.reference_frame = None
        self._group = None
        self._group_from_reference = None

    def query(self, snapshot, frame, scene, mask, reference_frame, *, object_usable=None):
        if snapshot.frame_id != frame.frame_id or snapshot.time_s != frame.source_time_s:
            raise ValueError("group source requires this frame's scene update")
        details = {
            "source": None,
            "group_id": self._group if self._group in snapshot.groups else None,
            "object_inliers": 0,
            "environment_inliers": 0,
            "group_count": len(snapshot.groups),
        }
        alive = {int(i) for g in snapshot.groups.values() for i in g.ids} | set(scene.ids.tolist())
        self.anchors = {i: p for i, p in self.anchors.items() if i in alive}
        if self.reference is None or str(reference_frame) != str(self.reference_frame):
            return None, details
        keep = inside(scene, mask)
        if object_usable is not None:
            keep &= inside(scene, object_usable)
        own_now = set(scene.ids[keep].tolist())
        measured_known_own = own_now & self.anchors.keys()
        candidates = []
        for gid, group in snapshot.groups.items():
            if not group.confirmed or not group.supported:
                continue
            known = [j for j, i in enumerate(group.ids) if int(i) in self.anchors]
            # Both coordinate frames stay fixed while the group ID survives.
            # Re-fitting this relation from whichever few points happen to be
            # visible would move the object even when the group did not move.
            relation = self._group_from_reference if gid == self._group else None
            # A new relation requires measured identities on the object now.
            # Once established, its fixed transform can survive full occlusion.
            visible_known = [j for j in known if group.measured[j] and int(group.ids[j]) in own_now]
            minimum = self.min_association_points
            # Measured membership that has separated contradicts the old
            # relation even when only a small part of the object is visible.
            # The scene group already required sustained 3D disagreement before
            # releasing each point. With none measured, retain the relation
            # for occlusion instead of treating hidden points as new evidence.
            if measured_known_own and 2 * len(visible_known) < len(measured_known_own):
                if gid == self._group:
                    self._group = self._group_from_reference = None
                    details["group_id"] = None
                    details["association_contradiction"] = "visible_object_points_separated"
                continue
            if relation is None and len(visible_known) >= minimum:
                a = np.array([self.anchors[int(group.ids[j])] for j in visible_known])
                b = group.local_xyz[visible_known]
                fitted = fit_point_motion(a, b, self.config)
                if fitted is not None:
                    relation = fitted[0]
            if relation is None:
                continue
            visible_own = sum(int(group.ids[j]) in own_now for j in known)
            candidates.append((visible_own, gid == self._group, len(known), -gid, group, relation))
        if not candidates:
            return None, details
        _, _, _, _, group, relation = max(candidates, key=lambda c: c[:4])
        self._group, self._group_from_reference = group.id, relation.copy()
        # Add fresh object tracks in established group coordinates; a fitted
        # object pose does not move these anchors on subsequent frames.
        for i, local in zip(group.ids, group.local_xyz, strict=True):
            if int(i) in own_now and int(i) not in self.anchors:
                self.anchors[int(i)] = move(local[None], np.linalg.inv(relation))[0]
        camera_from_base = np.linalg.inv(frame.base_from_camera)
        pose = camera_from_base @ group.base_from_group @ relation
        source = f"scene_group:{group.id}"
        reference = self.reference[spatial_sample(self.reference, min(256, len(self.reference)))]
        evidence = PointEvidence(
            reference,
            move(reference, pose),
            source,
            "predicted",
            max(self.config.fit_threshold_m, group.residual_m),
            self.point_evidence_weight,
        )
        motion = None
        if group.from_time_s is not None and len(group.previous_measured) >= self.config.min_inliers:
            motion = MeasuredGroupMotion(
                group.previous_measured, group.current_measured, group.from_time_s, group.time_s, source
            )
        measured_ids = group.ids[group.measured]
        own_count = sum(int(point_id) in own_now for point_id in measured_ids)
        details.update(
            source=source,
            group_id=group.id,
            object_inliers=own_count,
            environment_inliers=len(measured_ids) - own_count,
            hidden_members=int((~group.measured).sum()),
            residual_mm=1000 * group.residual_m,
            camera_from_reference=pose.tolist(),
        )
        if motion is not None:
            details["kalman_motion"] = {
                "source": source,
                "points": len(motion.previous),
                "from_time_s": motion.from_time_s,
            }
        return PoseHint(
            pose, source, frame.source_time_s, group.time_s, reference_frame, (evidence,), motion
        ), details

    def record_observation(self, frame, scene, mask, result, reference, *, object_usable=None):
        if result["status"] not in {"initialized", "observed_candidate"} or reference is None:
            return
        version = result["diagnostics"]["reference_frame"]
        same_reference = self.reference is not None and str(version) == str(self.reference_frame)
        keep = inside(scene, mask)
        if object_usable is not None:
            keep &= inside(scene, object_usable)
        if keep.sum() < 3 or result.get("camera_from_reference") is None:
            return
        base_from_reference = frame.base_from_camera @ np.asarray(result["camera_from_reference"])
        xyz = move(scene.base_xyz[keep], np.linalg.inv(base_from_reference))
        measured = dict(zip(scene.ids[keep].tolist(), xyz, strict=True))
        if same_reference:
            # A direct accepted observation may identify NEW object points.
            # Preserve old measured identities: no fitted-pose feedback moves
            # their reference positions or any scene-group state.
            for point_id, point in measured.items():
                self.anchors.setdefault(point_id, point)
            return
        self.anchors = measured
        self.reference = np.asarray(reference).copy()
        self.reference_frame = version
        self._group = self._group_from_reference = None
