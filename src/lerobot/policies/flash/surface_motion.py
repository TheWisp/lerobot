"""Carry one rigid pose with measured motion; qualify visual corrections by geometry.

This is an experimental RGB-D fusion component, not the default live backend.
It does not infer an attachment or a support relationship. Callers must supply
those relationships and synchronized motion. Missing motion is not a stationary
measurement, and a segmentation mask alone does not observe six pose axes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .causal_points import CausalRigidPointTracker, RigidPointMotion
from .geometry import transform
from .perception import RGBDFrame
from .registration import transform_points


@dataclass(frozen=True)
class SurfaceMotionConfig:
    # Metres: inherited depth-fit error budget, and the acceptable uncertainty
    # of a correction expressed as RMS displacement of the reference surface.
    depth_error_m: float = 0.005
    correction_uncertainty_m: float = 0.01
    # Four 3 mm surface voxels; enough neighbourhood to estimate a local plane.
    normal_radius_m: float = 0.012
    correspondence_distance_m: float = 0.03

    def __post_init__(self):
        if not all(np.isfinite(v) and v > 0 for v in vars(self).values()):
            raise ValueError("surface-motion scales must be finite and positive")


@dataclass(frozen=True)
class SurfaceSupport:
    # Columns are twist directions [rotation radians, centre translation metres].
    # Each column moves the *whole reference* by one metre RMS to first order.
    basis: np.ndarray
    sensitivity: np.ndarray
    matched_points: int
    reference_points: int


def displacement_jacobian(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """d(R(p-c)+c+t)/d[rotation, translation], shape N x 3 x 6."""
    relative = points - center
    jac = np.zeros((len(points), 3, 6))
    jac[:, :, 3:] = np.eye(3)
    jac[:, 0, 1], jac[:, 0, 2] = relative[:, 2], -relative[:, 1]
    jac[:, 1, 0], jac[:, 1, 2] = -relative[:, 2], relative[:, 0]
    jac[:, 2, 0], jac[:, 2, 1] = relative[:, 1], -relative[:, 0]
    return jac


def pose_with_delta(pose: np.ndarray, center: np.ndarray, delta: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(delta[:3]).as_matrix()
    result = pose.copy()
    result[:3, :3] = rotation @ pose[:3, :3]
    result[:3, 3] = rotation @ (pose[:3, 3] - center) + center + delta[3:]
    return result


def surface_support(
    reference: np.ndarray,
    pose: np.ndarray,
    observed: np.ndarray,
    config: SurfaceMotionConfig,
) -> SurfaceSupport:
    """Find motion directions constrained by the visible depth surface.

    Point-to-plane sensitivity distinguishes sliding along a smooth surface
    from moving through it. Normalize by whole-body displacement, not arbitrary
    world-frame origin, radians versus metres, or the number of depth pixels.
    Missing surface contributes zero information rather than inflating the
    confidence of the remaining tiny patch. This is a local geometric test,
    not a calibrated covariance or proof of object identity.
    """
    empty = SurfaceSupport(np.empty((6, 0)), np.zeros(6), 0, len(reference))
    if len(reference) < 3 or len(observed) < 3:
        return empty
    predicted = transform_points(pose, reference)
    jac = displacement_jacobian(predicted, predicted.mean(0))
    metric = np.einsum("nij,nik->jk", jac, jac) / len(reference)
    values, vectors = np.linalg.eigh(metric)
    if values[0] <= np.finfo(float).eps * values[-1]:
        return empty
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    tree = cKDTree(observed)
    distance, indices = tree.query(predicted)
    normals = np.zeros_like(observed)
    valid = np.zeros(len(observed), bool)
    for index in np.unique(indices[distance <= config.correspondence_distance_m]):
        patch = observed[tree.query_ball_point(observed[index], config.normal_radius_m)]
        if len(patch) < 3:
            continue
        _, singular, axes = np.linalg.svd(patch - patch.mean(0), full_matrices=False)
        if singular[1] <= np.finfo(float).eps * max(singular[0], 1):
            continue
        normals[index] = axes[-1]
        valid[index] = True
    supported = valid[indices] & (distance <= config.correspondence_distance_m)
    covered = np.zeros(len(reference), bool)
    reverse_distance, reverse_indices = cKDTree(predicted).query(observed)
    covered[reverse_indices[reverse_distance <= config.correspondence_distance_m]] = True
    supported &= covered
    if not supported.any():
        return empty
    rows = np.einsum("ni,nij->nj", normals[indices[supported]], jac[supported]) @ whitening
    # Points sharing a normal neighbourhood are correlated; increasing raster
    # density must not multiply evidence. Count spatial patches, not pixels.
    patches = np.unique(np.floor((predicted - predicted.min(0)) / config.normal_radius_m), axis=0)
    information = rows.T @ rows * len(patches) / len(reference)
    strength, directions = np.linalg.eigh(information)
    strength = np.maximum(strength, 0)
    # A displacement of the allowed uncertainty must change measured depth
    # by at least the configured depth error. No object-specific symmetry rule.
    keep = strength * config.correction_uncertainty_m**2 > config.depth_error_m**2
    return SurfaceSupport(whitening @ directions[:, keep], strength, int(supported.sum()), len(reference))


class RigidMotionState:
    """The sole pose owner for an object in a causal fusion pass.

    A bridge is a measured world-from-carrier transform and a relationship ID.
    Its delta moves the accepted state, never an unrelated visual candidate.
    A one-frame missing bridge does not erase the relationship, so the next
    measured transform spans the gap correctly. The gap itself returns no pose.
    Switching/releasing a relationship must seed a new carrier coordinate frame.
    """

    def __init__(self):
        self.pose: np.ndarray | None = None
        self.time_s: float | None = None
        self.bridge_id: str | None = None
        self.carrier: np.ndarray | None = None
        self.last_measurement_s: float | None = None

    def predict(
        self,
        time_s: float,
        bridge_id: str | None,
        carrier: np.ndarray | None,
        *,
        anchor_pose: np.ndarray | None = None,
    ) -> np.ndarray | None:
        if self.time_s is not None and time_s <= self.time_s:
            raise ValueError("fusion frames must have increasing source timestamps")
        if not np.isfinite(time_s):
            raise ValueError("fusion timestamps must be finite")
        carrier = transform(carrier) if carrier is not None else None
        anchor_pose = transform(anchor_pose) if anchor_pose is not None else None
        self.time_s = time_s
        changed = bridge_id != self.bridge_id
        if changed:
            self.bridge_id, self.carrier = bridge_id, None
        prior = None
        if anchor_pose is not None and carrier is not None:
            # A group backend may re-anchor its coordinates on every direct
            # image. Apply its measured delta to *our* state at that timestamp,
            # not to the backend's potentially conflicting visual anchor.
            prior = carrier @ anchor_pose
        elif self.pose is not None and carrier is not None:
            if self.carrier is not None:
                prior = carrier @ np.linalg.inv(self.carrier) @ self.pose
            elif changed:
                # The relationship begins at this frame, not at an unknown
                # earlier time. Its initial pose still needs visual validation.
                prior = self.pose.copy()
            self.carrier = carrier.copy()
        if prior is not None:
            self.pose = prior.copy()
        return prior

    def accept(self, pose: np.ndarray, *, measured: bool, carrier: np.ndarray | None = None) -> None:
        self.pose = transform(pose)
        # A new image without a synchronized carrier cannot reuse an old
        # carrier timestamp: doing so would apply part of its motion twice.
        self.carrier = carrier.copy() if carrier is not None else None
        if measured:
            self.last_measurement_s = self.time_s


class AcceptedPosePointBridge:
    """Carry an accepted pose using visible points, including across release.

    A partial view can measure relative motion without independently solving
    the object's complete pose. Seeding from a prediction preserves that
    uncertainty; it is never a new absolute observation. The caller must feed
    consecutive images even while another motion source owns the output.
    """

    def __init__(self, **tracker_kwargs):
        self.tracker_kwargs = tracker_kwargs
        self.tracker: CausalRigidPointTracker | None = None
        self.anchor_pose: np.ndarray | None = None
        self.anchor_time_s: float | None = None

    def seed(self, frame: RGBDFrame, mask: np.ndarray, pose: np.ndarray) -> bool:
        pose = transform(pose)
        tracker = CausalRigidPointTracker(**self.tracker_kwargs)
        if tracker.seed(frame, mask) < tracker.min_inliers:
            return False
        self.tracker, self.anchor_pose = tracker, pose.copy()
        self.anchor_time_s = frame.source_time_s
        return True

    def advance(self, frame: RGBDFrame, mask: np.ndarray) -> RigidPointMotion | None:
        if self.tracker is None:
            return None
        try:
            return self.tracker.update(frame, mask=mask)
        except ValueError as exc:
            if "lost a camera-frame interval" not in str(exc):
                raise
            self.tracker, self.anchor_pose, self.anchor_time_s = None, None, None
            return None


class PendingMotionCorrection:
    """Validate a visual correction on a later image without stopping prediction.

    The candidate remains separate from the accepted state. Only measured motion
    transports it; losing/switching the carrier invalidates the pending test.
    Scores must share units and geometry within a frame. The required improvement
    is supplied by the caller's observation error model, not a pose-jump limit.
    """

    def __init__(self):
        self.pending: tuple[str, np.ndarray, np.ndarray] | None = None

    def check(self, bridge_id, carrier, prior, score, minimum_improvement):
        pending, self.pending = self.pending, None
        if pending is None or carrier is None or pending[0] != bridge_id:
            return None
        _, previous_carrier, candidate = pending
        candidate = carrier @ np.linalg.inv(previous_carrier) @ candidate
        improvement = score(prior) - score(candidate)
        return candidate if np.isfinite(improvement) and improvement > minimum_improvement else None

    def propose(self, bridge_id, carrier, prior, candidate, score, minimum_improvement):
        self.pending = None
        if carrier is None or bridge_id is None:
            return
        improvement = score(prior) - score(candidate)
        if np.isfinite(improvement) and improvement > minimum_improvement:
            self.pending = (bridge_id, carrier.copy(), candidate.copy())
