"""Run recorded RGB-D through the live pose proposals and scene estimator."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np

from .causal_service import CausalTrackingConfig, build_causal_pose_service
from .perception import LatestPoseStore, ObjectReference, PoseBackend, RGBDFrame
from .perception_step import PerceptionPipeline, PerceptionResult, PerceptionRunner
from .symmetry import AxialSymmetry
from .trajectory import PoseEstimate


def extract_pose_trace(
    frames: Iterable[RGBDFrame],
    backend: PoseBackend,
    references: tuple[ObjectReference, ...],
    *,
    causal_tracking: CausalTrackingConfig | None,
    object_points_by_role: dict[str, np.ndarray] | None = None,
    axial_symmetries: tuple[AxialSymmetry, ...] = (),
    attached_role_at_frame: Callable[[int], str | None] | None = None,
    on_vision_result: Callable[[PerceptionResult], None] | None = None,
) -> list[dict[str, PoseEstimate]]:
    """Extract one causal pose row per recorded frame, without a robot clock.

    The direct backend and causal estimator are the same components as in a
    live attempt. Offline processing receives every recorded frame and may
    finish its direct proposal before advancing to the next frame. It does not
    claim to reproduce live scheduling or the gripper attachment updater.
    """
    pipeline = PerceptionPipeline(backend, references, require_masks=causal_tracking is not None)
    service = None
    if causal_tracking is not None:
        store = LatestPoseStore()
        service = build_causal_pose_service(
            store,
            causal_tracking,
            object_centers_by_role={
                role: np.asarray(points, dtype=float).mean(axis=0)
                for role, points in (object_points_by_role or {}).items()
            },
            object_points_by_role=object_points_by_role,
            axial_symmetries=axial_symmetries,
        )
    trace = []
    source = iter(frames)
    with PerceptionRunner(pipeline) as runner:
        while (item := runner.process_next(lambda: next(source, None))) is not None:
            frame, result = item
            if on_vision_result is not None:
                on_vision_result(result)
            proposals = result.estimates
            by_role = {estimate.role: estimate for estimate in proposals}
            if service is not None:
                masks = result.masks
                fused = service.process_offline(
                    frame,
                    proposals,
                    {role: mask.copy() for role, mask in masks.items()},
                    attached_role=(
                        attached_role_at_frame(frame.frame_id) if attached_role_at_frame else None
                    ),
                )
                by_role = {
                    role: fused[role]
                    if role in fused and fused[role].source_time_s == frame.source_time_s
                    else proposal
                    for role, proposal in by_role.items()
                }
            trace.append(by_role)
    return trace
