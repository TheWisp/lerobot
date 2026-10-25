"""One validated vision step, shared by offline, process and interactive consumers."""

from dataclasses import dataclass, field

import numpy as np

from .perception import ObjectReference, PoseBackend, RGBDFrame
from .trajectory import PoseEstimate


@dataclass(frozen=True)
class PerceptionResult:
    estimates: tuple[PoseEstimate, ...]
    failures: dict[str, str]
    frame_id: int
    source_time_s: float
    calibration_version: str
    masks: dict[str, np.ndarray]
    details: dict[str, dict] = field(default_factory=dict)


class PerceptionPipeline:
    """Own estimator state; clocks, queues and camera I/O belong to its caller."""

    def __init__(self, backend: PoseBackend, references: tuple[ObjectReference, ...], *, require_masks=False):
        self.backend = backend
        self.versions = {r.role: r.version for r in references}
        if len(self.versions) != len(references) or not references:
            raise ValueError("perception needs unique object references")
        if require_masks and not hasattr(backend, "last_masks"):
            raise TypeError("causal tracking requires a pose backend that exposes aligned instance masks")
        self.require_masks = require_masks
        backend.initialize(references)

    def add_reference(self, reference: ObjectReference) -> None:
        """Add a role only when the backend supports in-place registration."""
        if reference.role in self.versions:
            raise ValueError(f"{reference.role}: object role already exists")
        add = getattr(self.backend, "add_reference", None)
        if add is None:
            raise TypeError("this pose backend cannot add an object while tracking")
        add(reference)
        self.versions[reference.role] = reference.version

    def process(self, frame: RGBDFrame) -> PerceptionResult:
        estimates = self.backend.update(frame)
        roles = {estimate.role for estimate in estimates}
        if roles != set(self.versions) or len(estimates) != len(roles):
            raise ValueError("pose results differ from reference roles")
        for estimate in estimates:
            if estimate.reference_version != self.versions[estimate.role]:
                raise ValueError("pose result changed the registered object frame")
            if estimate.source_time_s != frame.source_time_s:
                raise ValueError("pose result belongs to a different source image")
        masks = getattr(self.backend, "last_masks", {})
        for role, mask in masks.items():
            if (
                role not in roles
                or np.asarray(mask).shape != frame.depth.shape
                or np.asarray(mask).dtype != bool
            ):
                raise ValueError(f"{role}: causal mask does not align with RGB-D")
        return PerceptionResult(
            estimates,
            dict(getattr(self.backend, "last_failure", {})),
            frame.frame_id,
            frame.source_time_s,
            frame.calibration_version,
            {role: mask.copy() for role, mask in masks.items()},
            dict(getattr(self.backend, "last_details", {})),
        )
