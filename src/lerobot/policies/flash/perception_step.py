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
        return self.result(frame, estimates)

    def result(self, frame, estimates) -> PerceptionResult:
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


class PerceptionRunner:
    """One prepared frame ahead of CPU fitting, with no unbounded work queue.

    next_frame runs on the preparation thread when overlap is enabled. It must
    return an owned RGB-D snapshot (or None), in increasing source order. Camera
    callers choose the latest snapshot there; offline callers supply every frame.
    Drain before changing selections/configuration; never discard a prepared
    delta and then apply its successor to an older pose.
    """

    def __init__(self, pipeline: PerceptionPipeline):
        from concurrent.futures import ThreadPoolExecutor

        self.pipeline = pipeline
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flash-frame-prepare")
        self.pending = None
        self.closed = False

    def _prepare_next(self, next_frame):
        frame = next_frame()
        if frame is None:
            return None
        return frame, self.pipeline.backend.prepare(frame)

    def _finish(self, item):
        if item is None:
            return None
        frame, prepared = item
        estimates = self.pipeline.backend.finish(prepared)
        return frame, self.pipeline.result(frame, estimates)

    def process_next(self, next_frame):
        if self.closed:
            raise RuntimeError("perception runner is closed")
        if self.pending is not None:
            pending, self.pending = self.pending, None
            item = pending.result()
            if item is None:
                return None
        elif getattr(self.pipeline.backend, "can_pipeline", False):
            item = self._prepare_next(next_frame)
            if item is None:
                return None
        else:
            frame = next_frame()
            return None if frame is None else (frame, self.pipeline.process(frame))
        # Only matching/segmentation can run ahead. Pose composition, reference
        # registration, symmetry and result validation remain ordered here.
        self.pending = self.pool.submit(self._prepare_next, next_frame)
        return self._finish(item)

    def drain(self):
        """Finish the single pending frame before a control changes track state."""
        if self.pending is None:
            return None
        pending, self.pending = self.pending, None
        return self._finish(pending.result())

    def close(self, *, commit_pending=True):
        if self.closed:
            return
        try:
            if commit_pending:
                self.drain()
        finally:
            self.pool.shutdown(wait=True)
            close = getattr(self.pipeline.backend, "close", None)
            if close is not None:
                close()
            self.pending = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.close(commit_pending=exc_type is None)
