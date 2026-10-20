"""Short pose gaps without pretending a missed match is a new measurement.

The wrapped backend must derive every non-lost pose from the current RGB-D
frame. A carried pose keeps its original source timestamp, so the planner's
freshness check continues to age it out even if more frames are missed.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .perception import ObjectReference, PoseBackend, RGBDFrame
from .trajectory import Evidence, PoseEstimate


class PoseGapBackend:
    """Carry the last data-derived pose across bounded matcher dropouts."""

    def __init__(self, inner: PoseBackend, max_gap_s: float) -> None:
        if not np.isfinite(max_gap_s) or max_gap_s <= 0:
            raise ValueError("pose gap needs a positive finite maximum age")
        self.inner = inner
        self.max_gap_s = max_gap_s
        self._anchor: dict[str, PoseEstimate] = {}

    @property
    def last_failure(self):
        return getattr(self.inner, "last_failure", {})

    def initialize(self, references: tuple[ObjectReference, ...]) -> None:
        self._anchor.clear()
        self.inner.initialize(references)

    def update(self, frame: RGBDFrame) -> tuple[PoseEstimate, ...]:
        out = []
        for estimate in self.inner.update(frame):
            if estimate.pose is not None and estimate.evidence != Evidence.LOST:
                self._anchor[estimate.role] = estimate
                out.append(estimate)
                continue
            anchor = self._anchor.get(estimate.role)
            if (
                anchor is not None
                and estimate.reference_version == anchor.reference_version
                and 0 <= estimate.source_time_s - anchor.source_time_s <= self.max_gap_s
            ):
                out.append(
                    replace(
                        anchor,
                        available_time_s=estimate.available_time_s,
                        evidence=Evidence.PREDICTED,
                    )
                )
            else:
                out.append(estimate)
        return tuple(out)
