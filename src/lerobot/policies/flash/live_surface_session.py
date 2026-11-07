"""Live mask acquisition around the same frame session used by experiments."""

import time

import numpy as np

from .experimental_surface_session import GrowingSurfaceConfig
from .shared_image_tracking import SharedMatchingConfig
from .surface_frame_session import SurfaceFrameSession
from .surface_group_hints import GroupConfig
from .surface_session_config import validate
from .vision_pipeline import VisionPipelineConfig

LIVE_METHOD = "group_growing"


class LiveSurfaceSession:
    def __init__(self, settings, mask_source, model=None, kernels=None, search_pool=None):
        self.settings = validate(settings)
        self.method, self.mask_source = LIVE_METHOD, mask_source
        self.engine = SurfaceFrameSession(
            tuple(mask_source.prompts),
            model,
            VisionPipelineConfig.from_dict(self.settings["config"]),
            kernels,
            search_pool,
            single_method=LIVE_METHOD,
            record_correspondences=True,
            surface_experiment=GrowingSurfaceConfig(**self.settings["surface_experiment"]),
            group_experiment=GroupConfig.from_dict(self.settings["group_experiment"])
            if self.settings["group_experiment"] is not None
            else None,
            shared_matching_config=SharedMatchingConfig.from_dict(self.settings["shared_matching"])
            if self.settings["shared_matching"] is not None
            else None,
        )

    def step(self, frame):
        start = time.perf_counter()
        measured = self.mask_source.masks(frame)
        masks = {role: measured.get(role, np.zeros(frame.depth.shape, bool)) for role in self.engine.sessions}
        elapsed = (time.perf_counter() - start) * 1000
        row, matches = self.engine.step(frame, masks, mask_ms=elapsed)
        return row, matches, masks

    def close(self):
        self.engine.close()
