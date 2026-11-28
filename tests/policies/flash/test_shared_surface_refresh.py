"""Live defaults execute replacement and latest-reliable checks without patches."""

import pytest

from lerobot.gui.api.flash_tracking import TrackingStart
from lerobot.policies.flash.experimental_surface_session import GrowingSurfaceConfig, GrowingSurfaceSession
from lerobot.policies.flash.surface_refresh import RefreshingSurfaceStore
from lerobot.policies.flash.vision_config import VisionPipelineConfig


def test_live_start_uses_shared_refresh_store_and_boundary_settings():
    settings = TrackingStart().model_dump()["pipeline_settings"]
    vision = VisionPipelineConfig.from_dict(settings["config"])
    session = GrowingSurfaceSession(vision.shape, GrowingSurfaceConfig(**settings["surface_experiment"]))
    assert isinstance(session.store, RefreshingSurfaceStore)
    assert vision.observation.mask_margin_px == 4
    assert vision.observation.outer_mask_margin_px == 2


def test_discarded_disabled_averaging_setting_cannot_break_saved_live_start():
    from copy import deepcopy

    from lerobot.policies.flash.surface_session_config import defaults

    settings = defaults()
    settings["surface_experiment"]["refinement_max_weight"] = 0
    before = deepcopy(settings)
    request = TrackingStart(pipeline_settings=settings)
    assert "refinement_max_weight" not in request.pipeline_settings["surface_experiment"]
    assert request.pipeline_settings["surface_experiment"]["refresh_observed_surface"]
    assert settings == before
    settings["surface_experiment"]["refinement_max_weight"] = 8
    with pytest.raises(ValueError, match="averaging experiment was removed"):
        TrackingStart(pipeline_settings=settings)
