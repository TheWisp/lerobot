"""Identical observations must have identical meanings in every consumer."""

import pytest

from lerobot.policies.flash.vision_config import (
    DetectionCorrectionConfig,
    VisionPipelineConfig,
)


def test_config_round_trip_and_invalid_values_are_common_to_all_consumers():
    config = VisionPipelineConfig.from_dict({"registration_threshold_m": 0.007, "backend": "surface"})
    assert VisionPipelineConfig.from_dict(config.to_dict()) == config
    assert config.registration_threshold_m == 0.007
    for change in (
        {"backend": "typo"},
        {"motion": {"grid_step_px": 2.5}},
        {"shape": {"depth_tolerance_m": float("nan")}},
        {"observation": {"mask_margin_px": -1}},
        {"motion": {"grid_step_px": 0}},
        {"parallel_surface_objects": 1},
        {"detection_correction": {"runtime_interval_frames": 0}},
        {"detection_correction": {"min_mask_overlap": 1.1}},
    ):
        with pytest.raises(ValueError):
            VisionPipelineConfig.from_dict(change)


def test_detection_correction_has_separate_offline_and_runtime_cadences():
    config = VisionPipelineConfig(
        detection_correction=DetectionCorrectionConfig(
            enabled=True,
            offline_interval_frames=12,
            runtime_interval_frames=48,
        )
    )
    assert VisionPipelineConfig.from_dict(config.to_dict()) == config
    assert config.detection_correction.control(offline=True)["interval_frames"] == 12
    assert config.detection_correction.control(offline=False)["interval_frames"] == 48
