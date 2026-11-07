# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""The depth display mapping: what the page may ask for, how it is keyed, and
what the renderer does with it. The stored depth is never part of this."""

from types import SimpleNamespace

import numpy as np
import pytest

from lerobot.gui.depth_display import (
    AUTO,
    DepthMapping,
    depth_range_hint_m,
    episode_auto_range_mm,
    mapping_for,
    parse_depth_spec,
    parse_mapping,
    render_kwargs,
)
from lerobot.gui.frame_cache import encode_frame_to_jpeg
from lerobot.utils.depth_preview import auto_range, depth_to_rgb_preview, nearness

CAM = "observation.images.top_depth"
H, W = 720, 1280


class TestSpecs:
    def test_modes_parse(self):
        assert parse_mapping("auto") is AUTO
        assert parse_mapping("fixed:0.2:0.9") == DepthMapping("fixed", 0.2, 0.9)

    @pytest.mark.parametrize(
        "bad", ["bogus", "log:0.2:1", "auto:1:2", "fixed:0.9:0.2", "fixed:0.5", "fixed:a:b"]
    )
    def test_bad_specs_are_refused(self, bad):
        with pytest.raises(ValueError):
            parse_mapping(bad)

    def test_per_camera_and_default(self):
        specs = parse_depth_spec(f"{CAM}=fixed:0.2:0.9;fixed:0.1:2")
        assert mapping_for(specs, CAM) == DepthMapping("fixed", 0.2, 0.9)
        assert mapping_for(specs, "other") == DepthMapping("fixed", 0.1, 2.0)
        assert mapping_for(parse_depth_spec(""), CAM) is AUTO

    def test_variant_keys_only_non_auto(self):
        assert AUTO.variant == ""
        assert DepthMapping("fixed", 0.2, 0.9).variant == "dfixed_0.2_0.9"
        assert DepthMapping("fixed", 0.2, 0.5).variant != DepthMapping("fixed", 0.2, 0.9).variant


def _dataset(episode_stats=True, dataset_stats=True, unit="mm", legacy=False):
    """The slice of a LeRobotDataset the mapping reads: features, depth keys,
    per-episode statistics in the record-time unit, and dataset-wide ones
    already rescaled to the read unit, millimetres, as the loader leaves them."""
    scale = 1 / 1000 if unit == "m" else 1
    info = {"is_depth_map": True, "depth_unit": unit}
    feature = {"dtype": "video", "shape": [H, W, 1], **({"video_info": info} if legacy else {"info": info})}
    ep = {"length": 10}
    if episode_stats:
        ep[f"stats/{CAM}/q01"] = np.array([[[230.0 * scale]]], dtype=object)
        ep[f"stats/{CAM}/q99"] = np.array([[[822.0 * scale]]], dtype=object)
    stats = {CAM: {"q01": [[[250.0]]], "q99": [[[900.0]]]}} if dataset_stats else {}
    meta = SimpleNamespace(features={CAM: feature}, depth_keys=[CAM], episodes=[ep], stats=stats)
    return SimpleNamespace(meta=meta, _depth_output_unit="mm")


class TestRanges:
    def test_auto_uses_the_episode_percentiles(self):
        assert episode_auto_range_mm(_dataset(), 0, CAM) == (230.0, 822.0)
        assert render_kwargs(_dataset(), 0, CAM, AUTO) == {
            "depth_lo": 230.0, "depth_hi": 822.0, "depth_invalid_below": 10.0,
        }  # fmt: skip

    def test_the_quantizer_floor_marks_no_measurement(self):
        """A stored zero dequantizes to depth_min, so a read-back pixel at the
        floor is drawn as no measurement, not as the nearest thing in view."""
        kwargs = render_kwargs(_dataset(), 0, CAM, AUTO)
        assert kwargs["depth_invalid_below"] == 10.0  # the default depth_min, 0.01 m
        floor = np.full((H, W, 1), 10.0, np.float32)
        floor[:, W // 2 :] = 500.0
        rgb = depth_to_rgb_preview(floor, lo=200, hi=900, invalid_below=kwargs["depth_invalid_below"])
        assert np.all(rgb[:, : W // 2] == 0) and np.any(rgb[:, W // 2 :] > 0)

    def test_a_percentile_at_a_hole_starts_the_range_at_the_floor(self):
        """The recorded statistics count holes (stored zeros): a lit rig
        recording with 2% of them had q01 = -1e-10, which drew the palette
        from 0 m and showed "-0.0". The range starts where a hole reads back,
        the quantizer's floor, and the seed never goes below zero."""
        ds = _dataset()
        ds.meta.episodes[0][f"stats/{CAM}/q01"] = np.array([[[-1e-10]]], dtype=object)
        assert episode_auto_range_mm(ds, 0, CAM) == (10.0, 822.0)
        assert depth_range_hint_m({CAM: {"q01": [[[-1e-10]]], "q99": [[[900.0]]]}}, CAM) == [0.0, 0.9]

    def test_auto_falls_back_to_dataset_then_frame(self):
        assert episode_auto_range_mm(_dataset(episode_stats=False), 0, CAM) == (250.0, 900.0)
        assert episode_auto_range_mm(_dataset(False, False), 0, CAM) is None
        assert render_kwargs(_dataset(False, False), 0, CAM, AUTO) == {"depth_invalid_below": 10.0}

    def test_metre_recordings_scale_only_the_episode_statistics(self):
        """Per-episode statistics are in the recorded unit; the dataset-wide
        ones were rescaled to millimetres at load and must not be scaled again."""
        ds = _dataset(unit="m")
        lo, hi = episode_auto_range_mm(ds, 0, CAM)
        assert (round(lo), round(hi)) == (230, 822)
        assert episode_auto_range_mm(_dataset(unit="m", episode_stats=False), 0, CAM) == (250.0, 900.0)
        assert depth_range_hint_m(ds.meta.stats, CAM, "mm") == [0.25, 0.9]

    def test_legacy_depth_spelling_still_has_a_floor(self):
        assert render_kwargs(_dataset(legacy=True), 0, CAM, AUTO)["depth_invalid_below"] == 10.0

    def test_fixed_is_in_metres_from_the_page(self):
        assert render_kwargs(_dataset(), 0, CAM, DepthMapping("fixed", 0.2, 0.9)) == {
            "depth_lo": 200.0, "depth_hi": 900.0, "depth_invalid_below": 10.0,
        }  # fmt: skip

    def test_a_colour_camera_gets_no_depth_arguments(self):
        assert render_kwargs(_dataset(), 0, "observation.images.top", DepthMapping("fixed", 0.2, 0.9)) == {}

    def test_hint_is_the_dataset_percentiles_in_metres(self):
        ds = _dataset()
        assert depth_range_hint_m(ds.meta.stats, CAM, "mm") == [0.25, 0.9]
        assert depth_range_hint_m({CAM: {"q01": [[[0.25]]], "q99": [[[0.9]]]}}, CAM, "m") == [0.25, 0.9]
        assert depth_range_hint_m({}, CAM, "mm") is None


def _ramp() -> np.ndarray:
    """Depth rising left to right from 200 to 1000 mm, a hole in one corner."""
    d = np.tile(np.linspace(200, 1000, W, dtype=np.float32), (H, 1))
    d[:10, :10] = 0
    return d[..., None]


def _turbo(code: int) -> np.ndarray:
    """The palette's colour for a nearness code, as the renderer produces it."""
    import cv2

    return cv2.applyColorMap(np.array([[code]], np.uint8), cv2.COLORMAP_TURBO)[0, 0, ::-1]


class TestRenderer:
    def test_auto_range_is_the_valid_percentiles(self):
        lo, hi = auto_range(_ramp())
        assert 200 <= lo < 220 and 980 < hi <= 1000  # the 1st and 99th percentile of the ramp

    def test_fixed_range_saturates_outside_it(self):
        n = nearness(_ramp()[..., 0], lo=900, hi=1000)
        assert n[H // 2, 5] == 1.0 and n[H // 2, W - 1] == 0.0
        assert 0.3 < n[H // 2, int(0.95 * (W - 1))] < 0.5  # about 960 mm: inside the range
        rgb = depth_to_rgb_preview(_ramp(), lo=900, hi=1000)
        assert np.array_equal(rgb[H // 2, 5], _turbo(255))  # the warm end of the palette
        assert np.all(rgb[:10, :10] == 0)  # no measurement stays black

    def test_funnel_takes_the_mapping(self):
        import torch

        depth = torch.from_numpy(_ramp()).permute(2, 0, 1)
        auto = encode_frame_to_jpeg(depth)
        fixed = encode_frame_to_jpeg(depth, depth_lo=900, depth_hi=1000)
        assert auto[:2] == b"\xff\xd8" and fixed[:2] == b"\xff\xd8"
        assert auto != fixed
