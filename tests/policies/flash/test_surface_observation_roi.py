"""Bound geometry work to the mask without changing measured surface evidence."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from lerobot.policies.flash import surface_observation as surface


def assert_full_frame_equivalent(depth, mask, intrinsics, config, monkeypatch):
    cropped = surface.observation(depth, mask, intrinsics, config)
    with monkeypatch.context() as patch:
        # Reconstruct the previous execution: all image coordinates are lifted
        # before the mask chooses points. Compare every output, not just XYZ.
        patch.setattr(
            surface, "_surface_region", lambda _: (slice(0, mask.shape[0]), slice(0, mask.shape[1]))
        )
        full = surface.observation(depth, mask, intrinsics, config)
    assert cropped.keys() == full.keys()
    for key in full:
        np.testing.assert_array_equal(cropped[key], full[key], err_msg=key)
    return cropped


@pytest.mark.parametrize("margin", [0, 2, 5])
@pytest.mark.parametrize(
    "bounds",
    [
        (9, 7, 40, 33),  # Offset does not align with validation tiles.
        (0, 0, 29, 31),
        (41, 37, 70, 60),
        (0, 0, 70, 60),
        (19, 17, 20, 18),
        (0, 0, 0, 0),
    ],
)
def test_crop_preserves_geometry_filters_and_full_image_outputs(bounds, margin, monkeypatch):
    y, x = np.indices((60, 70))
    depth = (0.6 + x * 0.0001 + y * 0.0002).astype(np.float32)
    depth[19:21, 23:26] = 0
    depth[27:29, 30:32] += 0.1
    # Unrelated depth affects the bilateral filter's global range. The cropped
    # implementation must preserve its full-frame input, including this range.
    depth[0, 69] = 3
    mask = np.zeros(depth.shape, bool)
    x0, y0, x1, y1 = bounds
    mask[y0:y1, x0:x1] = True
    mask[23:25, 18:20] = False
    k = np.array([[235, 0, 33.25], [0, 231, 28.5], [0, 0, 1]], float)
    config = asdict(surface.SurfaceObservationConfig(mask_margin_px=margin))
    result = assert_full_frame_equivalent(depth, mask, k, config, monkeypatch)
    if x1 - x0 > 20:
        assert len(result["points"]) > 10


def test_recorded_held_object_surface_is_unchanged(monkeypatch):
    with np.load(Path(__file__).with_name("fixtures") / "dowel_held_frame_315.npz") as data:
        depth = data["depth"].astype(np.float32) * 0.001
        mask = data["mask"].astype(bool)
        k = data["intrinsics"]
    result = assert_full_frame_equivalent(
        depth, mask, k, asdict(surface.SurfaceObservationConfig()), monkeypatch
    )
    assert len(result["points"]) > 20


def test_disconnected_mask_is_not_truncated_to_one_image_component(monkeypatch):
    mask = np.zeros((60, 70), bool)
    mask[10:20, 10:20] = True
    mask[39:49, 45:55] = True
    config = asdict(surface.SurfaceObservationConfig(component_gap_m=None))
    k = np.array([[235, 0, 33.25], [0, 231, 28.5], [0, 0, 1]], float)
    result = assert_full_frame_equivalent(np.full(mask.shape, 0.6, np.float32), mask, k, config, monkeypatch)
    assert (result["pixels"][:, 0] < 20).any()
    assert (result["pixels"][:, 0] > 45).any()


def test_small_object_avoids_full_image_geometry_allocations(monkeypatch):
    shapes = []
    original = surface.unproject

    def measured_unproject(depth, *args, **kwargs):
        shapes.append(depth.shape)
        return original(depth, *args, **kwargs)

    monkeypatch.setattr(surface, "unproject", measured_unproject)
    mask = np.zeros((720, 1280), bool)
    mask[337:464, 711:869] = True
    k = np.array([[600, 0, 640], [0, 600, 360], [0, 0, 1]], float)
    result = surface.observation(
        np.full(mask.shape, 0.6, np.float32), mask, k, asdict(surface.SurfaceObservationConfig())
    )
    assert shapes and all(np.prod(shape) < mask.size / 10 for shape in shapes)
    assert len(result["points"]) > 100
    px = result["pixels"]
    assert np.all(mask[px[:, 1], px[:, 0]])
    np.testing.assert_allclose(result["points"][:, 0], (px[:, 0] - 640) * np.float32(0.6) / 600)
