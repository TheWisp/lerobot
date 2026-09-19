"""The page's compositor and the library's are one definition (design: R6).

The JPEG path shows pixels composited by overlays/effects.py on the server;
Low Bandwidth composites in the page. Two languages cannot share the code, so
they share the definition and this test enforces it: the same frame, masks,
recipe and noise through both, compared pixel by pixel, and they must agree
exactly -- seams, blur and all. The library composites in 8-bit fixed point
(cv2's Gaussian and blendLinear), so the page mirrors that arithmetic rather
than approximating it in float; a tolerance here would hide the day one of
them changes its rounding.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

RUNNER = Path(__file__).with_name("mask_composite_runner.js")
W, H = 64, 48


def _frame() -> np.ndarray:
    rng = np.random.default_rng(3)
    return rng.integers(0, 256, (H, W, 3), np.uint8)


def _disc(cy, cx, r) -> np.ndarray:
    ys, xs = np.mgrid[:H, :W]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r


def _python(rgb, masks, treatments, background, textures, feather):
    from lerobot.overlays.effects import build_and_sample_regions, composite_regions

    cache = {}
    regions, sampled = build_and_sample_regions(
        masks, treatments, background, H, W, np.random.default_rng(0), cache, feather=feather
    )
    # The page draws the noise it is given; hand the same texture to both sides.
    for i, name in enumerate(["__bg__", *masks]):
        tex = textures.get(name)
        if tex is not None:
            sampled[i] = {"bg": np.asarray(tex, np.uint8).reshape(H, W, 3)}
    return composite_regions(rgb, regions, sampled)


def _page(rgb, masks, treatments, background, textures, feather):
    req = {
        "w": W,
        "h": H,
        "rgb": rgb.reshape(-1).tolist(),
        "masks": {n: m.astype(np.uint8).reshape(-1).tolist() for n, m in masks.items()},
        "treatments": treatments,
        "background": background,
        "textures": {
            n: (None if t is None else np.asarray(t, np.uint8).reshape(-1).tolist())
            for n, t in textures.items()
        },
        "feather": feather,
    }
    out = subprocess.run(
        ["node", str(RUNNER)], input=json.dumps(req), capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-800:]
    return np.asarray(json.loads(out.stdout)["rgb"], np.uint8).reshape(H, W, 3)


CASES = {
    "tint_default": ({"ball": {"key": "tint", "params": {}}}, {"key": "none", "params": {}}),
    "tint_custom": (
        {"ball": {"key": "tint", "params": {"color": [255, 0, 0], "strength": 0.3}}},
        {"key": "none", "params": {}},
    ),
    "solid": ({"ball": {"key": "solid", "params": {"color": [0, 200, 0]}}}, {"key": "none", "params": {}}),
    "random": ({"ball": {"key": "random", "params": {}}}, {"key": "none", "params": {}}),
    "blur": ({"ball": {"key": "blur", "params": {"strength": 3}}}, {"key": "none", "params": {}}),
    "none": ({"ball": {"key": "none", "params": {}}}, {"key": "none", "params": {}}),
    "background_solid_object_tint": (
        {"ball": {"key": "tint", "params": {}}},
        {"key": "solid", "params": {"color": [1, 2, 3]}},
    ),
    "background_random": ({"ball": {"key": "none", "params": {}}}, {"key": "random", "params": {}}),
    "two_objects_overlap_smallest_wins": (
        {
            "big": {"key": "solid", "params": {"color": [200, 0, 0]}},
            "small": {"key": "solid", "params": {"color": [0, 0, 200]}},
        },
        {"key": "none", "params": {}},
    ),
}


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
@pytest.mark.parametrize("feather", [0, 5], ids=["hard", "feathered"])
@pytest.mark.parametrize("case", list(CASES))
def test_the_page_composites_as_the_library_does(case, feather):
    treatments, background = CASES[case]
    rgb = _frame()
    if case == "two_objects_overlap_smallest_wins":
        masks = {"big": _disc(24, 30, 16), "small": _disc(24, 38, 7)}  # the small disc inside the big one
    else:
        masks = {"ball": _disc(24, 30, 12)}
    rng = np.random.default_rng(11)
    textures = {n: rng.integers(0, 256, (H, W, 3), np.uint8) for n in ["__bg__", *masks]}
    ours = _python(rgb, masks, treatments, background, textures, feather)
    theirs = _page(rgb, masks, treatments, background, textures, feather)
    diff = np.abs(ours.astype(int) - theirs.astype(int))
    assert diff.max() == 0, (
        f"{case} feather={feather}: max diff {diff.max()} on {(diff > 0).sum()} pixels, first at {np.argwhere(diff == diff.max())[0]}"
    )
    # The complement: the composite is not the input, unless every treatment is none.
    if any(t["key"] != "none" for t in treatments.values()) or background["key"] != "none":
        assert not np.array_equal(theirs, rgb), f"{case}: the page changed nothing"
